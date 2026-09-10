---
author: xiangel
pubDatetime: 2026-09-03T17:30:00Z
modDatetime: 2026-09-09T07:50:00Z
title: 从 Transformer 出发来看推理系统
slug: from-causal-lm-to-inference-system
featured: true
draft: false
tags:
  - 大模型推理系统
  - Transformer
  - 生成式模型
description: 从一次最朴素的 generate 循环讲起：先看清单条请求里发生了什么，再说明为什么一上并发就非得有 KV 管理、调度和 prefill/decode 分工。系列开篇，不绑任何推理引擎。
---

如果你用 HuggingFace 跑通过 `model.generate()`，大概知道流程：输入一段 prompt，模型一个词一个词往外吐。代码不长，真正上线后却长出一堆模块——KV 管理、调度器、prefill/decode 拆分、采样、多卡切分……

这篇只回答一件事：**这些模块，分别是在解决什么问题？**

顺序按「先单用户、后多用户」走。先把一条请求里发生了什么讲清楚，再谈并发时哪里会崩。后面几篇会分别展开 KV Cache（[第二篇](/posts/kv-cache-paged-attention-and-prefix-caching/)）和调度（[第三篇](/posts/llm-inference-scheduling-single-node/)），这里只做地图，不展开实现细节。

## 生成在算什么

现在的大模型，主体是一叠 **decoder-only** 层：没有单独的 encoder，prompt 和回答在同一条序列里，从左往右延长。训练目标就是 next-token prediction——给定前缀，猜下一个 token。

生成时，因果掩码保证位置 _i_ 只能看左边已经存在的 token。所以聊天、补全、续写，底层都是同一条循环：看已有前缀 → 算 logits → 抽一个词 → 接到末尾 → 再来。

一层 decoder 可以粗看成：

![一层 decoder：归一化、因果自注意力、前馈，以及读写历史 Key/Value](/assets/llm-inference-01/fig2-decoder-block.svg)

和「系统」直接相关的就三点：

- **因果掩码**：历史 token 的 Key/Value 算过就不用重算。
- **自回归**：训练可以整段并行；线上必须逐步吐，没有下一个 token 就没有下一轮。
- **采样**：最后一层输出 logits，按温度、top-p 等规则抽 token——发生在每步 forward 之后，但仍在关键路径上。

RMSNorm、RoPE、GQA 这些改的是「每步有多贵」，不改变「必须一步步生成」这个前提。

## 一条请求：朴素循环与 KV Cache

最直白的写法：

```python
def naive_generate(model, tokens, n):
    for _ in range(n):
        logits = model(tokens)       # 每步把整段再送进去
        nxt = sample(logits[:, -1])
        tokens = torch.cat([tokens, nxt], dim=1)
    return tokens
```

逻辑没问题。浪费在于：第 _t_ 步又把前 _t−1_ 个位置的注意力重算了一遍。序列一长，重复计算按平方涨。

因果掩码本来允许你**留下**已经算好的 Key/Value。改成先处理 prompt、再逐步 append：

```python
def cached_generate(model, tokens, n):
    out = []
    logits, kv = model(tokens, kv=None)
    nxt = sample(logits[:, -1])
    out.append(nxt)
    for _ in range(n - 1):
        logits, kv = model(nxt, kv=kv)
        nxt = sample(logits[:, -1])
        out.append(nxt)
    return out
```

![左边每步把整段再算一遍；右边先处理问题，再只追加新词](/assets/llm-inference-01/fig3-naive-vs-cached.svg)

GPT-2 源码里的 `past`，就是这个 cache。**到这一步，还只是「一条请求」的故事**——显存里多了一块随长度变长的 KV，计算从重复变成增量。

## 同一条请求里，其实有两种 forward

上面两段代码，第一行和循环里的每一行，函数签名一样，硬件行为却不同：

|          | Prefill                 | Decode                                |
| -------- | ----------------------- | ------------------------------------- |
| 输入     | 整段 prompt（可能很长） | 每次 1 个新 token                     |
| 算力     | 矩阵大，GPU 容易喂饱    | 矩阵极小，算力常空转                  |
| 访存     | 要写大量 KV             | 每步几乎要把全部权重 + 已有 KV 读一遍 |
| 用户感知 | 等**第一个字**出来      | 字与字之间的**间隔**                  |

前者论文里常叫 TTFT，后者叫 TPOT/TBT。名字不用背，记住体感上的两段就够。

![一次请求裂成处理 prompt 与逐步吐词](/assets/llm-inference-01/fig1-request-timeline.svg)

粗算一下 decode 为什么偏「访存」：7B 模型 FP16 权重大约 14 GB，显存带宽 2 TB/s 量级时，单请求 decode 往往也就一百多 token/s——算力还没打满，卡在搬权重和 KV 上。这条线后面会反复出现。

**重要**：prefill 和 decode 的分界，在「单请求 + KV Cache」阶段就已经存在了。不是上线之后才发明的概念。

## 从一条请求到在线服务：哪里开始不够用

脚本里 `generate` 循环跑完就结束。服务形态是：很多人同时连上来，请求长短不一，有人只要一句话，有人扔一篇 PDF 当 prompt。

这时 naive 方案会在三处撞墙。

### 墙 1：一次请求 ≠ 一次 forward

生成 100 个 token，至少是 1 次 prefill + 约 99 次 decode。图像分类那种任务，一次 forward 就交差，可以整批一起跑、一起返回。生成不行——批次里最短的那条写完了，还得等最长的那条，槽位空不出来，新人只能在门外等。

Orca（OSDI 2022）的核心观察是：调度粒度应该降到 **「每一次 forward」**，而不是「整次请求」。谁写完了立刻腾位，新人马上补进来。这就是后面 continuous batching 的起点，[第三篇](/posts/llm-inference-scheduling-single-node/) 会细讲。

### 墙 2：KV 随人、随长度涨，显存先于算力满

KV Cache 省的是重复计算，代价是显存。每个 token 每层每个 KV 头都要存一份 Key 和 Value：

```text
每 token KV 字节 ≈ 2 × 层数 × KV头数 × 头维度 × dtype 字节数
```

| 模型               | 形状      | 每 token | 4K 上下文 |
| ------------------ | --------- | -------- | --------- |
| 7B（MHA）          | 32×32×128 | 512 KB   | ~2 GB     |
| 8B（GQA, 8 KV 头） | 32×8×128  | 128 KB   | ~0.5 GB   |

权重加载一次就占住一大块（7B FP16 约 14 GB），KV 则随**并发数 × 序列长度**一直涨。80 GB 的卡，经常出现「GPU 利用率不高，但再加人就 OOM」——瓶颈在 KV，不在 FLOPs。

GQA、MLA 从模型侧把 KV 压小，但线性增长躲不掉。怎么分配、回收、让相同前缀共用一份，是 [第二篇](/posts/kv-cache-paged-attention-and-prefix-caching/) 的主线。

### 墙 3：prefill 和 decode 硬挤同一轮 forward，会互相拖

单请求内部已经是两种 forward。多用户时，如果把「新来的长 prompt prefill」和「正在吐字的 decode」塞进同一轮、不做区分，正在交互的用户会被拖慢——本来几十毫秒一步，可能要等别人整段 prompt 算完。

Sarathi-Serve、DistServe 等工作的动机都从这里来：要么把长 prefill 切块（chunked prefill），要么干脆拆到不同设备（PD 分离）。具体做法留到调度篇和后续，这里只记：**两种 forward 的忙闲相反，混跑就要付干扰成本。**

## 模块从哪来：一张简图

把上面三块墙和对应方向画在一起：

![从「只往右生成」长出调度、显存管理、两段安排、真正去算](/assets/llm-inference-01/fig4-constraint-to-modules.svg)

读任何推理引擎（vLLM、TensorRT-LLM、SGLang……），可以先问四个问题：

1. **KV 放哪、怎么回收、前缀能不能共用？** → 显存管理 / PagedAttention / Prefix Caching
2. **每一轮 forward 选哪些请求？** → 调度 / continuous batching
3. **prefill 和 decode 同卡混跑还是拆开？** → chunked prefill、PD 分离
4. **权重怎么切到多卡、请求从哪进出？** → 并行策略 + serving 层

名字会换，问题差不多。第二、三篇分别啃 1 和 2；PD 分离和分布式调度以后单独开一篇。

## 系列后续

| 篇  | 主题                            | 接哪条线   |
| --- | ------------------------------- | ---------- |
| 01  | 从 Transformer 出发来看推理系统 | 本篇       |
| 02  | KV Cache：分页与前缀共用        | 墙 2       |
| 03  | 单机调度                        | 墙 1、墙 3 |
| 04+ | PD 分离、多卡、采样与算子       | 待写       |

## 参考

1. Vaswani et al., [_Attention Is All You Need_](https://arxiv.org/abs/1706.03762), 2017.
2. Radford et al., [_Language Models are Unsupervised Multitask Learners_](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)（GPT-2，`past` 即 KV 缓存），2019.
3. Yu et al., [_Orca: A Distributed Serving System for Transformer-Based Generative Models_](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022.
4. Kwon et al., [_Efficient Memory Management for Large Language Model Serving with PagedAttention_](https://arxiv.org/abs/2309.06180), SOSP 2023.
5. Agrawal et al., [_Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve_](https://arxiv.org/abs/2403.02310), OSDI 2024.
6. Zhong et al., [_DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving_](https://arxiv.org/abs/2401.09670), OSDI 2024.
