---
author: xiangel
pubDatetime: 2026-09-03T17:30:00Z
title: 先别谈 vLLM：从因果自回归看推理系统的模块从哪来
slug: from-causal-lm-to-inference-system
featured: true
draft: false
tags:
  - 大模型推理系统
  - Transformer
  - LLM Serving
description: 生成式 Transformer 的计算性质，已经决定了推理系统必须长什么样。本文从因果自回归抽出四条约束，映射成调度、KV、两相与执行四件套。
---

会写 `model.generate`，并不等于理解大模型推理系统。打开 vLLM 或 SGLang 的文档，名词很多：PagedAttention、continuous batching、chunked prefill、prefix cache、P/D disaggregation。它们看起来像一份优化清单。更准确的读法是：这些模块不是后来「加」上去的，而是生成式 Transformer 的计算性质逼出来的。

同一套权重、同一次请求，prompt 阶段 GPU 算力往往打满，生成阶段却在等显存。这不是实现粗糙，而是**因果自回归把一次前向劈成了两种作业**。本篇是系列第一篇，只做一件事：从架构抽出四条推理约束，映射成系统模块地图。不写 CUDA kernel，不比引擎快慢。读完后，你应能把上面那些名词反推回 Transformer 的某条约束，并判断它在修哪一类瓶颈。

## 一次请求，两种时间

一次用户请求在服务端不是「跑一遍模型」。它先把整段 prompt 编成内部状态，再一个 token 一个 token 往外吐。前一段叫 **prefill**，后一段叫 **decode**：

![一次请求裂成 Prefill（TTFT）与 Decode（TPOT）](/assets/llm-inference-01/fig1-request-timeline.svg)

两个常用指标正好对应这两段：

- **TTFT**（time to first token）：用户等到第一个输出 token 的时间，主要由 prefill 决定。
- **TPOT**（time per output token）：之后每个 token 的间隔，主要由 decode 决定。有的论文写成 TBT（time between tokens），含义接近。

为什么同一套层会表现出两种脾气？看算术强度。prefill 对整段 prompt 并行，每个权重会被很多 token 复用，矩阵乘法能喂饱 Tensor Core，整体偏 **compute-bound**。decode 每步只进 1 个 token，却几乎要把全部权重和历史 KV 从 HBM 读一遍，算得少、搬得多，偏 **memory-bound**。单条 7B 量级模型，decode 的理论上限大致是「显存带宽 / 权重大小」：带宽 2 TB/s、权重 14 GB 时，单流大约一百多个 token/s。batch 可以把同一次权重读取摊到多条请求上，直到 KV 把显存吃满。

后面所有「调度」「切块」「分离」，都是在给这两种作业找共存方式。若把它们当成同一次 `forward`，TTFT 和 TPOT 会互相绑架。

## 生成式架构：只保留推理用得到的骨架

2017 年的 Transformer 是 encoder + decoder + cross-attention，面向机器翻译：encoder 双向看完整输入，decoder 一边看已生成的目标词，一边用 cross-attention 去问 encoder。GPT 路线做了两步简化：丢掉整个 encoder，也丢掉 cross-attention。prompt 和生成共用一条**因果栈**。训练目标变成纯粹的 next-token prediction——给定前缀，预测下一个 token。

这对推理的直接后果是：没有单独的「理解模块」。系统不能先 encode 再 decode；prefill 和 decode 是同一组层、同一次因果 attention 的两个阶段。prompt 只是这条序列左边的前缀，生成是把前缀往右延长。

现代 decoder-only 块可以画成：

![Decoder-only 块：RMSNorm、因果自注意力、FFN，以及向 KV Cache 读写](/assets/llm-inference-01/fig2-decoder-block.svg)

和推理相关的性质只有三条，值得反复钉死：

1. **因果掩码。** 位置 `i` 只能看更早的 token。因此历史的 Key/Value 一旦算过，下一步不必重算，只要存下来。
2. **自回归。** 训练时可一次算完整个序列的 next-token loss（每个位置的预测只依赖左边，可以并行）。推理必须逐步：每步一次完整前向，才能得到下一个 token。
3. **输出。** 最后一层投影到词表 logits，按温度 / top-p 等规则 sample 一个 token，拼回序列，再进入下一步。采样发生在模型外面，但仍占关键路径。

模型结构这些年也在改。本篇只记账、不展开训练故事：

| 改动 | 对推理的影响 |
| --- | --- |
| Pre-Norm / RMSNorm | 每层少做一点归约，带宽友好，但仍是小头 |
| RoPE | 位置进 Q/K，和长上下文、cache 布局绑在一起 |
| GQA / MQA | 减少 KV head 数，单位 token 显存和带宽下降 |
| SwiGLU FFN | 改变 MLP 计算量，不改变「逐步生成」这件事 |
| MoE | 专家路由，执行器要考虑 expert parallel，而不只是 TP/PP |

一句话：模型在改**单位 token 的成本形状**；系统要解决的是**如何在这个形状上把多请求跑满 GPU**。GQA 能让同一张卡塞下更大的 batch，但它替代不了调度器和分页。

## 朴素 `generate` 与四条约束

先写一个会正确出字、但上不了线的循环。这就是多数人第一次实现生成时的样子：

```python
def naive_generate(model, tokens, n):
    for _ in range(n):
        logits = model(tokens)       # 每步吃完整序列
        nxt = sample(logits[:, -1])
        tokens = torch.cat([tokens, nxt], dim=1)
    return tokens
```

第 `t` 步对前 `t-1` 个 token 的 K/V 是重复劳动。序列越长，浪费越明显：attention 对历史是平方级扫描。加上 KV cache 之后，循环长这样——GPT-2 源码里的 `past` 就是这个东西：

```python
def cached_generate(model, tokens, n):
    out = []
    logits, kv = model(tokens, kv=None)   # prefill：整段 prompt 一次算完
    nxt = sample(logits[:, -1])
    out.append(nxt)
    for _ in range(n - 1):
        logits, kv = model(nxt, kv=kv)    # decode：只进 1 个 token
        nxt = sample(logits[:, -1])
        out.append(nxt)
    return out
```

两边对比：

![朴素 generate 每步重算整段；Cached 路径 prefill 一次、decode 增量写 KV](/assets/llm-inference-01/fig3-naive-vs-cached.svg)

左边每步把已经看过的历史再算一遍；右边 prefill 一次写入 KV，之后每步只带上新 token，并追加新的 K/V。从这一对循环可以抽出四条约束。它们不是优化清单，是架构推出来的。

### 约束 1：自回归 = 多 iteration

一次用户请求 ≠ 一次模型调用。生成 100 个 token，至少要 1 次 prefill 加上约 99 次 decode。传统推理服务（图像分类、BERT 句向量）假设一次 forward 就结束，于是按「请求」组 batch：凑齐 N 条，一起跑完，一起返回。生成式 workload 会把这个假设撕开——短请求必须等最长的那个结束才能返回，新到达的请求也只能在门外等当前 batch 全部结束。

**系统萌芽：** 调度粒度必须到 iteration。Orca（OSDI 2022）把这一点写成 iteration-level scheduling：每个 step 只跑一批请求的**一轮**前向，完成的马上离开，新的马上进来。这也是后来常说的 continuous batching / in-flight batching。本篇只需要记住：没有 iteration 级调度，GPU 会把大量时间花在等待最长序列上。

举一个最小例子。三条请求同时开始，生成长度分别是 4、20、128 个 token。请求级 batch 必须等 128 那条结束，4 和 20 的调用方却已经可以返回。iteration 级调度下，第 4 步结束后腾出的槽位可以立刻塞进新请求。吞吐来自「槽位不被最长序列锁死」，不是来自「模型算得更快」。

### 约束 2：因果 ⇒ KV 可缓存，但显存线性涨

KV cache 把 decode 从「对整段历史做平方级重算」变成「对当前 token 做线性增量」。省的是**计算**，换来的是**显存**。每层、每个 KV head、每个 token，都要存一组 Key 和一组 Value：

```text
bytes / token = 2 × L × H_kv × d_h × elem_bytes
```

`2` 是 K 和 V；`L` 是层数；`H_kv` 是 KV head 数（GQA 之后通常小于 query head 数）；`d_h` 是 head 维度；`elem_bytes` 在 FP16/BF16 下是 2。手算两行：

| 模型 | 形状 | KV / token | 单条 4K |
| --- | --- | --- | --- |
| Llama-2-7B（无 GQA） | 32L × 32 KV × 128 | 512 KB | ~2.0 GB |
| Llama-3-8B（GQA 8 KV） | 32L × 8 KV × 128 | 128 KB | ~0.5 GB |

权重是固定成本：7B FP16 大约 14 GB，加载一次就在。KV 随请求数和长度涨，没有天花板。同一张 80 GB 卡，去掉权重和激活之后，剩下的显存决定了你能同时养活多少条 4K 上下文。这就是「为什么调大 batch 会 OOM，而看上去算力还没打满」的直接原因。

GQA 把 Llama-2-7B 那种 512 KB/token 打到 Llama-3-8B 的 128 KB/token，大约 4 倍。它是模型侧给系统减负，不是调度器的替代品：减完之后，KV 仍然随长度线性涨，仍然要分页、回收、共享前缀。

### 约束 3：Prefill 与 Decode 是两种作业

有了 cache 之后，同一次请求内部已经是两种 kernel 形态。若再把**新到达的长 prefill** 和**正在飞的 decode** 塞进同一次 forward，decode 会被拖死：本来 memory-bound、只要几十毫秒的一步，突然要陪着一段 compute-bound 的长 prompt 跑完。

Sarathi-Serve 在其论文设定下测到：朴素把长 prefill 和 decode 混跑，可使 token 间隔恶化一个数量级（文中约至 28×）。数字属于他们的 workload 和硬件，本文未复现；要记住的是方向——两相算力形态相反，不能当成同一次 `forward` 来调度。后面会看到三条对策：继续混跑但接受干扰、把 prefill 切成小块、或者把两相拆到不同的 GPU 上。

### 约束 4：请求异构

线上几乎不会出现「所有请求一样长、一起开始、一起结束」。有人 32 个 token 的短问，有人 8K 的文档问答；有人立刻结束，有人要生成两千词；很多请求还共享同一段系统提示。按最大长度做静态 padding，短序列的算力被浪费——batch 里 8 条请求、最长 4096、平均 512 时，大量 attention 算在 pad 上。按最大长度预分配一整块 KV，短请求和已结束请求占用的显存收不回来。系统必须在运行时分配、回收内存，并动态组 batch。能共享的前缀，不该每条请求各算一遍：同一条系统提示被 100 个用户复用，prefill 那一段 KV 只该存在一份。

## 模块地图：约束如何长成系统

把四条约束画在一张图上，就是本篇的交付物：

![从 Decoder-only 因果自回归到 Scheduler、KV Manager、两相策略、Executor/Serving](/assets/llm-inference-01/fig4-constraint-to-modules.svg)

每个模块只记职责和代表工作，不展开算法。读到具体引擎时，用这张图当索引，而不是当实现说明书。

### KV 管理：修约束 2 和约束 4 的显存部分

职责：按 token 追加 K/V，请求结束立刻回收，能共享的前缀尽量共享。朴素做法是给每条请求预留一块「最大长度 × 层数」的连续 buffer。输出比上限短时，尾部空着；请求结束到下一次分配之间，中间可能碎掉。vLLM 的 PagedAttention 把 KV 做成接近操作系统的分页：逻辑连续、物理按块；论文报告相对连续分配可把浪费压到接近零，吞吐在同延迟下相对当时的 FasterTransformer / Orca 有数倍提升。系统提示、RAG 的固定模板、多轮对话的历史，则不必重复 prefill——这是 prefix cache。SGLang 的 RadixAttention 把共享前缀做成树，是这一路的代表。

细节留给第 02 篇：页表、block size、radix tree、以及量化 KV 时的精度取舍。

### Scheduler：修约束 1 和约束 4 的时间部分

职责：每个 iteration 决定哪几个请求进入下一次 forward。Orca 的连续批处理解决「短请求被长请求拖死」；它还要在「先打完新来的 prefill」和「别饿死正在 decode 的人」之间做选择。FCFS 简单，但对延迟敏感的交互请求不友好；priority 队列要定义清楚「优先」指 TTFT 还是 TPOT。waiting / running / swapped 这些队列名字会在 vLLM 文档里出现，本篇只要求你知道：调度器不是可有可无的外围，它就是把多 iteration 变成可持续服务的那一层。

队列策略、抢占、token budget，是第 03 篇的事。

### 两相策略：修约束 3

三条路，只对比意图，不在本篇里判胜负：

| 策略 | 在修什么 | 代价 |
| --- | --- | --- |
| 同卡混合 + 连续批处理 | 提高利用率 | prefill 干扰 decode |
| Chunked prefill | 把长 prefill 切块，和 decode 拼成较稳定的 token budget | TTFT 与 TPOT 仍耦合 |
| P/D 分离 | 两相分卡，分别满足 TTFT 与 TPOT | KV 要跨阶段搬运 |

Chunked prefill（Sarathi-Serve、DeepSpeed-FastGen 的 splitfuse 同属这一族）利用 decode 算术强度低、GPU 还有空的事实，把一小块 prefill 捎上。它能避免「整段长 prompt 卡住所有 decode」，但两相仍在抢同一组 SM。DistServe 的问题意识比「吞吐更高」更准：应用往往**同时**有 TTFT 和 TPOT 的 SLO。吞吐高不等于 SLO 内的 goodput。把两相拆开后，并行策略也可以分别选：prefill 更吃算力，decode 更吃带宽和 KV。

集群怎么放置、KV 怎么搬、什么时候不该分离，留给第 04 篇。

### Executor 与 Serving：把一次 step 真正跑出去

Executor 真正跑 fused kernel、FlashAttention 类算子，以及张量并行 / 流水线并行 / 专家并行。MoE 出现后，专家并行变成一等公民：不再只是「把同一层切到多卡」，还要把不同 token 送到不同专家。Sampler 和 structured output 发生在 logits 之后——温度、top-p、停用词、JSON / 工具调用的约束解码——看起来像后处理，但仍在每步关键路径上，第 06 篇再写。

Serving 是离线 `LLM.generate` 与在线异步引擎的差别。vLLM 的解剖文章把层次写得很清楚：Engine Core 里是调度 + KV + 执行；外包一层才是 HTTP、多进程客户端、分布式路由。没有这一层，你仍然可以做离线吞吐测试，但不能接用户流量。

读任何引擎文档时，先填这四格，再看高级特性：

1. KV 怎么存？能否分页、共享前缀？
2. 每个 step 谁被调度进来？
3. Prefill / Decode 同卡、切块，还是分卡？
4. 执行器如何并行？请求从哪进、token 从哪出？

四格填得上，你就已经在读系统，而不是在背名词。

用 vLLM 的公开解剖文当一次练习：Engine Core 里能看到 Scheduler（waiting / running 队列）、KV cache manager（free block 池）、Model executor（paged attention kernel）。外层才是把离线 `LLM.generate` 变成在线服务的 client。chunked prefill、prefix cache、投机解码、P/D 分离被放在「高级特性」——它们都挂在这四件套上，而不是另起炉灶。SGLang 的差别主要会落在 KV 那一格（radix 前缀树）和调度对前缀命中的偏好；TensorRT-LLM 的差别更多落在 Executor 那一格（编译出来的 kernel 与 in-flight batching 的实现）。先填格，再比较，才比较得动。

还有三个常见误读，值得在进第 02 篇之前丢掉：

- **「有 KV cache 就不需要推理系统」。** cache 只解决单条请求内部的重算。多请求、变长、回收、共享前缀，仍然要管理器。
- **「batch 越大越好」。** decode 需要 batch 来摊销权重读取，但 batch 被 KV 显存卡住；超过这个点，再加大只会 OOM 或换入换出。
- **「P/D 分离一定更快」。** 它修的是两相干扰和 SLO 耦合。KV 搬运贵、流量偏短 decode、或者单卡本来就喂不饱时，分离可能得不偿失。这正是第 04 篇要量化的问题。

## 系列怎么往下读

| 篇 | 主题 | 本篇已点名 |
| --- | --- | --- |
| 01 | 架构 → 模块地图 | 本篇 |
| 02 | KV：分页与前缀复用 | PagedAttention, prefix cache |
| 03 | 调度：连续批处理与 chunked prefill | Orca, Sarathi-Serve |
| 04 | Prefill / Decode 分离与 goodput | DistServe |
| 05 | 并行：TP / PP / EP | MoE |
| 06 | 采样、投机解码、结构化输出 | Sampler |
| 07 | 量化与 kernel | FlashAttention |

本篇没有原机评测。公式可手算复现；Orca / vLLM / Sarathi / DistServe 的加速比都来自各自论文，硬件和 workload 不同，不能横比。若后续要「show, not tell」的数字，优先做三件事：玩具模型上无 cache 与有 cache 的墙钟对比；固定 `max_model_len` 下 KV 能撑的并发；同卡插入长 prefill 时在飞 decode 的 TPOT 变化。

推理系统不是一张优化清单，是生成式 Transformer 的运算后果。下一篇从 KV 显存公式往下挖，直到分页为什么像操作系统。

## 参考

1. Vaswani et al., *Attention Is All You Need*, 2017.
2. Radford et al., GPT-2；实现里的 `past` 即为 KV cache。
3. Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022.
4. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023.
5. Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024.
6. Zhong et al., *DistServe*, OSDI 2024.
7. Gordić, [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), 2025.

图用 [draw.io skill](https://github.com/bahayonghang/drawio-skills) 从 YAML 生成，源文件在仓库 `diagrams/llm-inference-01/`。本环境无 draw.io Desktop，导出为 SVG。
