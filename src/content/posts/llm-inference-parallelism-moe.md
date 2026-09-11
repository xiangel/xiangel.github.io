---
author: xiangel
pubDatetime: 2026-09-11T02:30:00Z
title: "大模型推理的并行策略与 MoE：从张量切分到专家并行"
slug: llm-inference-parallelism-moe
featured: true
draft: false
tags:
  - 大模型推理系统
  - 并行策略
  - MoE
  - LLM
description: 一张卡装不下，就得把模型切开。可切哪一维，决定了通信长什么样。用大厨团队的分工做类比，沿着 TP → PP → DP → EP 这条问题链，讲清 Megatron 的列/行切分、流水线气泡、DP Attention、专家并行，以及 DeepEP / EPLB / Dual-batch overlap 怎样把跨节点 all-to-all 藏进计算空隙。附一组无需 GPU、可复现的仿真。
---

如果你把一个 70B、乃至 DeepSeek-V3 那种 671B 的模型搬上线，多半会依次撞见四件"切不开就跑不动"的事：

1. **一张卡装不下权重**，得把同一层横着切开。
2. 切开之后 **节点内还够快，一出节点通信就把加速吃掉**。
3. 切完层还要吞吐，得 **再复制几份** 同时接单。
4. 模型变成 MoE 之后，用张量并行去切一个已经很小的 expert，**通信占比高到不划算**——得改切专家。

这四件事对应四种并行：**张量并行 TP（Tensor Parallel）**、**流水线并行 PP（Pipeline Parallel）**、**数据并行 DP（Data Parallel）**、**专家并行 EP（Expert Parallel）**。它们切的不是同一刀，通信形态也完全不同。

这是本系列第六篇。[第一篇](/posts/from-causal-lm-to-inference-system/)把"权重怎么切到多卡"列为后续模块；[第四篇](/posts/llm-inference-pd-disaggregation/)已经说了 prefill 想要**小 TP**、decode 想要**大 TP**，但还没解释 TP 本身是什么；[第五篇](/posts/llm-inference-scheduling-distributed/)把请求派到了集群上的某一台机器。这一篇钻进那台机器（以及它所属的并行组）里面：**Model Executor 到底按哪一维把模型切开，切开之后每一步要付多少通信账。**

> **说明**：本篇讲的是**一层 forward 内部**的切分，不是第五篇那种跨实例的请求路由。Context Parallel（按序列切注意力）放到后续 GPU / 长上下文篇；训练里的 DualPipe（前向/反向重叠）只在脚注里对照，正文只谈推理的 Dual-batch overlap。

全程我用一个贯穿的类比:**大厨团队的分工**。一张卡是一个灶台，一层计算是一道菜。TP 是**几个人围着同一口锅各炒一角**，每下一刀都要对齐；PP 是**备菜、热锅、装盘的流水线**；DP 是**开几间分店，各做各的单**；EP 是**每个师傅只做自己那几道拿手菜**，跑堂按菜单把半成品送到对应的灶。沿着系列一贯的"**每暴露一个问题，就引入一种优化，又带出新问题**"的主线往下讲。

## Table of contents

## 一、并行在切哪一维

先把四刀的"形状"摆在一张图上。同一份 Transformer，可以沿四个轴切开：

![同一份模型的四种切开方式：TP 切 hidden，PP 切 layers，DP 切 batch，EP 切 experts](/assets/posts/llm-inference-parallelism-moe/diagram-four-axes.png)

| 并行   | 切开的维                   | 典型通信                          | 推理里最怕什么                 |
| ------ | -------------------------- | --------------------------------- | ------------------------------ |
| **TP** | 一层里的 hidden / 权重矩阵 | 层内 **all-reduce**               | 跨节点；小消息的延迟           |
| **PP** | 层与层之间                 | 阶段边界 **点对点**               | 气泡；batch=1 灌不满           |
| **DP** | 请求 / batch               | 推理里通常**不**梯度同步          | 副本间的 KV 与前缀缓存被摊薄   |
| **EP** | MoE 的 experts             | **dispatch / combine all-to-all** | 热点专家；两种通信内核不能共存 |

四刀可以叠。DeepSeek-V3 线上就是：**attention 走 TP + SP + DP，MoE 走 EP**。不是选一种并行，而是给每一段计算挑最合适的那一刀。

> **类比**：这就是后厨的工位图。有人围着同一口锅分工（TP），有人按工序排成流水（PP），有人再开一间分店接溢出的单（DP），还有人按菜品专精（EP）。工位切错了，不是人手不够，是**传菜的走位**把时间吃掉了。

## 二、第一刀：一张卡装不下 —— 张量并行 TP

最朴素的需求：权重比一张卡的显存大。Megatron-LM 的做法是把每一层的线性层沿 **hidden** 切开，让几张卡**同时算同一层的不同切片**。

标准拆法是 **Column Parallel → Row Parallel** 成对出现：

![列并行按输出维切开、无需通信；行并行按输入维切开，之后一次 all-reduce，体积正比于 (t−1)/t。节点内走 NVLink，跨节点走 IB 会把加速吃掉](/assets/posts/llm-inference-parallelism-moe/diagram-tp-allreduce.png)

1. **列并行（Column Parallel）**：`Y = X W`，按**列**切 `W`。每张卡拿完整的 `X`，算出 `Y` 的一块。逻辑上拼接即完整输出，**这一步没有通信**。Attention 的 QKV、MLP 的第一段（升维）走这条。
2. **行并行（Row Parallel）**：按**行**切 `W`，输入 `X` 也跟着切。每张卡算出一份**部分和**，再 **all-reduce** 得到完整的 `Y`。Attention 的 output projection、MLP 的第二段（降维）走这条。

一层 decoder 通常两次 all-reduce。Ring all-reduce 的数据量正比于 **(t−1)/t**：TP 从 1 到 2，通信从 0 跳到一半；再往上增量变缓，但每次都要同步。**墙钟跟最慢的那张卡、以及这次同步走的那条线走。**

所以 TP 有一条硬约束：**尽量停在 NVLink 域里**（单机 8 卡、或 NVLink 连接的超节点）。H100 节点内 NVLink 是数百 GB/s；一出节点，InfiniBand 掉到数十 GB/s，延迟也高一个数量级。实验 A 会把这条线画出来：同样的一层，NVLink 上 TP 还在加速，IB 上通信很快反超。这也解释了 DeepSeek-V3 为什么把 attention 的 TP **钉死在 4**——"小 TP 限制通信开销"。

> **类比**：几个人围着同一口锅炒同一道菜，每下一刀都要互相报一声"我这边好了"（all-reduce）。灶台挨着（NVLink），报一声很快；灶台隔了两条街（IB），报一声的时间比炒菜还长。

**序列并行 SP（Sequence Parallel）** 是 TP 的伴生，不是另一种切模型的方式。LayerNorm、Dropout 这些不吃权重、却吃激活显存的算子，按**序列维**切开，用 all-gather / reduce-scatter 替换部分 all-reduce，把激活显存再摊薄一档。DeepSeek-V3 的 attention 写的是 **TP4 + SP**，就是这套组合。按序列切**注意力计算本身**（Context Parallel）是另一件事，留给硬件与长上下文那一篇。

## 三、第二刀：出节点太贵 —— 流水线并行 PP

TP 出了节点就不划算。可模型还是太大，8 张卡的 NVLink 域装不下。下一刀改切 **layers**：把连续若干层交给一个**阶段（stage）**，激活沿阶段往下传。这就是 **流水线并行 PP**。

![p=4 个阶段、m=8 个微批次的流水线；斜线格是气泡。气泡比例 (p−1)/(m+p−1)；batch=1 时利用率只剩 1/p](/assets/posts/llm-inference-parallelism-moe/diagram-pp-bubble.png)

通信形态变了：不再是每层一次全集同步，而是阶段边界一次**点对点**传递激活。看起来更便宜。新问题是 **气泡（bubble）**——流水线灌不满时，前面的阶段在算、后面的阶段空手等，收尾时反过来。

Narayanan 等人给出的理想利用率是 `m / (m + p − 1)`：`p` 是阶段数，`m` 是微批次（micro-batch）数。`p=8`、`m=1` 时利用率只剩 **12.5%**——几乎退化成串行。训练可以堆大 global batch 把 `m` 灌满；**在线 decode 常常一步只推一个 token**，`m` 灌不满，PP 的气泡就会把卡空出来。

> **类比**：备菜、热锅、装盘排成四个工位。一桌十道菜（m 大），流水线转起来每个工位都忙；只来一份例汤（batch=1），后面三个工位只能干等。训练是婚宴，推理经常是单人套餐。

所以生产推理里，PP 更多是"实在跨不出去 NVLink 时的备选"，而不是 decode 的主方案。DeepSeek-V3 的推理单元**不用 PP**，把跨节点的预算留给了 EP 的 all-to-all。

## 四、还要吞吐：数据并行 DP，以及 MoE 入口的 DP Attention

TP 和 PP 解决的是"**一份模型怎么切开**"。流量再大，一份切开的模型一次还是只能吞一个（或一小批）请求。下一刀是 **复制**：多放几份，各接各的单。这就是 **数据并行 DP**。

训练里的 DP 每步要 **all-reduce 梯度**；**推理里没有梯度**，副本之间默认不说话。它看起来最便宜，却会把[第二篇](/posts/kv-cache-paged-attention-and-prefix-caching/)的前缀缓存和[第五篇](/posts/llm-inference-scheduling-distributed/)的缓存感知路由摊薄——每份副本各缓存各的，命中率随副本数下降。DP 在 serving 里的正确用法，常常不是"无脑复制一份完整 70B"，而是和别的并行叠在一起。

MoE 把这件事逼出了一个专门变体：**DP Attention**。

MoE 的 FFN 是稀疏的，专家可以按 EP 切开；**attention 不是**。MLA / GQA 的 KV 是按序列、按请求长出来的，每个 DP rank 要看到**自己那批请求的完整 KV**。于是同一层里出现了两种并行：

- **Attention**：按 batch 切开（DP），每张卡算自己那几条请求的注意力，KV 不跨卡。
- **MoE**：按 expert 切开（EP），token 按路由飞到专家所在的卡。

DeepSeek-V3 prefill 写的 **TP4 + SP + DP8**，decode 写的 **TP4 + SP + DP80**，指的就是 attention 这一侧。EP 变大时，DP 也跟着变大——因为 `EP ≈ TP × DP` 这一侧的 MoE 世界里，多出来的卡主要用来**多住专家**，attention 就按 DP 复制。

> **类比**：分店可以各做各的家常菜（DP）；但如果后厨改成"凉菜归凉菜组、热菜归热菜组"（EP），前厅点单还是按桌走（attention 仍按请求切）。两套工位叠在同一班次里。

## 五、稀疏起来：专家并行 EP

Dense 模型的 FFN 是一整块大矩阵，用 TP 切是划算的。MoE 把这块换成**很多个小专家**，每个 token 只激活其中 top-k 个。DeepSeekMoE 的账是：

- **671B** 总参数，一次前向大约只激活 **37B**；
- 每层 **1 个共享专家 + 256 个路由专家**，每个 token **top-8**。

单个 expert 的 GEMM 已经很小。再用 TP 把它切开，通信占比会高到不划算。专家是天然的并行粒度——**一张卡住若干个完整的 expert**，token 按路由飞过去。这就是 **专家并行 EP**。

![token 经 Gate 选 top-k 专家，dispatch all-to-all 送到专家所在 GPU，grouped GEMM 之后再 combine all-to-all 写回](/assets/posts/llm-inference-parallelism-moe/diagram-moe-dispatch.png)

一步 MoE 层的关键路径是三段：

1. **Gate / Router**：每个 token 打分，选出 top-k 个专家。
2. **Dispatch**：按专家把 token **打包**，一次 **all-to-all** 送到专家所在的 GPU。
3. **Grouped GEMM + Combine**：各卡对自己收到的 token 做专家计算，再一次 all-to-all **加权写回**原位。

和 TP 的 all-reduce 不同：all-reduce 是"每人一份相同大小的求和"；all-to-all 是"每人发给每人的量取决于路由"。路由一偏，有的卡 inbox 爆满，有的卡空手——**墙钟跟最忙的那张卡走**。这就是下一节的两个问题。

> **类比**：不再围着一口锅炒，改成每个师傅只做自己那几道拿手菜。跑堂（dispatch）按菜单把半成品送到对应的灶，做好再收齐（combine）。菜单若总点同一道爆款，那位师傅会被点爆，别的灶却在空转。

## 六、all-to-all 贵 + 热点专家 —— DeepEP、EPLB 与 Dual-batch overlap

EP 把两个新问题同时放到台面上。

**问题一：热点专家。** 真实流量的路由几乎从来不是均匀的，更接近 zipf：少数专家吃掉大部分 token。DeepSeek-V3 的对策是 **冗余专家（redundant experts）+ EPLB（Expert Parallelism Load Balancer）**：统计一段时间里谁最烫，把这些专家**再复制一份**放到相对空闲的卡上，新来的 token 发给当前更轻的那份副本；并且在节点内重排专家，尽量别增加跨节点 all-to-all。Prefill 部署了 **32 个冗余专家**，EP32 上每卡从 8 个 routed 变成 **8+1**。Decode 把共享专家也当成一个"永远被选中的热专家"，64 张卡专门托管冗余和共享。LMSYS 在 96×H100 上测到 EPLB 带来约 **1.49× prefill / 2.54× decode** 的吞吐。

![zipf 热点让单卡过载；EPLB 把最烫的专家复制到闲卡，墙钟跟最忙的卡走](/assets/posts/llm-inference-parallelism-moe/diagram-eplb.png)

**问题二：通信形态随阶段而变。** Prefill 的 dispatch 消息大，要的是**带宽**；decode 每步只有一个新 token，消息小，要的是**延迟**。DeepEP 为此做了两种内核：

- **高吞吐（normal）**：打满管道，适合 prefill；输出形状是 symbolic 的，和 CUDA Graph 不对付。
- **低延迟（low-latency）**：走 IB P2P / IBGDA（NVSHMEM），适合 decode，能进 CUDA Graph。

关键限制：**同一通信组不能同时跑两种内核。** 这就是[第四篇](/posts/llm-inference-pd-disaggregation/)的 PD 分离在 MoE 上从"可选优化"变成**刚需**的原因——DeepEP 的 auto 模式在 colocate 引擎里选不了"prefill 走 normal、decode 走 low-latency"。拆开之后，两个池各绑一种内核。

通信再优化，all-to-all 还是藏不住。下一步是 **Dual-batch overlap / TBO（Two-Batch Overlap）**：把一个 batch 切成两个微批次，**一批在做 grouped GEMM 时，另一批在做 dispatch/combine**。墙钟从 `计算 + 通信` 变成 `max(计算, 通信)`。LMSYS 测到 prefill 上 TBO 在同等 token 数下 **+27–35%** 吞吐，顺带把单卡能吞的 token 上限从 8k 抬到 16k。

![DeepEP 两种内核不能共存，必须 PD 分离；两个微批次交错后墙钟取 max(计算, 通信)](/assets/posts/llm-inference-parallelism-moe/diagram-deepep-overlap.png)

> **脚注**：DeepSeek 训练里的 **DualPipe** 是把前向和反向叠在同一条流水线上，用来藏 PP 的气泡。名字像，问题不是同一个。推理这边重叠的是**两个微批次的计算与 all-to-all**，不要和 DualPipe 混为一谈。

> **类比**：爆款菜复制一份到空闲灶（EPLB）；午餐和夜宵用两套完全不同的传菜节奏，不能共用一条对讲频道（DeepEP 两种内核 → PD 分离）；两个跑堂交错着送——一个在传、一个在炒（TBO）。

## 七、生产里怎么配：DeepSeek-V3 与开源框架

把前面的刀叠回去，就是 DeepSeek-V3 论文里的推理单元。注意：**prefill 和 decode 的并行度故意不一样**——这正是第四篇"阶段专属优化"在 MoE 上的落点。

![Prefill：4 节点 / 32 GPU，attention TP4+SP+DP8，MoE EP32；Decode：40 节点 / 320 GPU，attention TP4+SP+DP80，MoE EP320](/assets/posts/llm-inference-parallelism-moe/diagram-pd-ep-scale.png)

|           | Prefill                            | Decode                        |
| --------- | ---------------------------------- | ----------------------------- |
| 最小单元  | 4 节点 / **32 GPU**                | 40 节点 / **320 GPU**         |
| Attention | **TP4 + SP + DP8**                 | **TP4 + SP + DP80**           |
| MoE       | **EP32**（每卡 8 routed + 1 冗余） | **EP320**（每卡 1 个 routed） |
| 冗余      | 32 个冗余专家                      | 64 张卡托管冗余 + 共享        |
| 通信      | 高吞吐内核 + 双微批次              | IB P2P + IBGDA                |

Prefill 算力密集，EP 不必极大，小 TP 喂饱计算即可；decode 访存密集，把专家摊到更多卡上，每卡权重更少、能进更大的 batch，聚合带宽才够。后来 DeepSeek 开源周的系统概述里，decode 单元收成过 **EP144 / 18 节点** 的写法，和论文的 EP320 不是同一版部署——数字会随硬件和流量改，**"两阶段用不同 EP"** 这条原则不变。

开源侧的缩小版是 LMSYS / SGLang 在 **96×H100**（12 节点）上的工作：prefill **EP32**、decode **EP72**（大约是论文 decode 规模的一半），相对 vanilla TP16 输出吞吐最高约 **5×**，单节点约 52.3k input / 22.3k output tok/s（2k 输入）。框架开关大致是：

- **SGLang**：`--enable-deepep-moe`、`--deepep-mode {normal,low_latency}`、`--enable-eplb`、`--enable-two-batch-overlap`、`--enable-dp-attention`；PD 分离用 `--disaggregation-mode {prefill,decode}`。
- **vLLM**：`--enable-expert-parallel`（EP 规模 = `TP × DP`，单独开 PP 不会激活 EP）、`--all2all-backend`、`--enable-eplb`、`--enable-dbo`。

没 PD 分离就强开两种 DeepEP 内核，通信组会卡住——这不是开关没打开，是架构不允许。

## 八、三组无需 GPU 的实验

下面三组实验都在 CPU 上跑，`SEED=42` 可复现。它们**不是**端到端 GPU benchmark：实验 A 是一层 decoder 的解析墙钟（计算按 1/TP 缩放 + ring all-reduce），实验 B 是流水线气泡公式，实验 C 是 zipf 路由下的 GPU 负载。用来把正文里的数量级钉死，不代替真机上的 kernel 剖面。脚本在 `diagrams/parallelism-moe/sim_experiments.py`。

### 实验 A：TP 计算 vs all-reduce —— NVLink 还在加速，IB 上通信反超

一层两次 all-reduce（attention out + MLP out）。激活大小 = `hidden × tokens × 2`（bf16，hidden=8192）。Ring all-reduce：`2(t−1)` 跳，每跳发送 `nbytes/t`。NVLink 按 400 GB/s、2 µs/hop；IB 按 50 GB/s、10 µs/hop。Prefill 取 4096 token、TP=1 时计算 6.0 ms；decode 取 64 并发、TP=1 时计算 0.55 ms（访存主导的粗口径）。

```python
def ring_allreduce_ms(nbytes, tp, bw_GBps, hop_lat_us):
    if tp <= 1:
        return 0.0
    hops = 2 * (tp - 1)
    chunk = nbytes / tp
    return hops * hop_lat_us / 1000.0 + hops * chunk / (bw_GBps * 1e6)

def layer_ms(tp, tokens, compute_tp1, bw, lat_us, hidden=8192):
    compute = compute_tp1 / tp
    nbytes = hidden * tokens * 2          # bf16 激活
    comm = 2 * ring_allreduce_ms(nbytes, tp, bw, lat_us)  # 一层两次
    return compute + comm
```

![Prefill 在 NVLink 上随 TP 下降、在 IB 上几乎走平；Decode 在 NVLink 上 TP8 最好，在 IB 上 TP16 已经慢于 TP1](/assets/posts/llm-inference-parallelism-moe/sim-tp-compute-vs-comm.png)

一手结果（一层 decoder 墙钟）：

| TP  | Prefill · NVLink | Prefill · IB | Decode · NVLink | Decode · IB |
| --- | ---------------- | ------------ | --------------- | ----------- |
| 1   | 6.000 ms         | 6.000 ms     | 0.550 ms        | 0.550 ms    |
| 2   | 3.344            | 5.724        | 0.288           | 0.357       |
| 4   | 2.027            | **5.647**    | 0.169           | **0.320**   |
| 8   | 1.393            | 5.728        | **0.134**       | 0.422       |
| 16  | **1.124**        | 6.008        | 0.164           | 0.713       |

Prefill 在 NVLink 上从 TP1 到 TP16 仍有 **5.3×**；同样一层放到 IB 上，最快的 TP4 也只比 TP1 快 6%，TP16 已经略慢于单卡。Decode 更刺：NVLink 的甜点在 **TP8**（再往上延迟项抬头），IB 上 TP16 是 TP1 的 **1.3× 慢**。这就是"TP 停在节点内、attention 钉在 TP4"的数量级来源。

### 实验 B：PP 气泡 —— batch=1 时利用率按 1/p 塌掉

```python
def utilization(p, m):
    return m / (m + p - 1)          # 理想利用率；气泡 = (p-1)/(m+p-1)
```

![PP 阶段越多、微批次越少，理想利用率越低；m=1 时 PP=8 只剩 12.5%](/assets/posts/llm-inference-parallelism-moe/sim-pp-bubble.png)

一手结果：

|       | m=1       | m=4   | m=8   | m=32  |
| ----- | --------- | ----- | ----- | ----- |
| PP=2  | 50.0%     | 80.0% | 88.9% | 97.0% |
| PP=4  | 25.0%     | 57.1% | 72.7% | 91.4% |
| PP=8  | **12.5%** | 36.4% | 53.3% | 82.1% |
| PP=16 | **6.2%**  | 21.1% | 34.8% | 68.1% |

训练可以把 `m` 堆到 32 以上，PP=8 还能到 82%；**在线 decode 的 m≈1，PP=8 只剩八分之一的卡在干活**。气泡公式不涉及 GPU，却足够解释为什么推理单元更愿意把跨节点预算花在 EP 而不是 PP 上。

### 实验 C：zipf 路由 vs EPLB —— 冗余副本把 straggler 压下来

256 个专家、32 张 GPU（EP32，每卡 8 个专家）、top-8、8 万 token，路由按 zipf（`a=1.15`）。对比三种放置：专家 0–7 连续放在 GPU 0（最热的都挤在一起）、按热度贪心装箱（每专家仍一份）、装箱后再给最热的 32 个专家加一份冗余。每个 token 发给该专家当前最轻的那份副本。墙钟跟 `max/mean` 走。

```python
# routes[i] = 第 i 个 token 的 top-k 专家；owner_lists[e] = 持有专家 e 的 GPU 列表
load = np.zeros(n_gpus)
for e in routes.ravel():
    owners = owner_lists[e]
    g = owners[int(np.argmin(load[owners]))]   # 发给当前最轻的副本
    load[g] += 1
imbalance = load.max() / load.mean()
```

![左：专家热度呈 zipf；右：连续放置 13.96× 不均，装箱降到 3.73×，再加 32 个冗余降到 2.11×](/assets/posts/llm-inference-parallelism-moe/sim-eplb-imbalance.png)

一手结果：

| 放置                         | max / mean |
| ---------------------------- | ---------- |
| 连续放置（热专家挤在 GPU 0） | **13.96×** |
| 按热度装箱                   | 3.73×      |
| 装箱 + 32 冗余               | **2.11×**  |

只改放置、不加副本，不均已经从 14 倍掉到 3.7 倍；再复制 32 个最烫的专家，straggler 降到 2.1 倍。这和 LMSYS 观察到的"EPLB 在大规模下拉吞吐"是同一件事的负载侧：墙钟不看平均值，看最忙的那张卡。仿真没有建模 all-to-all 的字节数，所以**不能**直接读成 1.49× / 2.54× 那些端到端数字——那些来自真机，见第七节。

## 九、总结与延伸

并行策略的主线，还是那条"发现问题 → 引入优化 → 带出新问题"的链子：

- **一张卡装不下** → **TP（Megatron 列/行切分）**：层内 all-reduce，体积 `(t−1)/t`，停在 NVLink 域。
- **出节点太贵** → **PP**：切 layers，气泡 `(p−1)/(m+p−1)`；batch=1 时几乎串行。
- **还要吞吐** → **DP / DP Attention**：推理不梯度同步；MoE 下 attention 按 batch 复制、专家按 EP 切。
- **稀疏 FFN 不值得用 TP 切** → **EP**：dispatch / combine all-to-all。
- **all-to-all 贵 + 热点专家** → **DeepEP 两种内核 + EPLB 冗余副本**。
- **通信还是藏不住** → **Dual-batch overlap**；两种内核不能共存 → **咬合第四篇的 PD 分离**。

一句话带走:**并行策略就是给每一段计算选一刀——TP 切 hidden、PP 切层、DP 切请求、EP 切专家；MoE 把最后一刀推到跨节点之后，真正决定墙钟的是 all-to-all 和最烫的那个专家，而不是平均 FLOPs。**

延伸阅读:下一篇会把负载换成 **long-CoT / 推理模型**：思维链把 decode 拉得很长，KV 生命周期、straggler 和调度都会变味。再往后是 GPU 架构与 attention kernel（含本篇刻意留给那里的 Context Parallel）。若对"把 attention 和 FFN 拆到两类机器"感兴趣，见 MegaScale-Infer（ByteDance，把 MoE 的 attention / expert 再做一次分离，是 EP 之后的另一刀，本篇不展开）。

## 参考

1. Shoeybi et al., [_Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism_](https://arxiv.org/abs/1909.08053), 2019.
2. Narayanan et al., [_Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM_](https://arxiv.org/abs/2104.04473), SC 2021.
3. Liu et al., [_DeepSeek-V3 Technical Report_](https://arxiv.org/abs/2412.19437), 2024.（§3.4 推理部署：prefill EP32 / decode EP320）
4. Zhao et al., [_Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures_](https://arxiv.org/abs/2505.09343), 2025.
5. DeepSeek, [_DeepEP: an efficient expert-parallel communication library_](https://github.com/deepseek-ai/DeepEP), 2025.
6. LMSYS, [_Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs_](https://lmsys.org/blog/2025-05-05-large-scale-ep/), 2025.
7. vLLM, [_Expert Parallel Deployment_](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)（`--enable-expert-parallel`，EP = TP×DP）.
8. SGLang, [_Expert Parallelism_](https://docs.sglang.io/docs/advanced_features/expert_parallelism.html)（DeepEP / EPLB / TBO）.
9. Chen et al., [_MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism_](https://arxiv.org/abs/2504.02263), 2025.
10. DeepSeek, [_One More Thing: DeepSeek-V3/R1 Inference System Overview_](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)（开源周：decode 单元的 EP144 写法与论文 EP320 不同）.
