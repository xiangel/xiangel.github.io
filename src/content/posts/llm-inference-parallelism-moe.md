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

如果你把一个 70B、乃至 DeepSeek-V3 那种 671B 的模型搬上线，很少能靠"再买一张更大的卡"收场。更常见的是，问题会按这个顺序一层层冒出来：

1. **一张卡装不下权重。** 同一层的矩阵太大，只能横着切开，几张卡同时算这一层的不同切片。
2. **切开之后，节点里面还快，一出节点就不划算。** 层内同步走 NVLink 还能被计算盖住；换成 InfiniBand，通信自己就变成瓶颈。
3. **一份切开的模型，一次还是只能吞一小批请求。** 要吞吐，就得再复制几份，各接各的单。
4. **模型变成 MoE 之后，张量并行会切错地方。** 单个 expert 已经很小，再用 TP 切开，通信占比高到不划算——得改切专家。

这四件事对应四种并行：**张量并行 TP（Tensor Parallel）**、**流水线并行 PP（Pipeline Parallel）**、**数据并行 DP（Data Parallel）**、**专家并行 EP（Expert Parallel）**。名字都带"并行"，切的却不是同一刀，通信形态也完全不同。选错一刀，不是人手不够，是传菜把时间吃掉了。

这是本系列第六篇。[第一篇](/posts/from-causal-lm-to-inference-system/)把"权重怎么切到多卡"列为后续模块，当时只点了题。[第四篇](/posts/llm-inference-pd-disaggregation/)已经说了 prefill 想要**小 TP**、decode 想要**大 TP**，但还没解释 TP 本身在切什么。[第五篇](/posts/llm-inference-scheduling-distributed/)把请求派到了集群上的某一台机器。这一篇钻进那台机器、以及它所属的并行组里面，问的是：**Model Executor 到底按哪一维把模型切开？切开之后，每一步要付多少通信账？**

> **说明**：本篇讲的是**一层 forward 内部**怎么切分，不是第五篇那种跨实例的请求路由。按序列切注意力计算本身（Context Parallel）放到后续 GPU / 长上下文篇。训练里的 DualPipe（把前向和反向叠在一起）只在后面脚注里对照一下，正文只谈推理的 Dual-batch overlap。

全程用一个类比贯穿：**大厨团队的分工**。一张卡是一个灶台，一层计算是一道菜。TP 是几个人围着同一口锅各炒一角，每下一刀都要对齐；PP 是备菜、热锅、装盘排成流水线；DP 是再开几间分店，各做各的单；EP 是每个师傅只做自己那几道拿手菜，跑堂按菜单把半成品送到对应的灶。还是系列那条主线：**每暴露一个问题，就引入一种优化，又带出新问题。**

## Table of contents

## 一、并行在切哪一维

先把四刀的"形状"摆清楚。同一份 Transformer，可以沿四个轴切开——切 hidden、切层、切请求、切专家。它们不是互斥选项，后面会看到，生产系统几乎总是叠着用。

![同一份模型的四种切开方式：TP 切 hidden，PP 切 layers，DP 切 batch，EP 切 experts](/assets/posts/llm-inference-parallelism-moe/diagram-four-axes.png)

| 并行   | 切开的维                   | 典型通信                          | 推理里最怕什么                 |
| ------ | -------------------------- | --------------------------------- | ------------------------------ |
| **TP** | 一层里的 hidden / 权重矩阵 | 层内 **all-reduce**               | 跨节点；小消息的延迟           |
| **PP** | 层与层之间                 | 阶段边界 **点对点**               | 气泡；batch=1 灌不满           |
| **DP** | 请求 / batch               | 推理里通常**不**梯度同步          | 副本间的 KV 与前缀缓存被摊薄   |
| **EP** | MoE 的 experts             | **dispatch / combine all-to-all** | 热点专家；两种通信内核不能共存 |

读这张表时，不妨盯着"典型通信"那一列。TP 的 all-reduce 是"每个人手里都有一份同样大小的数据，加总之后每人再拿回完整结果"；PP 的点对点只在相邻阶段之间递一次激活；DP 在推理里常常根本不通信；EP 的 all-to-all 则是"每个人都可能给其他所有人发一份，份量还取决于路由"。**通信的形状，比"用了几张卡"更能决定这刀能不能赚钱。**

四刀也可以叠在同一层里。DeepSeek-V3 线上就是这样：attention 走 TP + SP + DP，MoE 走 EP。不是在四种并行里单选一个，而是给每一段计算挑最合适的那一刀——稠密、要同步的部分用 TP，稀疏、按专家走的部分用 EP，请求维度再用 DP 摊开。

> **类比**：这就是后厨的工位图。有人围着同一口锅分工（TP），有人按工序排成流水（PP），有人再开一间分店接溢出的单（DP），还有人按菜品专精（EP）。工位切错了，不是人手不够，是**传菜的走位**把时间吃掉了。

## 二、第一刀：一张卡装不下 —— 张量并行 TP

最朴素的需求：权重比一张卡的显存大。一张 H100 是 80 GB，70B 的 bf16 权重就要 140 GB，还没算 KV Cache。所以 Megatron-LM 的做法是：不要把整层交给一张卡，把每一层的线性层沿 **hidden** 切开，让几张卡**同时算同一层的不同切片**。

一个线性层写成 `Y = X W`。切开的方式有两种，而且在 Transformer 里几乎总是**成对出现**：先 Column Parallel，再 Row Parallel。

![列并行按输出维切开、无需通信；行并行按输入维切开，之后一次 all-reduce，体积正比于 (t−1)/t。节点内走 NVLink，跨节点走 IB 会把加速吃掉](/assets/posts/llm-inference-parallelism-moe/diagram-tp-allreduce.png)

1. **列并行（Column Parallel）**：按**列**切 `W`，也就是按输出维切开。每张卡都拿到完整的 `X`，各自算出 `Y` 的一块。这几块在逻辑上拼起来就是完整输出，**这一步没有通信**。Attention 的 QKV、MLP 的第一段（升维）走这条。
2. **行并行（Row Parallel）**：按**行**切 `W`，也就是按输入维切开。上一层刚切出来的那一块 `X`，正好对得上这一层的一块 `W`。每张卡算出一份**部分和**，再做一次 **all-reduce**，才能得到完整的 `Y`。Attention 的 output projection、MLP 的第二段（降维）走这条。

之所以要成对，是为了少通信。升维用列并行，中间那截激活可以就地留给下一层；降维用行并行，只在层的出口同步一次。一层 decoder 通常就是这样两次 all-reduce：一次在 attention 出口，一次在 MLP 出口。

Ring all-reduce 的数据量正比于 **(t−1)/t**。这个式子有个不太直观的后果：TP 从 1 到 2，通信从 0 一下子跳到"全量的一半"；再往上，增量变缓，但每一次都要等所有卡对齐。墙钟既取决于最慢的那张卡，也取决于这次同步走的是哪条线——同一机箱里的 NVLink，还是机箱外面的 InfiniBand。

所以 TP 有一条硬约束：**尽量停在 NVLink 域里**。常见的是单机 8 卡，或者 NVLink 连起来的超节点。H100 节点内 NVLink 是数百 GB/s 这个量级；一出节点，InfiniBand 掉到数十 GB/s，延迟也高一个数量级。实验 A 会把这条线画出来：同样算一层，NVLink 上加大 TP 还在加速，IB 上通信很快反超计算。这也解释了 DeepSeek-V3 为什么把 attention 的 TP **钉死在 4**——论文原话就是用小 TP 限制通信开销。再大，切出来的计算更碎，同步却更密。

> **类比**：几个人围着同一口锅炒同一道菜，每下一刀都要互相报一声"我这边好了"（all-reduce）。灶台挨着（NVLink），报一声很快；灶台隔了两条街（IB），报一声的时间比炒菜还长。

**序列并行 SP（Sequence Parallel）** 是 TP 的伴生，不是另一种切模型的方式。LayerNorm、Dropout 这些算子几乎不吃权重，却把激活张量整份摊在显存里。TP 只切了线性层的权重，激活还是整段序列都在。SP 的做法是：把这些算子按**序列维**切开，每张卡只留自己那一段 token 的激活，用 all-gather / reduce-scatter 替换掉一部分 all-reduce。换来的是激活显存再薄一档。DeepSeek-V3 的 attention 写成 **TP4 + SP**，指的就是这套组合。

按序列去切**注意力计算本身**（Context Parallel）是另一件事：那是在超长上下文里，把 QK 的计算摊到多卡上。本篇不展开，留给硬件与长上下文那一篇。

## 三、第二刀：出节点太贵 —— 流水线并行 PP

TP 出了节点就不划算。可模型还是太大：8 张卡的 NVLink 域依然装不下整份权重。下一刀改切 **layers**——把连续若干层交给一个**阶段（stage）**，激活算完这一段，再递给下一个阶段。这就是 **流水线并行 PP**。

![p=4 个阶段、m=8 个微批次的流水线；斜线格是气泡。气泡比例 (p−1)/(m+p−1)；batch=1 时利用率只剩 1/p](/assets/posts/llm-inference-parallelism-moe/diagram-pp-bubble.png)

通信形态变便宜了：不再是每层做一次全集同步，只在阶段边界**点对点**传一次激活。新问题换成了 **气泡（bubble）**。流水线要先灌满才会转起来：第 0 个阶段开始算第 1 个微批次时，后面的阶段只能空手等；最后一个微批次离开第 0 个阶段之后，前面的阶段又要空转到收尾。图里的斜线格，就是这两头空出来的时间。

Narayanan 等人给出的理想利用率是 `m / (m + p − 1)`。`p` 是阶段数，`m` 是微批次（micro-batch）数。气泡占比则是 `(p − 1) / (m + p − 1)`。`p=8`、`m=1` 时，利用率只剩 **12.5%**——八张卡里几乎只有一张在干活，其余在等。训练可以堆很大的 global batch，再切成许多微批次，把 `m` 灌满；**在线 decode 常常一步只推一个 token**，你没有那么多微批次可切。就算硬切，也会把单条请求的延迟拉长：用户要等整条流水线灌完，才看到这个 token。

> **类比**：备菜、热锅、装盘排成四个工位。一桌十道菜（`m` 大），流水线转起来每个工位都忙；只来一份例汤（batch=1），后面三个工位只能干等。训练更像婚宴，可以提前备一大桌；推理经常是单人套餐，客人还站在窗口等这一份。

所以生产推理里，PP 更多是"实在跨不出 NVLink 时的备选"，而不是 decode 的主方案。DeepSeek-V3 的推理单元**不用 PP**，把跨节点的预算留给了后面要讲的 EP all-to-all——那种通信虽然更散，但至少每张卡都有专家在算，不会整段空转。

## 四、还要吞吐：数据并行 DP，以及 MoE 入口的 DP Attention

TP 和 PP 解决的是"**一份模型怎么切开**"。切完之后，这一份模型一次仍然只能吞一个、或一小批请求。流量再大，单份并行组也会先被请求队列堵住。下一刀是 **复制**：同样的（或已经按 TP 切开的）模型多放几份，各接各的单。这就是 **数据并行 DP**。

训练里的 DP 每一步都要 **all-reduce 梯度**，副本之间同步很重。**推理里没有梯度**，副本之间默认不说话，所以它看起来最便宜。便宜的代价在别处：每份副本各自缓存各自的 KV。你加的副本越多，同一个前缀就越容易被摊到不同机器上，[第二篇](/posts/kv-cache-paged-attention-and-prefix-caching/)的前缀缓存、[第五篇](/posts/llm-inference-scheduling-distributed/)的缓存感知路由都会被稀释。DP 在 serving 里的正确用法，常常不是再复制一份完整的 70B，而是和别的并行叠在一起，只复制那些**必须按请求切开**的部分。

MoE 把这件事逼出了一个专门变体：**DP Attention**。

原因在于同一层里其实有两种计算。MoE 的 FFN 是稀疏的，256 个专家里每个 token 只走 top-k 个，专家可以按 EP 切开、各住各的卡。**Attention 不是这样。** MLA / GQA 的 KV 是按这条请求、按已经生成的序列长出来的，做注意力时必须看到**自己那批请求的完整 KV**。你不能把一条请求的 KV 拆去另一张卡上的"注意力专家"——注意力这边没有专家可切。

于是同一层里叠了两套并行：

- **Attention**：按 batch 切开（DP）。每张卡算自己分到的那几条请求，KV 留在本地，不跨卡。
- **MoE**：按 expert 切开（EP）。token 算完注意力之后，按路由飞到专家所在的卡，算完再飞回来。

DeepSeek-V3 的 prefill 写成 **TP4 + SP + DP8**，decode 写成 **TP4 + SP + DP80**，指的都是 attention 这一侧。为什么 EP 变大时 DP 也跟着变大？在常见的实现里，专家并行的规模满足 `EP = TP × DP`（vLLM 的 `--enable-expert-parallel` 就是这么算的）。多出来的卡主要用来**多住专家**；attention 不能按专家切，就按 DP 复制一份，让每张卡继续看着自己那几条请求的 KV。

> **类比**：分店可以各做各的家常菜（纯 DP）。后厨一旦改成"凉菜归凉菜组、热菜归热菜组"（EP），前厅点单却还是按桌走——每一桌的菜单（KV）必须留在自己那一桌的服务员手里。两套工位叠在同一班次里：专精的是菜，按桌分开的是点单。

## 五、稀疏起来：专家并行 EP

Dense 模型的 FFN 是一整块大矩阵，用 TP 切开是划算的：切片仍然够大，all-reduce 摊得下去。MoE 把这块换成**很多个小专家**，每个 token 只激活其中 top-k 个。以 DeepSeekMoE 为例：

- 整网 **671B** 参数，一次前向大约只激活 **37B**——稀疏比大约 1:18。
- 每层有 **1 个共享专家**（每个 token 必走）和 **256 个路由专家**（每个 token 只选 **top-8**）。

单个 routed expert 的 GEMM 已经很小。如果还用 TP 把它再切成 4 份、8 份，每份更碎，却仍要做 all-reduce。通信占比会高到不划算。专家本身就是天然的并行粒度：**一张卡住若干个完整的 expert**，token 按路由飞过去，在那张卡上把这个专家算完。这就是 **专家并行 EP**。

![token 经 Gate 选 top-k 专家，dispatch all-to-all 送到专家所在 GPU，grouped GEMM 之后再 combine all-to-all 写回](/assets/posts/llm-inference-parallelism-moe/diagram-moe-dispatch.png)

一步 MoE 层可以拆成三段，每一段的通信都不一样：

1. **Gate / Router**。每个 token 对 256 个路由专家打分，选出 top-k（这里是 8）个，并得到一组权重。这一步通常还在本地，几乎不通信。
2. **Dispatch**。按"这个专家住在哪张卡"把 token **打包**，然后做一次 **all-to-all**：每张卡把自己手里要送给别人的 token 发出去，同时收齐别人送给自己专家的 token。
3. **Grouped GEMM + Combine**。各卡对自己 inbox 里的 token 做专家计算（同一张卡上的多个专家常常打成一次 grouped GEMM），再按权重加权，第二次 all-to-all **写回** token 原来所在的位置。

和 TP 的 all-reduce 对比，差别很硬。all-reduce 是"每人一份相同大小的数据，做一次求和"；all-to-all 是"每人发给每人的量取决于路由"。路由一均匀，每张卡的 inbox 差不多；路由一偏，有的卡被灌满，有的卡空手。**墙钟不看平均值，看最忙的那张卡。** 这就是下一节要同时处理的两件事：热点专家，以及 all-to-all 本身太贵。

> **类比**：不再围着一口锅炒，改成每个师傅只做自己那几道拿手菜。跑堂（dispatch）按菜单把半成品送到对应的灶，做好再收齐（combine）。菜单若总点同一道爆款，那位师傅会被点爆，别的灶却在空转。

## 六、all-to-all 贵 + 热点专家 —— DeepEP、EPLB 与 Dual-batch overlap

EP 把两个新问题同时放到台面上。它们不是先后发生的，而是同一套 all-to-all 上的两面：一面是**谁更忙**，一面是**通信本身怎么走**。

**问题一：热点专家。** 真实流量的路由几乎从来不是均匀的，更接近 zipf：少数专家吃掉大部分 token。你就算把 256 个专家整齐地每卡放 8 个，最热的那几个如果碰巧住在同一张卡上，这张卡就会变成整层的 straggler。DeepSeek-V3 的对策是两步一起做。

第一步，先承认热点存在，给最烫的专家做 **冗余副本（redundant experts）**。统计一段时间里谁被选得最多，把这些专家**再复制一份**放到相对空闲的卡上；新来的 token 发给当前更轻的那份副本。第二步，用 **EPLB（Expert Parallelism Load Balancer）** 在节点内重排专家，尽量让热的和冷的搭在一起，并且不要为此增加跨节点 all-to-all——跨节点比节点内贵得多。

数字可以对照着看。Prefill 部署了 **32 个冗余专家**，EP32 上每卡从 8 个 routed 变成 **8+1**。Decode 更进一步，把共享专家也当成一个"永远被选中的热专家"，64 张卡专门托管冗余和共享。LMSYS 后来在 96×H100 上做消融，EPLB 带来大约 **1.49× prefill / 2.54× decode** 的吞吐——decode 受益更大，因为 EP 更大、不均被放大得更厉害。实验 C 会把"只改放置"和"再加副本"这两步分开量。

![zipf 热点让单卡过载；EPLB 把最烫的专家复制到闲卡，墙钟跟最忙的卡走](/assets/posts/llm-inference-parallelism-moe/diagram-eplb.png)

**问题二：通信形态随阶段而变。** 就算专家放匀了，all-to-all 还在。Prefill 一次 dispatch 往往带着很长的一段序列，消息大，瓶颈在**带宽**；decode 每步只有一个新 token，消息小，瓶颈在**延迟**——启动一次通信、握手、绕过 CPU，本身就可能比有效载荷更贵。DeepEP 为此做了两种内核，而不是用一个内核应付两种流量：

- **高吞吐（normal）**：把管道打满，适合 prefill。代价是输出形状是 symbolic 的，和 CUDA Graph 合不来——CUDA Graph 要的是"这次和上次形状一样，才能把启动开销录下来重放"。
- **低延迟（low-latency）**：走 IB 点对点，再用 IBGDA（经由 NVSHMEM）把控制面也放到 GPU 上，适合 decode。形状稳定，能进 CUDA Graph。

关键限制写在 DeepEP / SGLang 的文档里，也很硬：**同一通信组不能同时跑两种内核。** 这就是[第四篇](/posts/llm-inference-pd-disaggregation/)的 PD 分离，在 MoE 上从"可选优化"变成**刚需**的原因。DeepEP 有一个 auto 模式，理论上能按负载切换；可一旦 prefill 和 decode 还挤在同一个引擎、同一组通信里，它选不了"prefill 走 normal、decode 走 low-latency"。拆成两个池之后，每个池绑死一种内核，这件事才做得到。

就算内核选对了，all-to-all 的时间往往仍和 grouped GEMM 同量级，藏不住。下一步是 **Dual-batch overlap / TBO（Two-Batch Overlap）**：把一个 batch 切成两个微批次，**一批在做 grouped GEMM 时，另一批正好在做 dispatch / combine**。墙钟从 `计算 + 通信` 变成大约 `max(计算, 通信)`。LMSYS 测到，prefill 在同等 token 数下 TBO 能再加 **27–35%** 吞吐；因为它把峰值激活也摊成了两半，单卡能吞的 token 上限还从 8k 抬到了 16k。decode 上 TBO 要更挑：batch 太小（例如每卡 32 token）时，切成两半会让 kernel 更不饱，甚至出现负收益。

![DeepEP 两种内核不能共存，必须 PD 分离；两个微批次交错后墙钟取 max(计算, 通信)](/assets/posts/llm-inference-parallelism-moe/diagram-deepep-overlap.png)

> **脚注**：DeepSeek 训练里的 **DualPipe** 是把前向和反向叠在同一条流水线上，用来藏 PP 的气泡。名字像，问题不是同一个。推理这边重叠的是**两个微批次的计算与 all-to-all**，不要和 DualPipe 混为一谈。

> **类比**：爆款菜复制一份到空闲灶（EPLB）；午餐和夜宵用两套完全不同的传菜节奏，不能共用一条对讲频道（DeepEP 两种内核，所以必须 PD 分离）；两个跑堂交错着送——一个在传、一个在炒（TBO）。

## 七、生产里怎么配：DeepSeek-V3 与开源框架

把前面的刀叠回去，就是 DeepSeek-V3 论文 §3.4 里的推理单元。最值得盯住的不是某一行的数字，而是：**prefill 和 decode 的并行度故意不一样。** 这正是第四篇说的"阶段专属优化"，落到 MoE 上的具体配法。

![Prefill：4 节点 / 32 GPU，attention TP4+SP+DP8，MoE EP32；Decode：40 节点 / 320 GPU，attention TP4+SP+DP80，MoE EP320](/assets/posts/llm-inference-parallelism-moe/diagram-pd-ep-scale.png)

|           | Prefill                            | Decode                        |
| --------- | ---------------------------------- | ----------------------------- |
| 最小单元  | 4 节点 / **32 GPU**                | 40 节点 / **320 GPU**         |
| Attention | **TP4 + SP + DP8**                 | **TP4 + SP + DP80**           |
| MoE       | **EP32**（每卡 8 routed + 1 冗余） | **EP320**（每卡 1 个 routed） |
| 冗余      | 32 个冗余专家                      | 64 张卡托管冗余 + 共享        |
| 通信      | 高吞吐内核 + 双微批次              | IB P2P + IBGDA                |

可以按行读一遍。Attention 两侧都钉死 **TP4 + SP**：节点内同步还便宜，再大就不划算，实验 A 会印证这一点。两边真正拉开的是 DP 和 EP。Prefill 算力密集，专家不必摊得太碎，**EP32** 就能喂饱计算，每卡还留得下 8 个 routed 加 1 个冗余。Decode 访存密集，权重和 KV 都在抢带宽，于是把专家摊到 **EP320**，每卡只住 1 个 routed——单卡上的权重更少，就能进更大的 batch，聚合起来的 HBM 带宽才够把 decode 喂饱。通信内核也跟着阶段走：prefill 打带宽，decode 走 IB 点对点和 IBGDA 压延迟。

数字本身不是教条。后来 DeepSeek 开源周的系统概述里，decode 单元收成过 **EP144 / 18 节点** 的写法，和论文的 EP320 不是同一版部署。硬件换一代、流量结构一变，具体 EP 会改。不变的是原则：**两阶段不要共用一套并行度。**

开源侧能复现到什么程度？LMSYS / SGLang 在 **96×H100**（12 节点）上做了一个缩小版：prefill 仍是 **EP32**，decode 用 **EP72**（大约是论文 decode 规模的一半）。相对同一资源上的 vanilla TP16，输出吞吐最高大约 **5×**；2k 输入时，单节点大约 52.3k input tok/s、22.3k output tok/s。框架上对应的开关，可以按"开了它在解决哪一节的问题"来记：

- **SGLang**：`--enable-deepep-moe` 换上 DeepEP 的 all-to-all；`--deepep-mode {normal,low_latency}` 给两个池各绑一种内核；`--enable-eplb` 做冗余和重排；`--enable-two-batch-overlap` 打开 TBO；`--enable-dp-attention` 让 attention 按 DP 走。PD 分离本身用 `--disaggregation-mode {prefill,decode}`。
- **vLLM**：`--enable-expert-parallel` 把 MoE 从"用 TP 切专家"改成"按专家切开"，EP 规模等于 `TP × DP`（只开 PP、TP=1 且 DP=1 时，这个开关不会生效）；`--all2all-backend` 选 `deepep_high_throughput` 或 `deepep_low_latency`；`--enable-eplb` 和 `--enable-dbo` 分别对应 EPLB 与 Dual-batch overlap。

有一件事两边都成立：没有 PD 分离就强行让同一通信组跑两种 DeepEP 内核，组会卡住。这不是哪个开关没打开，是通信组的语义不允许。

## 八、三组无需 GPU 的实验

下面三组实验都在 CPU 上跑，`SEED=42` 可复现。它们**不是**端到端 GPU benchmark，也不会复现 LMSYS 那组 5×。实验 A 是一层 decoder 的解析墙钟：计算按 `1/TP` 缩放，通信按 ring all-reduce 估价。实验 B 直接代入流水线气泡公式。实验 C 是 zipf 路由下的 GPU 负载，不管字节数，只看谁最忙。目的是把正文里的数量级钉死，方便对照，不代替真机上的 kernel 剖面。脚本在 `diagrams/parallelism-moe/sim_experiments.py`。

### 实验 A：TP 计算 vs all-reduce —— NVLink 还在加速，IB 上通信反超

一层做两次 all-reduce（attention 出口 + MLP 出口）。激活按 `hidden × tokens × 2` 字节算（bf16，hidden=8192）。Ring all-reduce 是 `2(t−1)` 跳，每跳发送 `nbytes/t`。NVLink 按 400 GB/s、2 µs/hop 估价；IB 按 50 GB/s、10 µs/hop。Prefill 取 4096 token，TP=1 时计算 6.0 ms；decode 取 64 路并发，TP=1 时计算 0.55 ms——decode 这一侧按访存主导给了一个粗口径，不假装是算力屋顶。

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

Prefill 在 NVLink 上从 TP1 到 TP16 仍有 **5.3×**，计算还盖得住通信。同样一层换到 IB 上，最好的点是 TP4，也只比单卡快 6%；TP16 已经略慢于 TP1——跨节点做大 TP，等于花钱买同步。Decode 对延迟更敏感：NVLink 的甜点在 **TP8**，再往上 hop 数把延迟项抬起来，TP16 反而回退；IB 上 TP16 是 TP1 的 **1.3 倍慢**。这就是正文里那句"TP 停在节点内、attention 钉在 TP4"的数量级来源：不是不能切到 8 或 16，是一出节点，切得越大越亏。

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

训练可以把 `m` 堆到 32 以上，PP=8 还能到 82%，气泡被大 batch 稀释掉了。**在线 decode 的 `m` 往往接近 1**，同一张表上 PP=8 只剩八分之一的卡在干活，PP=16 更只剩 6%。这个公式完全不涉及 GPU 型号，却足够解释为什么推理单元更愿意把跨节点预算花在 EP 而不是 PP 上：EP 至少让每张卡都有专家可算，PP 在 batch=1 时会让大部分阶段空转。

### 实验 C：zipf 路由 vs EPLB —— 冗余副本把 straggler 压下来

256 个专家、32 张 GPU（EP32，每卡先放 8 个专家）、top-8、8 万 token，路由按 zipf（`a=1.15`）。对比三种放置。第一种最朴素：专家 0–7 连续放在 GPU 0——而 zipf 里编号越小越热，等于把最烫的一簇堆在同一张卡上。第二种按热度贪心装箱，热的优先放到当前最轻的卡，但每个专家仍然只有一份。第三种在装箱后再给最热的 32 个专家加一份冗余。每个 token 发给该专家当前最轻的那份副本。墙钟用 `max / mean` 近似：均值是"如果完全均匀该是多少"，最大值是 straggler。

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

只改放置、不加副本，不均已经从将近 14 倍掉到 3.7 倍——说明"热专家碰巧住在一起"本身就很伤。再复制 32 个最烫的专家，straggler 降到 2.1 倍。这和 LMSYS 说的"EPLB 在大规模下把吞吐拉起来"是同一件事的负载侧：墙钟不看平均值，看最忙的那张卡。仿真没有建模 all-to-all 的字节数，也没有 kernel 启动，所以**不能**把 13.96 → 2.11 直接读成 1.49× / 2.54× 那些端到端数字。那些来自真机，见第七节。这里只说明：冗余副本解决的是"谁更忙"，不是"通信有多贵"。

## 九、总结与延伸

并行策略的主线，还是那条"发现问题 → 引入优化 → 带出新问题"的链子：

- **一张卡装不下** → **TP（Megatron 列/行切分）**。层内 all-reduce，体积 `(t−1)/t`，最好停在 NVLink 域。
- **出节点太贵** → **PP**。改切层，气泡是 `(p−1)/(m+p−1)`；batch=1 时几乎串行。
- **还要吞吐** → **DP / DP Attention**。推理不梯度同步；MoE 下 attention 按请求复制，专家按 EP 切。
- **稀疏 FFN 不值得再用 TP 切** → **EP**。dispatch / combine 两次 all-to-all。
- **all-to-all 贵，路由还不均匀** → **DeepEP 两种内核 + EPLB 冗余副本**。
- **通信还是和计算同量级** → **Dual-batch overlap**；两种内核不能住在同一通信组里 → **咬合第四篇的 PD 分离**。

一句话带走：**并行策略就是给每一段计算选一刀——TP 切 hidden，PP 切层，DP 切请求，EP 切专家。MoE 把最后一刀推到跨节点之后，真正决定墙钟的不再是平均 FLOPs，而是 all-to-all 走哪条线、以及最烫的那个专家住在哪。**

延伸阅读：下一篇会把负载换成 **long-CoT / 推理模型**。思维链把 decode 拉得很长，KV 占得更久，straggler 和调度的形状都会变。再往后是 GPU 架构与 attention kernel；本篇刻意没展开的 Context Parallel，会放在那里一起讲。若还想往 MoE 上再砍一刀——把 attention 和 FFN 拆到两类机器上——见 MegaScale-Infer（ByteDance）：它是 EP 之后的另一次分离，本篇只点到为止。

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
