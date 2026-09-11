---
author: xiangel
pubDatetime: 2026-09-11T02:30:00Z
title: "大模型的各种并行：从张量切分到上下文并行"
slug: llm-inference-parallelism-moe
featured: true
draft: false
tags:
  - 大模型推理系统
  - 并行策略
  - 上下文并行
  - MoE
  - LLM
description: 一张卡装不下，就得把模型切开——但并行远不止 TP。用 decoder 结构图对照 TP / SP / PP / DP / CP / EP 各切哪一段，并写清每刀对显存、通信、延迟和缓存的影响与硬约束。附 MoE 的 DeepEP / EPLB，以及一组无需 GPU、可复现的仿真。
---

并行这两个字，在大模型圈子里被用滥了。有人说的 TP，有人说的 SP，还有人把 DeepSpeed 的 Ulysses 也叫序列并行。名字听着像一家，切的维、通信形态、适用场景完全不是一回事。这篇就把桌上能见到的刀一次摊开。

如果你把一个 70B、乃至 DeepSeek-V3 那种 671B 的模型搬上线，很少能靠"再买一张更大的卡"收场。问题会按这个顺序一层层冒出来：

1. **一张卡装不下权重** → 沿 hidden 切开，这是 **TP**。
2. **TP 切完，激活显存还在** → 把 LayerNorm 按序列切开，这是 **SP**。
3. **出了节点再做大 TP 不划算，层又太多** → 沿深度切开，这是 **PP**。
4. **一份模型吞吐不够** → 复制几份接单，这是 **DP**。
5. **序列太长，注意力的 KV 比权重大** → 沿 token 切开注意力，这是 **CP**（推理里还要再分成 DCP / PCP）。
6. **模型变成 MoE，单个 expert 已经很小** → 再切 TP 不划算，改切专家，这是 **EP**。

这是本系列第六篇。[第一篇](/posts/from-causal-lm-to-inference-system/)把"权重怎么切到多卡"点过题；[第四篇](/posts/llm-inference-pd-disaggregation/)说了 prefill 想要小 TP、decode 想要大 TP，但没解释这些缩写在切什么；[第五篇](/posts/llm-inference-scheduling-distributed/)把请求派到了某台机器。这一篇钻进那台机器所属的并行组，问三件事：**Model Executor 按哪一维切？切完显存、通信、延迟变成什么样？哪些刀其实不是一回事？**

> **说明**：本篇讲的是**模型怎么切开**，不是第五篇那种跨实例的请求路由。训练里的 DualPipe、ZeRO 会点明"这是训练刀，推理不用"，避免和推理的 Dual-batch overlap、DP 混在一起。

全程用一个类比贯穿：**大厨团队的分工**。一张卡是一个灶台，一层计算是一道菜。TP 是几个人围着同一口锅各炒一角；SP 是案板上的配菜按份分开，少占地方；PP 是备菜、热锅、装盘排成流水；DP 是再开几间分店；CP 是一篇超长菜单撕成几段，各灶先处理自己那段，再把配料转一圈；EP 是每个师傅只做拿手菜，跑堂按菜单送半成品。还是那条主线：**每暴露一个问题，就引入一种优化，又带出新问题。**

## Table of contents

## 一、先把所有刀摊在桌上

推理里真正会叠在一起的，是下面这六刀。训练里还有几把，名字容易混进来，表后再单列。

![六刀全景：TP / PP / DP / SP / CP / EP 各切哪一维；ZeRO、VPP、Ulysses 同名不是同一刀](/assets/posts/llm-inference-parallelism-moe/diagram-taxonomy.png)

| 并行   | 切开的维                   | 典型通信                          | 主要出现在  | 最怕什么                       |
| ------ | -------------------------- | --------------------------------- | ----------- | ------------------------------ |
| **TP** | hidden / 权重（含词表）    | 层内 **all-reduce**               | 训练 + 推理 | 跨节点；小消息延迟             |
| **SP** | LayerNorm / Dropout 的序列 | all-gather / reduce-scatter       | 训练 + 推理 | 被叫成"序列并行"的其实是 CP    |
| **PP** | layers                     | 阶段边界 **点对点**               | 训练为主    | 气泡；batch=1 灌不满           |
| **DP** | batch / 请求               | 推理里通常**没有**                | 训练 + 推理 | 前缀缓存被摊薄                 |
| **CP** | 序列，**含注意力本身**     | Ring 传 KV，或 Ulysses all-to-all | 长上下文    | 和 SP 同名；DCP / PCP 还要再分 |
| **EP** | MoE experts                | dispatch / combine **all-to-all** | MoE         | 热点专家；两种通信内核不能共存 |

读这张表时，盯"切开的维"比盯缩写有用。TP 切的是矩阵的宽，PP 切的是网络的深，DP 切的是请求的份数，SP 切的是**不算注意力的那截激活**，CP 切的才是**注意力看见的那段序列**，EP 切的是专家。**通信的形状，比"用了几张卡"更能决定这刀能不能赚钱。**

先把还没切的模型放在桌上。Decoder-only 就是 Embed → N 层（RMSNorm → Attention → FFN）→ LM Head，激活一路带着 `[B, S, H]`。六刀分别切这几个字母，或把 FFN 换成一排专家。

![未切开时一张卡要装下全部权重、KV 和激活；B / S / H / 层号 / 专家是五条可切的缝](/assets/posts/llm-inference-parallelism-moe/diagram-model-backbone.png)

同一份 Embed → 层栈 → Head，六刀切完是六种样子。颜色表示 GPU：哪一块换了颜色，就是这刀切到的结构。

![TP 竖切每层矩阵；PP 把层栈横切给不同卡；DP 复制整网；SP 只切 Norm 的序列；CP 切 Attention 的 token；EP 只拆 FFN 里的专家](/assets/posts/llm-inference-parallelism-moe/diagram-six-slices.png)

再把镜头推进**一层**。PP 决定这一层住在哪张卡，DP 决定这张卡接哪几条请求；层内的刀落在不同算子上。SP 到 Attention 门口就停，CP 才走进注意力；Dense 的 FFN 用 TP 切 hidden，换成 MoE 就改切专家。

![一层 decoder 的数据流：DP 切 B，SP 切 Norm 的 S，TP 切线性层的 H，CP 切 Attention 的 S，EP 切 MoE 专家](/assets/posts/llm-inference-parallelism-moe/diagram-decoder-ops.png)

切开不是免费的。下面这张表和后面各节的「影响与约束」说的是同一件事：省了哪一档显存，立刻多出哪一笔通信，以及什么条件下这刀会反过来变慢。

| 并行   | 每卡权重                     | 每卡 KV                        | 同步节奏                      | 硬约束                                   | 推理上最明显的副作用                         |
| ------ | ---------------------------- | ------------------------------ | ----------------------------- | ---------------------------------------- | -------------------------------------------- |
| **TP** | `/ t`                        | 常不降；GQA 头不够时还**复制** | 每层 **2× all-reduce**        | `t` 整除 hidden / 头数；最好停在 NVLink  | decode 小消息被延迟打穿                      |
| **SP** | 同 TP                        | **不降**                       | all-gather / RS 换掉一部分 AR | 必须已经开 TP                            | 推理 decode 激活本来就小，显存收益远小于训练 |
| **PP** | `/ p`（只持有若干层）        | 只留本地层，约 `/ p`           | 阶段边界点对点                | 层数最好能整除；`m≈1` 时利用率 `1/p`     | TTFT 被灌流水线拉长；decode 大量空转         |
| **DP** | × 副本（模型仍要装得下一份） | × 副本                         | 推理里通常没有                | 装不下的模型，加副本也装不下             | 前缀缓存被摊薄                               |
| **CP** | 基本不变                     | `/ C`                          | 每次注意力都通信              | Ulysses：头数 ≥ C；DCP ≤ `tp / KV头`     | decode 每步都要扫别人的 KV；PCP 才加卡       |
| **EP** | 专家权重 `/ ep`              | 跟 attention 侧走              | 每层 2× all-to-all            | 专家数应能整除；两种 DeepEP 内核不能同居 | 热点专家；必须 PD 分离才能绑对内核           |

卡数怎么乘，也是一条硬约束。常见的 world size 是：

```text
#GPU = TP × PP × DP × PCP
```

**DCP 不出现在乘法里**——它只在已有 TP 组里交错切 KV，不新开卡。vLLM 里开了 expert parallel 之后，`EP = TP × DP`。SP 不能单独开，它寄生在 TP 上。所以"开了 32 卡"这句话本身没有信息量：可能是 TP8×DP4，也可能是 EP32，通信形态完全不同。

六刀也可以叠。DeepSeek-V3 线上就是：**attention 走 TP + SP + DP，MoE 走 EP**。长上下文再叠 CP。不是单选，是给每一段计算挑刀。

还有几把不要塞进同一张菜单：

- **ZeRO / FSDP**：把优化器状态、梯度、有时连参数都按 DP 维切开。推理没有优化器，这把刀不上桌。
- **虚拟流水线 VPP**：同一张卡上交错多个 PP 阶段，用来减气泡，几乎是训练技巧。
- **DualPipe**：训练里把前向和反向叠在一条流水线上。推理的 Dual-batch overlap 是另一件事。
- **MegaScale-Infer** 那种 attention / FFN 拆到两类机器：不是第七种切维，是 EP 之后再做一次**模块分离**，第九节末尾会点到。

> **类比**：后厨工位图。围着一口锅分工（TP），案板配菜分开摆（SP），按工序排流水（PP），再开分店（DP），超长菜单撕成几段互相转配料（CP），按拿手菜专精（EP）。工位切错了，不是人手不够，是**传菜的走位**把时间吃掉了。

## 二、一张卡装不下：张量并行 TP

最朴素的需求：权重比一张卡的显存大。一张 H100 是 80 GB，70B 的 bf16 权重就要 140 GB，还没算 KV Cache。所以 Megatron-LM 的做法是：不要把整层交给一张卡，把每一层的线性层沿 **hidden** 切开，让几张卡**同时算同一层的不同切片**。

一个线性层写成 `Y = X W`。切开的方式有两种，而且在 Transformer 里几乎总是**成对出现**：先 Column Parallel，再 Row Parallel。

![列并行按输出维切开、无需通信；行并行按输入维切开，之后一次 all-reduce，体积正比于 (t−1)/t。节点内走 NVLink，跨节点走 IB 会把加速吃掉](/assets/posts/llm-inference-parallelism-moe/diagram-tp-allreduce.png)

1. **列并行（Column Parallel）**：按**列**切 `W`，也就是按输出维切开。每张卡都拿到完整的 `X`，各自算出 `Y` 的一块。这几块在逻辑上拼起来就是完整输出，**这一步没有通信**。Attention 的 QKV、MLP 的第一段（升维）走这条。
2. **行并行（Row Parallel）**：按**行**切 `W`，也就是按输入维切开。上一层刚切出来的那一块 `X`，正好对得上这一层的一块 `W`。每张卡算出一份**部分和**，再做一次 **all-reduce**，才能得到完整的 `Y`。Attention 的 output projection、MLP 的第二段（降维）走这条。

之所以要成对，是为了少通信。升维用列并行，中间那截激活可以就地留给下一层；降维用行并行，只在层的出口同步一次。一层 decoder 通常就是这样两次 all-reduce：一次在 attention 出口，一次在 MLP 出口。

Ring all-reduce 的数据量正比于 **(t−1)/t**。这个式子有个不太直观的后果：TP 从 1 到 2，通信从 0 一下子跳到"全量的一半"；再往上，增量变缓，但每一次都要等所有卡对齐。墙钟既取决于最慢的那张卡，也取决于这次同步走的是哪条线——同一机箱里的 NVLink，还是机箱外面的 InfiniBand。

所以 TP 有一条硬约束：**尽量停在 NVLink 域里**。常见的是单机 8 卡，或者 NVLink 连起来的超节点。H100 节点内 NVLink 是数百 GB/s 这个量级；一出节点，InfiniBand 掉到数十 GB/s，延迟也高一个数量级。实验 A 会把这条线画出来：同样算一层，NVLink 上加大 TP 还在加速，IB 上通信很快反超计算。这也解释了 DeepSeek-V3 为什么把 attention 的 TP **钉死在 4**——论文原话就是用小 TP 限制通信开销。再大，切出来的计算更碎，同步却更密。

词表和 LM Head 通常也跟着 TP 切，叫 **vocab parallel**：embedding 按词表维切开，最后的 softmax 只在各卡的那一段词表上做，再做一个并行的采样。它不是独立的第七刀，是 TP 在模型两头的延伸。

**影响与约束。** 权重按 `1/t` 变薄，这是 TP 能把 70B 塞进多张 80GB 卡的原因。但有三笔账立刻上门。

第一笔是**通信**。一层两次 all-reduce，体积正比于 `(t−1)/t`。Prefill 序列长、算得动，NVLink 上加大 TP 往往还在加速；decode 每步只推一个 token，消息小、次数密，延迟项比带宽项更刺。实验 A 里 IB 上把 TP 开到 16，一层已经慢于单卡——不是算力不够，是报数报不过来。

第二笔是 **KV**。很多人以为"切了权重，KV 也会变薄"。GQA / MLA 的 KV 头本来就少，`t` 大于 `num_kv_heads` 时，实现会把同一份 KV **复制**到多张卡上。权重省了，KV 反而更肥。这正是第六节 DCP 要砍掉的那截复制。

第三笔是**形状**。hidden 和注意力头必须能被 `t` 整除；切得太碎，GEMM 也不饱，通信占比更高。CUDA Graph 还要求这次和上次的 collective 形状一样，动态 batch 会把图录不进去。

硬约束可以记三条：尽量停在 NVLink 域；`t` 要整除头数；decode 不要为了显存把 TP 开到跨节点。DeepSeek-V3 把 attention 钉在 TP4，就是这三条叠在一起的结果。

> **类比**：几个人围着同一口锅炒同一道菜，每下一刀都要互相报一声"我这边好了"（all-reduce）。灶台挨着（NVLink），报一声很快；灶台隔了两条街（IB），报一声的时间比炒菜还长。

## 三、TP 切完激活还在 —— 序列并行 SP

TP 只切了线性层的权重。LayerNorm、Dropout 这些算子几乎不吃权重，却把**整段序列的激活**摊在显存里。训练时这一点尤其疼：激活要留给反向，SP 就是为这件事发明的。

**Megatron 的序列并行 SP（Sequence Parallel）** 做法是：把这些算子按序列维切开，每张卡只留自己那一段 token 的激活，用 all-gather / reduce-scatter 替换掉一部分 all-reduce。换来的是激活显存再薄一档。它**要求已经开了 TP**，而且**不切注意力计算本身**——Q 还是要和整段 KV 见面。DeepSeek-V3 的 attention 写成 **TP4 + SP**，指的就是这套组合。

这里必须把三个常被混用的名字拆开：

| 名字                             | 切什么                          | 注意力怎么算                            |
| -------------------------------- | ------------------------------- | --------------------------------------- |
| **Megatron SP**                  | 只切 LayerNorm / Dropout 的激活 | 注意力仍在 TP 组里，不按序列摊 KV       |
| **Ring CP**                      | 切完整序列，含 QKV              | 环形把别人的 KV 传过来                  |
| **Ulysses（DeepSpeed 也叫 SP）** | 切完整序列                      | 两次 all-to-all，改成按头切开再算注意力 |

后两个才是第六节的 **CP**。如果有人说"我们开了序列并行"，先问一句：切的是 LayerNorm，还是注意力。

**影响与约束。** SP 省的是 LayerNorm / Dropout 那截**激活**显存。训练时激活要留给反向，这一档非常疼，SP 就是为这件事发明的。推理的 decode 每步只有一个新 token，激活本来就小，显存收益远小于训练——DeepSeek-V3 仍写成 TP4+SP，主要是和 Megatron 的实现绑在一起，顺手把激活路径也切开，不是因为 decode 激活爆了。

硬约束有两条，都容易踩错。第一，**必须已经开 TP**，SP 不能单独存在。第二，注意力前要把切开的序列 **all-gather** 拼回来，Q 仍然看见整段 KV，所以 **KV 并不变少**。把它当成"长上下文方案"会用错刀：长上下文该找 CP。

> **类比**：TP 是几个人围着一口锅；SP 只是把案板上的配菜按份分开摆，锅还是那一口。真正把长菜单撕开、各炒一段的，是 CP。

## 四、出节点太贵：流水线并行 PP

TP 出了节点就不划算。可模型还是太大：8 张卡的 NVLink 域依然装不下整份权重。下一刀改切 **layers**——把连续若干层交给一个**阶段（stage）**，激活算完这一段，再递给下一个阶段。这就是 **流水线并行 PP**。

![p=4 个阶段、m=8 个微批次的流水线；斜线格是气泡。气泡比例 (p−1)/(m+p−1)；batch=1 时利用率只剩 1/p](/assets/posts/llm-inference-parallelism-moe/diagram-pp-bubble.png)

通信形态变便宜了：不再是每层做一次全集同步，只在阶段边界**点对点**传一次激活。新问题换成了 **气泡（bubble）**。流水线要先灌满才会转起来：第 0 个阶段开始算第 1 个微批次时，后面的阶段只能空手等；最后一个微批次离开第 0 个阶段之后，前面的阶段又要空转到收尾。图里的斜线格，就是这两头空出来的时间。

Narayanan 等人给出的理想利用率是 `m / (m + p − 1)`。`p` 是阶段数，`m` 是微批次（micro-batch）数。气泡占比则是 `(p − 1) / (m + p − 1)`。`p=8`、`m=1` 时，利用率只剩 **12.5%**——八张卡里几乎只有一张在干活，其余在等。训练可以堆很大的 global batch，再切成许多微批次，把 `m` 灌满；**在线 decode 常常一步只推一个 token**，你没有那么多微批次可切。就算硬切，也会把单条请求的延迟拉长：用户要等整条流水线灌完，才看到这个 token。

训练里还有两件专门减气泡的事，推理几乎用不上。**1F1B** 把后面微批次的前向，和前面微批次的反向叠在一起；**虚拟流水线 VPP** 让同一张卡交错持有多个非连续阶段，把气泡摊得更碎。它们吃的是训练的大 batch 加反向。在线 decode 没有反向，也没有那么多微批次，这两招帮不上忙。

**影响与约束。** PP 让每张卡只持有若干层，**权重和 KV 都按阶段变薄**——这是它相对 TP 少被提到的优点：KV 只为本地那些层而存在。通信也便宜，只在阶段边界点对点传一份 `[B, S, H]` 的激活，不必每层全集同步。

账单在利用率上。公式 `m / (m + p − 1)` 不留情：在线 decode 常常 `m≈1`，八张卡里同时只有一张在算这一步，其余在等。用户侧的感觉是 **TTFT 变长**——第一个 token 要等流水线灌到最后一阶段。你若为了填气泡去切微批次，单条请求的延迟会被切得更碎、更长。层数最好能被 `p` 整除，否则短的阶段在等长的阶段，气泡比公式还大。

所以生产推理里，PP 是"模型跨不出 NVLink、又还不是 MoE"时的备选，不是 decode 的主方案。DeepSeek-V3 的推理单元**不用 PP**，把跨节点预算留给 EP：那种通信更散，但至少每张卡都有专家在算。

> **类比**：备菜、热锅、装盘排成四个工位。一桌十道菜（`m` 大），流水线转起来每个工位都忙；只来一份例汤（batch=1），后面三个工位只能干等。训练更像婚宴，可以提前备一大桌；推理经常是单人套餐，客人还站在窗口等这一份。

## 五、还要吞吐：数据并行 DP，以及 MoE 入口的 DP Attention

TP 和 PP 解决的是"**一份模型怎么切开**"。切完之后，这一份模型一次仍然只能吞一个、或一小批请求。流量再大，单份并行组也会先被请求队列堵住。下一刀是 **复制**：同样的（或已经按 TP 切开的）模型多放几份，各接各的单。这就是 **数据并行 DP**。

训练里的 DP 每一步都要 **all-reduce 梯度**，副本之间同步很重。**推理里没有梯度**，副本之间默认不说话，所以它看起来最便宜。便宜的代价在别处：每份副本各自缓存各自的 KV。你加的副本越多，同一个前缀就越容易被摊到不同机器上，[第二篇](/posts/kv-cache-paged-attention-and-prefix-caching/)的前缀缓存、[第五篇](/posts/llm-inference-scheduling-distributed/)的缓存感知路由都会被稀释。DP 在 serving 里的正确用法，常常不是再复制一份完整的 70B，而是和别的并行叠在一起，只复制那些**必须按请求切开**的部分。

不要把训练里的 **ZeRO / FSDP** 算进推理的 DP。ZeRO-1 切的是优化器状态，ZeRO-2 再切梯度，ZeRO-3 / FSDP 连参数也按 DP 维切开。推理没有优化器，也没有梯度；参数如果还要切，走的是 TP、PP 或 EP。有人说"我们开了 ZeRO-3"，那是训练刀。

MoE 把这件事逼出了一个专门变体：**DP Attention**。

原因在于同一层里其实有两种计算。MoE 的 FFN 是稀疏的，256 个专家里每个 token 只走 top-k 个，专家可以按 EP 切开、各住各的卡。**Attention 不是这样。** MLA / GQA 的 KV 是按这条请求、按已经生成的序列长出来的，做注意力时必须看到**自己那批请求的完整 KV**。你不能把一条请求的 KV 拆去另一张卡上的"注意力专家"——注意力这边没有专家可切。

于是同一层里叠了两套并行：

- **Attention**：按 batch 切开（DP）。每张卡算自己分到的那几条请求，KV 留在本地，不跨卡。
- **MoE**：按 expert 切开（EP）。token 算完注意力之后，按路由飞到专家所在的卡，算完再飞回来。

DeepSeek-V3 的 prefill 写成 **TP4 + SP + DP8**，decode 写成 **TP4 + SP + DP80**，指的都是 attention 这一侧。为什么 EP 变大时 DP 也跟着变大？在常见的实现里，专家并行的规模满足 `EP = TP × DP`（vLLM 的 `--enable-expert-parallel` 就是这么算的）。多出来的卡主要用来**多住专家**；attention 不能按专家切，就按 DP 复制一份，让每张卡继续看着自己那几条请求的 KV。

**影响与约束。** 纯 DP 的吞吐近似随副本线性涨，推理里副本之间也不梯度同步，看起来最香。但加副本**不会**让一个装不下的模型突然装得下——每份仍然要放下自己那份权重（或 TP 切片）和自己那份 KV。副作用在缓存：同一条系统提示被摊到不同副本，[第二篇](/posts/kv-cache-paged-attention-and-prefix-caching/)的前缀命中、[第五篇](/posts/llm-inference-scheduling-distributed/)的亲和路由都会被稀释。DP 开得越大，调度越难"把熟客送到同一张桌子"。

DP Attention 还改写了"DP 不说话"这条。token 从"按请求切"的 attention 走进"按专家切"的 MoE 时，必须做一次集体通信，把各卡上的 token 交给持有对应专家的卡。所以 MoE 上的 DP 不是免费的复制，是 EP 的入口税。

> **类比**：分店可以各做各的家常菜（纯 DP）。后厨一旦改成"凉菜归凉菜组、热菜归热菜组"（EP），前厅点单却还是按桌走——每一桌的菜单（KV）必须留在自己那一桌的服务员手里。两套工位叠在同一班次里：专精的是菜，按桌分开的是点单。

## 六、序列太长：上下文并行 CP

前几刀切的是权重、层、请求。序列一旦拉到 32k、128k，新问题换了对象：**KV Cache 比权重大**。一张卡也许还装得下 70B 的切片，却装不下这条请求已经生成的 KV。TP 帮不上这个忙——GQA / MLA 的 KV 头本来就少，TP 加大以后，每张卡反而要把同一份 KV 再复制一遍。第三节的 SP 也帮不上：它只切 LayerNorm，不切注意力。

这一刀叫 **上下文并行 CP（Context Parallel）**。把序列按 token 切开，每张卡只负责一段；注意力要用到别人那段 KV 时，再把 KV 转过去。切的是**注意力看见的那段序列**，所以和 Megatron SP 不是一回事。DeepSpeed 把 Ulysses 也叫 Sequence Parallelism，名字撞了，维没撞。

![Ring 环形传 KV；Ulysses 两次 all-to-all 换成按头计算；DCP 不增加 GPU，只在 TP 组里交错切 KV](/assets/posts/llm-inference-parallelism-moe/diagram-context-parallel.png)

通信有两条主流路。

**Ring CP / Ring Attention。** Megatron 默认走这条。每张卡先拿自己那段 QKV，先和本地 KV 做注意力；再把 KV 沿环传给邻居，和下一块再做一次；转满一圈，就等价于看见了完整序列。一层的通信体积大约是 `(1 − 1/C) · S · d_kv · 2`（K 和 V），bf16 再乘 2 字节。GQA / MLA 让 `d_kv` 远小于 `hidden`，环传比把整段激活 all-reduce 便宜一个数量级。计算和通信还能重叠：本块注意力在算时，下一块 KV 已经在路上。

**Ulysses（DeepSpeed；Megatron 里也叫 a2a）。** 两次 all-to-all 换轴：先按序列切开拿到 QKV，再换成按注意力头切开——每张卡拿到**完整序列、但只有一部分头**，注意力在本地做完，再 all-to-all 换回去。体积大约是 `2 · S · hidden / C`。一次集体通信，NVLink 上往往比一圈 P2P 更快；约束是头数得 ≥ CP 度。跨节点则常退回 Ring。

实验 D 会把这两种载荷和 TP 的激活同步画在同一张图上：序列一长，该切 CP，而不是把 TP 再加大。

推理还要再拆一刀，因为 prefill 和 decode 的 KV 形状完全不同。

**Decode Context Parallel（DCP）。** 不增加 GPU 数。它复用现有 TP 组，按 token 交错把 KV 切开：`token i` 住在 `i % dcp` 那张卡上。GQA 头不够切时，TP 会把 KV 复制 `tp / H` 份；DCP 就是来砍掉这段复制的。vLLM 的开关是 `--decode-context-parallel-size`（也写 `-dcp`），上界大约是 `tp_size / num_kv_heads`。开得越大，KV 越省，通信越重。

**Prefill Context Parallel（PCP）。** 才真正加卡。一条超长 prompt 按序列切开，用来压 TTFT。world size 变成 `TP × PCP`。vLLM 对应 `--prefill-context-parallel-size`。它和 DCP 正交，不要共用一个开关：一个改的是"decode 时 KV 怎么摊在已有卡上"，一个改的是"prefill 要不要多叫几张卡来切序列"。

**影响与约束。** CP 打的是 KV：按 `C` 切开之后，每卡大约只留 `1/C` 的缓存，32k、128k 才装得下。权重几乎没变薄，所以它**补的是 TP 做不到的那一刀**，不是 TP 的替代品。实验 D 把数量级钉死：切错维，字节数差一个数量级。

通信是每步都要付的。Ring 能和计算重叠，但 hop 随 `C` 涨；Ulysses 一次 all-to-all 在 NVLink 上更快，**头数必须 ≥ CP 度**，跨节点常退回 Ring。Decode 更苛刻：每步只有一个新 Q，却要扫很长的 KV。DCP 复用 TP 组、不加卡，上界大约 `tp / num_kv_heads`，开太大，省下的 HBM 会被通信吃回去。PCP 加卡压 TTFT，会乘进 world size。因果掩码没有被切掉：你分得再碎，逻辑上还是要看完整前缀。

> **类比**：一篇超长菜单撕成几段。Ring 是邻桌把配料转一圈，你才能把整道菜的味道对齐；Ulysses 是先按段拆单，再改成按菜系拆——每个人拿到完整菜单，但只负责一类菜。DCP 不新开灶，只是把已经坐满的那一桌，按单号单双号把菜单分开夹；PCP 才是再叫几个人来一起备这道超长前菜。

## 七、稀疏起来：专家并行 EP

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

**影响与约束。** 专家权重按 `1/ep` 变薄，decode 才能把 671B 摊到每卡一个 routed expert，腾出 HBM 去喂更大的 batch。单个 expert 已经很小，再叠加 TP 会把 GEMM 切得更碎，通信占比更高——这是"稀疏起来就不要用 TP 切 FFN"的原因。

新账单有三笔。第一，每层两次 all-to-all，体积随路由变，不再是 TP 那种固定 `(t−1)/t`。第二，真实路由接近 zipf，放置稍有不慎就会出现 straggler；冗余副本能压不均，但不能把通信变免费。第三，DeepEP 的高吞吐内核和低延迟内核**不能住在同一通信组**，prefill / decode 必须拆开才能各绑一种。vLLM 里 `EP = TP × DP`，所以放大 EP 会连带放大 attention 侧的 DP，前缀缓存和调度都要跟着改。

> **类比**：不再围着一口锅炒，改成每个师傅只做自己那几道拿手菜。跑堂（dispatch）按菜单把半成品送到对应的灶，做好再收齐（combine）。菜单若总点同一道爆款，那位师傅会被点爆，别的灶却在空转。

## 八、all-to-all 贵 + 热点专家 —— DeepEP、EPLB 与 Dual-batch overlap

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

## 九、生产里怎么配：DeepSeek-V3 与开源框架

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

选刀的顺序，其实就是开篇那条问题链，只是加上约束之后可以写成检查清单：

1. **一份模型装不进一张卡？** 先 TP，停在 NVLink；头数整除不了就不要硬开。还不够再叠 PP，但要接受 decode 的气泡。
2. **KV 比权重大、序列很长？** 不要再加大 TP——GQA 会复制 KV。改 CP；decode 用 DCP 砍复制，prefill 长 prompt 才用 PCP 加卡。
3. **流量把一份并行组堵住了？** 再叠 DP。同时接受前缀缓存被摊薄，调度要做亲和。
4. **FFN 已经是 MoE？** 专家改 EP，attention 留 TP+SP+DP。`EP = TP × DP` 时，放大 EP 等于放大 DP。
5. **all-to-all 和热点把墙钟吃掉？** EPLB + DeepEP 两种内核 + TBO；两种内核不能同居，必须 PD 分离。

叠的时候记住乘法：`#GPU = TP × PP × DP × PCP`，DCP 不乘进去。Attention 的 TP 和 MoE 的 EP 可以不是同一个数——V3 就是 attention 钉 TP4，MoE 在 prefill 走 EP32、decode 走 EP320。

开源侧能复现到什么程度？LMSYS / SGLang 在 **96×H100**（12 节点）上做了一个缩小版：prefill 仍是 **EP32**，decode 用 **EP72**（大约是论文 decode 规模的一半）。相对同一资源上的 vanilla TP16，输出吞吐最高大约 **5×**；2k 输入时，单节点大约 52.3k input tok/s、22.3k output tok/s。框架上对应的开关，可以按"开了它在解决哪一节的问题"来记：

- **SGLang**：`--enable-deepep-moe` 换上 DeepEP 的 all-to-all；`--deepep-mode {normal,low_latency}` 给两个池各绑一种内核；`--enable-eplb` 做冗余和重排；`--enable-two-batch-overlap` 打开 TBO；`--enable-dp-attention` 让 attention 按 DP 走。PD 分离本身用 `--disaggregation-mode {prefill,decode}`。
- **vLLM**：`--enable-expert-parallel` 把 MoE 从"用 TP 切专家"改成"按专家切开"，EP 规模等于 `TP × DP`（只开 PP、TP=1 且 DP=1 时，这个开关不会生效）；`--all2all-backend` 选 `deepep_high_throughput` 或 `deepep_low_latency`；`--enable-eplb` 和 `--enable-dbo` 分别对应 EPLB 与 Dual-batch overlap。长上下文再叠 CP：`--decode-context-parallel-size`（也写 `-dcp`）在现有 TP 组里交错切 KV，**不增加 GPU**；`--prefill-context-parallel-size` 才按序列加卡，world size 变成 `TP × PCP`。

有一件事两边都成立：没有 PD 分离就强行让同一通信组跑两种 DeepEP 内核，组会卡住。这不是哪个开关没打开，是通信组的语义不允许。

EP 把 FFN 按专家切开之后，还可以再走一步：**把 attention 和 MoE FFN 拆到两类机器上**。ByteDance 的 MegaScale-Infer 做的就是这件事——attention 机器盯 KV 和延迟，专家机器盯吞吐。它不是第七种切维，是模块分离：和[第四篇](/posts/llm-inference-pd-disaggregation/)的 PD 分离同一思路，只是切的对象从"两个阶段"换成"两种算子"。本篇不展开它的调度，只把它从六刀里拿出去，免得和 EP 叠在一起分不清。

## 十、四组无需 GPU 的实验

下面四组实验都在 CPU 上跑，`SEED=42` 可复现。它们**不是**端到端 GPU benchmark，也不会复现 LMSYS 那组 5×。实验 A 是一层 decoder 的解析墙钟：计算按 `1/TP` 缩放，通信按 ring all-reduce 估价。实验 B 直接代入流水线气泡公式。实验 C 是 zipf 路由下的 GPU 负载，不管字节数，只看谁最忙。实验 D 只算一层注意力附近的通信载荷：TP 激活同步 vs Ring CP / Ulysses。目的是把正文里的数量级钉死，方便对照，不代替真机上的 kernel 剖面。脚本在 `diagrams/parallelism-moe/sim_experiments.py`。

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

只改放置、不加副本，不均已经从将近 14 倍掉到 3.7 倍——说明"热专家碰巧住在一起"本身就很伤。再复制 32 个最烫的专家，straggler 降到 2.1 倍。这和 LMSYS 说的"EPLB 在大规模下把吞吐拉起来"是同一件事的负载侧：墙钟不看平均值，看最忙的那张卡。仿真没有建模 all-to-all 的字节数，也没有 kernel 启动，所以**不能**把 13.96 → 2.11 直接读成 1.49× / 2.54× 那些端到端数字。那些来自真机，见第九节。这里只说明：冗余副本解决的是"谁更忙"，不是"通信有多贵"。

### 实验 D：长序列该切 CP，而不是把 TP 再加大

hidden=8192，GQA 的 `d_kv=1024`，CP 度 `C=8`，bf16。只比较一层注意力附近的通信载荷，不算计算、也不算 hop 延迟。

```python
tp_mb = 2 * S * hidden * 2 / 1e6                 # 两次激活 all-reduce
ring_mb = (1 - 1 / C) * S * d_kv * 2 * 2 / 1e6   # 环传 K+V
ulysses_mb = 2 * S * hidden * 2 / C / 1e6        # 两次 all-to-all 换轴
```

![序列从 2k 拉到 128k，TP 激活同步涨到 4.3 GB；Ring CP（GQA）和 Ulysses 仍在数百 MB](/assets/posts/llm-inference-parallelism-moe/sim-cp-vs-tp-comm.png)

一手结果（一层通信载荷，MB / GPU）：

| S      | TP all-reduce | Ring CP | Ulysses |
| ------ | ------------- | ------- | ------- |
| 2048   | 67.1          | 7.3     | 8.4     |
| 8192   | 268.4         | 29.4    | 33.6    |
| 32768  | 1073.7        | 117.4   | 134.2   |
| 131072 | **4295.0**    | 469.8   | 536.9   |

S=2048 时，Ring 已经只有 TP 激活同步的约 **1/9**；到 128k，TP 一侧涨到 **4.3 GB**，Ring / Ulysses 还在 470–540 MB。这就是第六节那句话的数量级：长上下文该切序列，不该把 TP 再加大——GQA 把 KV 头压得很窄，环传吃的是 `d_kv`，TP 同步吃的是整个 `hidden`。仿真没有建模 ring 的 hop 延迟，也没有 all-to-all 的集体启动，所以不能直接读成墙钟。它只说明：**切错维，字节数会差一个数量级。**

## 十一、总结与延伸

并行策略的主线，还是那条"发现问题 → 引入优化 → 带出新问题"的链子：

- **一张卡装不下** → **TP（Megatron 列/行切分，含 vocab parallel）**。层内 all-reduce，体积 `(t−1)/t`，最好停在 NVLink 域。
- **TP 切完激活还在** → **SP**。只切 LayerNorm / Dropout，不切注意力；别和 Ulysses 同名混淆。
- **出节点太贵** → **PP**。改切层，气泡是 `(p−1)/(m+p−1)`；batch=1 时几乎串行。1F1B / VPP 是训练刀。
- **还要吞吐** → **DP / DP Attention**。推理不梯度同步；ZeRO / FSDP 不上推理桌。MoE 下 attention 按请求复制，专家按 EP 切。
- **序列太长，KV 比权重大** → **CP**。Ring 传 KV，或 Ulysses 换轴；推理再拆 DCP（不加人）和 PCP（加卡切 prefill）。
- **稀疏 FFN 不值得再用 TP 切** → **EP**。dispatch / combine 两次 all-to-all。
- **all-to-all 贵，路由还不均匀** → **DeepEP 两种内核 + EPLB 冗余副本**。
- **通信还是和计算同量级** → **Dual-batch overlap**；两种内核不能住在同一通信组里 → **咬合第四篇的 PD 分离**。

一句话带走：**并行就是给每一段计算选一刀——TP 切 hidden，SP 切激活，PP 切层，DP 切请求，CP 切序列，EP 切专家。切完立刻要付账：TP 省权重但不一定省 KV，SP 不切注意力，PP 省层却买气泡，DP 涨吞吐却摊薄缓存，CP 才把 KV 切开，EP 把 straggler 和 all-to-all 放到台面上。同名不是同一刀；DCP 不加卡，PCP 才加。**

延伸阅读：下一篇会把负载换成 **long-CoT / 推理模型**。思维链把 decode 拉得很长，KV 占得更久，straggler 和调度的形状都会变——第六节的 CP 就是给那种负载预备的刀，本篇已经摊开。再往后是 GPU 架构与 attention kernel。若还想往 MoE 上再砍一刀——把 attention 和 FFN 拆到两类机器上——见 MegaScale-Infer（ByteDance）：它是 EP 之后的另一次模块分离，第九节只点到为止。

## 参考

1. Shoeybi et al., [_Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism_](https://arxiv.org/abs/1909.08053), 2019.
2. Narayanan et al., [_Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM_](https://arxiv.org/abs/2104.04473), SC 2021.
3. Korthikanti et al., [_Reducing Activation Recomputation in Large Transformer Models_](https://arxiv.org/abs/2205.05198), 2022.（Megatron Sequence Parallelism）
4. Rajbhandari et al., [_ZeRO: Memory Optimizations Toward Training Trillion Parameter Models_](https://arxiv.org/abs/1910.02054), 2020.
5. Liu et al., [_Ring Attention with Blockwise Transformers for Near-Infinite Context_](https://arxiv.org/abs/2310.01889), 2023.
6. Jacobs et al., [_DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models_](https://arxiv.org/abs/2309.14509), 2023.
7. NVIDIA, [_Megatron Context Parallelism_](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html)（Ring P2P 与 a2a）.
8. vLLM, [_Context Parallel Deployment_](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)（DCP 不增加 GPU；PCP 加卡切 prefill）.
9. Liu et al., [_DeepSeek-V3 Technical Report_](https://arxiv.org/abs/2412.19437), 2024.（§3.4 推理部署：prefill EP32 / decode EP320）
10. Zhao et al., [_Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures_](https://arxiv.org/abs/2505.09343), 2025.
11. DeepSeek, [_DeepEP: an efficient expert-parallel communication library_](https://github.com/deepseek-ai/DeepEP), 2025.
12. LMSYS, [_Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs_](https://lmsys.org/blog/2025-05-05-large-scale-ep/), 2025.
13. vLLM, [_Expert Parallel Deployment_](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)（`--enable-expert-parallel`，EP = TP×DP）.
14. SGLang, [_Expert Parallelism_](https://docs.sglang.io/docs/advanced_features/expert_parallelism.html)（DeepEP / EPLB / TBO）.
15. Chen et al., [_MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism_](https://arxiv.org/abs/2504.02263), 2025.
16. DeepSeek, [_One More Thing: DeepSeek-V3/R1 Inference System Overview_](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)（开源周：decode 单元的 EP144 写法与论文 EP320 不同）.
