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
description: 推理里怎么把模型切开：用 decoder 结构图对照 TP / SP / PP / DP / CP / EP 各切哪一段，并写清每种并行对显存、通信、延迟和缓存的影响。附 MoE 的 DeepEP / EPLB。训练里同名的并行只在文末点到。
---

并行这两个字，在大模型圈子里被用滥了。有人说的 TP，有人说的 SP，还有人把 DeepSpeed 的 Ulysses 也叫序列并行。名字听着像一家，切的维、通信形态、适用场景完全不是一回事。

**本篇只讲推理。** 训练也会用这些缩写，问题却不是同一套——它要同步梯度、给反向留激活、用大 batch 填流水线。那些同名的并行收到文末点一下，这里不展开。

先把一张卡上装的东西说成人话。用户打进来几个字，模型把**每个字变成一条数字向量**。这条向量有多长，就叫 **hidden size**（记作 H）——70B 常见是 4096 或 8192。几个请求、一段话叠在一起，写成 `[B, S, H]`：

- **B**：同时进来几条请求
- **S**：这段话有多长（几个 token）
- **H**：每个字那条向量有多宽

Decoder 就是 Embed → 重复 N 层（归一化 → 注意力 → 前馈）→ LM Head。一张卡要同时装三样东西：全部权重、已经生成的 KV、当前这一层的激活。装不下、或者装得下但人太多，从最好懂的切法说起：

1. **模型已经装得下，只是请求太多** → 复制几份各接各的，这是 **DP**。
2. **整网太大，最容易想到按层切开** → 前几层一张卡、后几层另一张卡，这是 **PP**。
3. **每一层的矩阵仍然太宽** → 把每个字那条向量沿宽度切开，几张卡同时算同一层的不同段，这是 **TP**。
4. **TP 只切了「乘矩阵」**。归一化（LayerNorm / RMSNorm）几乎没有权重，却要给每个字做一次，激活仍是整段话 → 按字的位置把这段激活分开，这是 **SP**。
5. **序列更长，真正爆的是注意力里的 KV** → 把注意力看见的那段序列切开，这是 **CP**（推理里再分成 DCP / PCP）。
6. **前馈变成很多小专家** → 按专家切开，这是 **EP**。

前两步（复制、按层）不需要新名词。卡在第 3、4 步的，通常是 H 和 LayerNorm 没有对上图。下面这张把这两件事画在一起：上面是「一个词 = 一条向量，TP 切宽度」；下面是「归一化看着整段话，SP 只切这一段激活，不切注意力」。

![每个词是一条长度为 H 的向量，TP 沿这条宽度切开；LayerNorm 几乎无权重却占整段激活，SP 按词的位置切开这段激活，注意力仍要看完整序列](/assets/posts/llm-inference-parallelism-moe/diagram-hidden-layernorm.png)

这是本系列第六篇。[第一篇](/posts/from-causal-lm-to-inference-system/)把"权重怎么切到多卡"点过题；[第四篇](/posts/llm-inference-pd-disaggregation/)说了 prefill 想要小 TP、decode 想要大 TP，但没解释这些缩写在切什么；[第五篇](/posts/llm-inference-scheduling-distributed/)把请求派到了某台机器。这一篇钻进那台机器所属的并行组，问三件事：**按哪一维切？切完显存、通信、延迟变成什么样？哪些并行其实不是一回事？**

> **说明**：本篇讲的是**推理时模型怎么切开**，不是第五篇那种跨实例的请求路由。下面按「先复制、再按层、再切层内宽度」往下走，越往后越细。主线还是：**每暴露一个问题，就引入一种优化，又带出新问题。**

## Table of contents

## 一、先把各种并行摊开

推理里真正会叠在一起的，是下面这六种。表的顺序和开篇一样：先复制，再按层，再切层内。

![六种并行全景：从复制、按层，到切宽度、切序列、切专家](/assets/posts/llm-inference-parallelism-moe/diagram-taxonomy.png)

| 并行   | 切开的维                   | 典型通信                          | 推理里干什么                         | 最怕什么                       |
| ------ | -------------------------- | --------------------------------- | ------------------------------------ | ------------------------------ |
| **DP** | batch / 请求               | 推理里通常**没有**                | 复制接单；MoE 上变成 DP Attention    | 前缀缓存被摊薄                 |
| **PP** | layers                     | 阶段边界 **点对点**               | 跨节点时的备选；decode 很吃气泡      | 气泡；batch=1 灌不满           |
| **TP** | 每个词那条向量的宽度 H     | 层内 **all-reduce**               | 把同一层的权重摊到多张卡             | 跨节点；小消息延迟             |
| **SP** | 归一化看到的那段序列       | all-gather / reduce-scatter       | 寄生在 TP 上，再削一档激活           | 被叫成"序列并行"的其实是 CP    |
| **CP** | 序列，**含注意力本身**     | Ring 传 KV，或 Ulysses all-to-all | 切开 KV；再拆 DCP / PCP              | 和 SP 同名；DCP / PCP 还要再分 |
| **EP** | MoE experts                | dispatch / combine **all-to-all** | 切开稀疏 FFN                         | 热点专家；两种通信内核不能共存 |

读这张表时，盯"切开的维"比盯缩写有用。DP 切的是请求的份数，PP 切的是网络的深，TP 切的是每个词那条向量的宽，SP 切的是**不算注意力的那截激活**，CP 切的才是**注意力看见的那段序列**，EP 切的是专家。**通信的形状，比"用了几张卡"更能决定这种并行能不能赚钱。**

整网还没切的时候，长这样：Embed → N 层（RMSNorm → Attention → FFN）→ LM Head。

![未切开时一张卡要装下全部权重、KV 和激活；B / S / H / 层号 / 专家是五条可切的缝](/assets/posts/llm-inference-parallelism-moe/diagram-model-backbone.png)

同一份结构，六种并行切完是六种样子。颜色表示 GPU：哪一块换了颜色，就是这种并行切到的结构。

![DP 复制整网；PP 把层栈横切给不同卡；TP 竖切每层矩阵；SP 只切 Norm 的序列；CP 切 Attention 的 token；EP 只拆 FFN 里的专家](/assets/posts/llm-inference-parallelism-moe/diagram-six-slices.png)

再把镜头推进**一层**。PP 决定这一层住在哪张卡，DP 决定这张卡接哪几条请求；层内的并行落在不同算子上。SP 到 Attention 门口就停，CP 才走进注意力；Dense 的 FFN 用 TP 切宽度，换成 MoE 就改切专家。

![一层 decoder 的数据流：DP 切 B，SP 切 Norm 的 S，TP 切线性层的 H，CP 切 Attention 的 S，EP 切 MoE 专家](/assets/posts/llm-inference-parallelism-moe/diagram-decoder-ops.png)

切开不是免费的。下面这张表和后面各节的「影响与约束」说的是同一件事：省了哪一档显存，立刻多出哪一笔通信，以及什么条件下这种并行会反过来变慢。

| 并行   | 每卡权重                     | 每卡 KV                        | 同步节奏                      | 硬约束                                   | 推理上最明显的副作用                         |
| ------ | ---------------------------- | ------------------------------ | ----------------------------- | ---------------------------------------- | -------------------------------------------- |
| **DP** | × 副本（模型仍要装得下一份） | × 副本                         | 推理里通常没有                | 装不下的模型，加副本也装不下             | 前缀缓存被摊薄                               |
| **PP** | `/ p`（只持有若干层）        | 只留本地层，约 `/ p`           | 阶段边界点对点                | 层数最好能整除；`m≈1` 时利用率 `1/p`     | TTFT 被灌流水线拉长；decode 大量空转         |
| **TP** | `/ t`                        | 常不降；GQA 头不够时还**复制** | 每层 **2× all-reduce**        | `t` 整除 H / 头数；最好停在 NVLink       | decode 小消息被延迟打穿                      |
| **SP** | 同 TP                        | **不降**                       | all-gather / RS 换掉一部分 AR | 必须已经开 TP                            | decode 激活本来就小，显存收益远小于 prefill |
| **CP** | 基本不变                     | `/ C`                          | 每次注意力都通信              | Ulysses：头数 ≥ C；DCP ≤ `tp / KV头`     | decode 每步都要扫别人的 KV；PCP 才加卡       |
| **EP** | 专家权重 `/ ep`              | 跟 attention 侧走              | 每层 2× all-to-all            | 专家数应能整除；两种 DeepEP 内核不能同居 | 热点专家；必须 PD 分离才能绑对内核           |

## 二、模型已经装得下，只是人太多：数据并行 DP，以及 MoE 入口的 DP Attention

最好懂的一种并行，其实不是切模型，而是**复制**。一份模型已经能塞进一张卡（或已经按后面的办法切开、能塞进一个并行组），但请求排队还是太长。那就同样的模型多放几份，各接各的单。这就是 **数据并行 DP**。

**推理里没有梯度**，副本之间默认不说话，所以它看起来最便宜。便宜的代价在别处：每份副本各自缓存各自的 KV。你加的副本越多，同一个前缀就越容易被摊到不同机器上，[第二篇](/posts/kv-cache-paged-attention-and-prefix-caching/)的前缀缓存、[第五篇](/posts/llm-inference-scheduling-distributed/)的缓存感知路由都会被稀释。DP 在 serving 里的正确用法，常常不是再复制一份完整的 70B，而是和别的并行叠在一起，只复制那些**必须按请求切开**的部分。训练里的 ZeRO / FSDP 不要算进这种并行，第十节点一下。

MoE 把这件事逼出了一个专门变体：**DP Attention**。

原因在于同一层里其实有两种计算。MoE 的 FFN 是稀疏的，256 个专家里每个 token 只走 top-k 个，专家可以按 EP 切开、各住各的卡。**Attention 不是这样。** MLA / GQA 的 KV 是按这条请求、按已经生成的序列长出来的，做注意力时必须看到**自己那批请求的完整 KV**。你不能把一条请求的 KV 拆去另一张卡上的"注意力专家"——注意力这边没有专家可切。

于是同一层里叠了两套并行：

- **Attention**：按 batch 切开（DP）。每张卡算自己分到的那几条请求，KV 留在本地，不跨卡。
- **MoE**：按 expert 切开（EP）。token 算完注意力之后，按路由飞到专家所在的卡，算完再飞回来。

DeepSeek-V3 的 prefill 写成 **TP4 + SP + DP8**，decode 写成 **TP4 + SP + DP80**，指的都是 attention 这一侧。为什么 EP 变大时 DP 也跟着变大？在常见的实现里，专家并行的规模满足 `EP = TP × DP`（vLLM 的 `--enable-expert-parallel` 就是这么算的）。多出来的卡主要用来**多住专家**；attention 不能按专家切，就按 DP 复制一份，让每张卡继续看着自己那几条请求的 KV。

**影响与约束。** 纯 DP 的吞吐近似随副本线性涨，推理里副本之间也不梯度同步，看起来最香。但加副本**不会**让一个装不下的模型突然装得下——每份仍然要放下自己那份权重（或 TP 切片）和自己那份 KV。副作用在缓存：同一条系统提示被摊到不同副本，[第二篇](/posts/kv-cache-paged-attention-and-prefix-caching/)的前缀命中、[第五篇](/posts/llm-inference-scheduling-distributed/)的亲和路由都会被稀释。DP 开得越大，调度越难把同一条前缀送到同一份副本。

DP Attention 还改写了"DP 不说话"这条。token 从"按请求切"的 attention 走进"按专家切"的 MoE 时，必须做一次集体通信，把各卡上的 token 交给持有对应专家的卡。所以 MoE 上的 DP 不是免费的复制，是 EP 的入口税。

## 三、整网太大，先按层切开：流水线并行 PP

复制解决不了「一份模型就装不进一张卡」。最容易想到的切法是按深度切开：连续若干层交给一个**阶段（stage）**，激活算完这一段，再递给下一个阶段。这就是 **流水线并行 PP**。后面会讲的 TP 最好停在 NVLink 域里；出了节点再把一层切碎，通信往往比计算还贵。那时代价更低的办法，就是按层往下传。

![p=4 个阶段、m=8 个微批次的流水线；斜线格是气泡。气泡比例 (p−1)/(m+p−1)；batch=1 时利用率只剩 1/p](/assets/posts/llm-inference-parallelism-moe/diagram-pp-bubble.png)

通信形态变便宜了：不再是每层做一次全集同步，只在阶段边界**点对点**传一次激活。新问题换成了 **气泡（bubble）**。流水线要先灌满才会转起来：第 0 个阶段开始算第 1 个微批次时，后面的阶段只能空手等；最后一个微批次离开第 0 个阶段之后，前面的阶段又要空转到收尾。图里的斜线格，就是这两头空出来的时间。

Narayanan 等人给出的理想利用率是 `m / (m + p − 1)`。`p` 是阶段数，`m` 是微批次（micro-batch）数。气泡占比则是 `(p − 1) / (m + p − 1)`。`p=8`、`m=1` 时，利用率只剩 **12.5%**——八张卡里几乎只有一张在干活，其余在等。**在线 decode 常常一步只推一个 token**，没有那么多微批次可切。就算硬切，也会把单条请求的延迟拉长：用户要等整条流水线灌完，才看到这个 token。训练里用来填气泡的招，收到第十节。

**影响与约束。** PP 让每张卡只持有若干层，**权重和 KV 都按阶段变薄**——这是它相对 TP 少被提到的优点：KV 只为本地那些层而存在。通信也便宜，只在阶段边界点对点传一份 `[B, S, H]` 的激活，不必每层全集同步。

账单在利用率上。公式 `m / (m + p − 1)` 不留情：在线 decode 常常 `m≈1`，八张卡里同时只有一张在算这一步，其余在等。用户侧的感觉是 **TTFT 变长**——第一个 token 要等流水线灌到最后一阶段。你若为了填气泡去切微批次，单条请求的延迟会被切得更碎、更长。层数最好能被 `p` 整除，否则短的阶段在等长的阶段，气泡比公式还大。

所以生产推理里，PP 是"模型跨不出 NVLink、又还不是 MoE"时的备选，不是 decode 的主方案。DeepSeek-V3 的推理单元**不用 PP**，把跨节点预算留给 EP：那种通信更散，但至少每张卡都有专家在算。

## 四、一层仍然太宽：张量并行 TP

按层切开之后，**每一层自己**仍可能太大。一张 H100 是 80 GB，70B 的 bf16 权重就要 140 GB，还没算 KV。问题出在每个词那条向量太宽——就是开篇图里的 H。线性层是这条向量乘一个大矩阵，矩阵的一边等于 H。Megatron-LM 的做法是：不要把整层交给一张卡，把这个宽度切开，让几张卡**同时算同一层的不同段**。这就是 **张量并行 TP**。

一个线性层写成 `Y = X W`。切开的方式有两种，而且在 Transformer 里几乎总是**成对出现**：先 Column Parallel，再 Row Parallel。

![列并行按输出维切开、无需通信；行并行按输入维切开，之后一次 all-reduce，体积正比于 (t−1)/t。节点内走 NVLink，跨节点走 IB 会把加速吃掉](/assets/posts/llm-inference-parallelism-moe/diagram-tp-allreduce.png)

1. **列并行（Column Parallel）**：按**列**切 `W`，也就是按输出维切开。每张卡都拿到完整的 `X`，各自算出 `Y` 的一块。这几块在逻辑上拼起来就是完整输出，**这一步没有通信**。Attention 的 QKV、MLP 的第一段（升维）走这条。
2. **行并行（Row Parallel）**：按**行**切 `W`，也就是按输入维切开。上一层刚切出来的那一块 `X`，正好对得上这一层的一块 `W`。每张卡算出一份**部分和**，再做一次 **all-reduce**，才能得到完整的 `Y`。Attention 的 output projection、MLP 的第二段（降维）走这条。

之所以要成对，是为了少通信。升维用列并行，中间那截激活可以就地留给下一层；降维用行并行，只在层的出口同步一次。一层 decoder 通常就是这样两次 all-reduce：一次在 attention 出口，一次在 MLP 出口。

Ring all-reduce 的数据量正比于 **(t−1)/t**。这个式子有个不太直观的后果：TP 从 1 到 2，通信从 0 一下子跳到"全量的一半"；再往上，增量变缓，但每一次都要等所有卡对齐。墙钟既取决于最慢的那张卡，也取决于这次同步走的是哪条线——同一机箱里的 NVLink，还是机箱外面的 InfiniBand。

所以 TP 有一条硬约束：**尽量停在 NVLink 域里**。常见的是单机 8 卡，或者 NVLink 连起来的超节点。H100 节点内 NVLink 是数百 GB/s 这个量级；一出节点，InfiniBand 掉到数十 GB/s，延迟也高一个数量级。同样算一层，NVLink 上加大 TP 往往还在加速，IB 上通信很快反超计算。这也解释了 DeepSeek-V3 为什么把 attention 的 TP **钉死在 4**——论文原话就是用小 TP 限制通信开销。再大，切出来的计算更碎，同步却更密。

词表和 LM Head 通常也跟着 TP 切，叫 **vocab parallel**：embedding 按词表维切开，最后的 softmax 只在各卡的那一段词表上做，再做一个并行的采样。它不是独立的第七种并行，是 TP 在模型两头的延伸。

**影响与约束。** 权重按 `1/t` 变薄，这是 TP 能把 70B 塞进多张 80GB 卡的原因。但有三笔账立刻上门。

第一笔是**通信**。一层两次 all-reduce，体积正比于 `(t−1)/t`。Prefill 序列长、算得动，NVLink 上加大 TP 往往还在加速；decode 每步只推一个 token，消息小、次数密，延迟项比带宽项更刺。一出节点再把 TP 开大，很容易让通信反超计算——不是算力不够，是报数报不过来。

第二笔是 **KV**。很多人以为"切了权重，KV 也会变薄"。GQA / MLA 的 KV 头本来就少，`t` 大于 `num_kv_heads` 时，实现会把同一份 KV **复制**到多张卡上。权重省了，KV 反而更肥。这正是第六节 DCP 要砍掉的那截复制。

第三笔是**形状**。H 和注意力头必须能被 `t` 整除；切得太碎，GEMM 也不饱，通信占比更高。CUDA Graph 还要求这次和上次的 collective 形状一样，动态 batch 会把图录不进去。

硬约束可以记三条：尽量停在 NVLink 域；`t` 要整除头数；decode 不要为了显存把 TP 开到跨节点。DeepSeek-V3 把 attention 钉在 TP4，就是这三条叠在一起的结果。

## 五、线性层切完，归一化还占着序列：序列并行 SP

看回开篇那张图的下半。TP 切的是「乘矩阵」。**LayerNorm / RMSNorm**（以及 Dropout）几乎没有权重：它只是给每个词的那条向量做一次归一化。没权重可切，激活却还是整段话 × 整条向量。Prefill 序列一长，这一档就看得见。

**Megatron 的序列并行 SP（Sequence Parallel）** 做法是：把这些算子按序列维切开，每张卡只留自己那一段 token 的激活，用 all-gather / reduce-scatter 替换掉一部分 all-reduce。换来的是激活显存再薄一档。它**要求已经开了 TP**，而且**不切注意力计算本身**——Q 还是要和整段 KV 见面。DeepSeek-V3 的 attention 写成 **TP4 + SP**，指的就是这套组合。

这里必须把三个常被混用的名字拆开：

| 名字                             | 切什么                          | 注意力怎么算                            |
| -------------------------------- | ------------------------------- | --------------------------------------- |
| **Megatron SP**                  | 只切 LayerNorm / Dropout 的激活 | 注意力仍在 TP 组里，不按序列摊 KV       |
| **Ring CP**                      | 切完整序列，含 QKV              | 环形把别人的 KV 传过来                  |
| **Ulysses（DeepSpeed 也叫 SP）** | 切完整序列                      | 两次 all-to-all，改成按头切开再算注意力 |

后两个才是第六节的 **CP**。如果有人说"我们开了序列并行"，先问一句：切的是 LayerNorm，还是注意力。

**影响与约束。** SP 省的是 LayerNorm / Dropout 那截**激活**显存。推理 decode 每步只有一个新 token，激活本来就小，显存收益远小于 prefill——DeepSeek-V3 仍写成 TP4+SP，主要是和 Megatron 的实现绑在一起，顺手把激活路径也切开，不是因为 decode 激活爆了。

硬约束有两条，都容易踩错。第一，**必须已经开 TP**，SP 不能单独存在。第二，注意力前要把切开的序列 **all-gather** 拼回来，Q 仍然看见整段 KV，所以 **KV 并不变少**。把它当成"长上下文方案"会用错：长上下文该找 CP。

## 六、序列太长：上下文并行 CP

前面切的是请求、层、每个词的宽度。序列一旦拉到 32k、128k，新问题换了对象：**KV Cache 比权重大**。一张卡也许还装得下 70B 的切片，却装不下这条请求已经生成的 KV。TP 帮不上这个忙——GQA / MLA 的 KV 头本来就少，TP 加大以后，每张卡反而要把同一份 KV 再复制一遍。第五节的 SP 也帮不上：它只切归一化，不切注意力。

这种并行叫 **上下文并行 CP（Context Parallel）**。把序列按 token 切开，每张卡只负责一段；注意力要用到别人那段 KV 时，再把 KV 转过去。切的是**注意力看见的那段序列**，所以和 Megatron SP 不是一回事。DeepSpeed 把 Ulysses 也叫 Sequence Parallelism，名字撞了，维没撞。

![Ring 环形传 KV；Ulysses 两次 all-to-all 换成按头计算；DCP 不增加 GPU，只在 TP 组里交错切 KV](/assets/posts/llm-inference-parallelism-moe/diagram-context-parallel.png)

通信有两条主流路。

**Ring CP / Ring Attention。** Megatron 默认走这条。每张卡先拿自己那段 QKV，先和本地 KV 做注意力；再把 KV 沿环传给邻居，和下一块再做一次；转满一圈，就等价于看见了完整序列。一层的通信体积大约是 `(1 − 1/C) · S · d_kv · 2`（K 和 V），bf16 再乘 2 字节。GQA / MLA 让 `d_kv` 远小于 `hidden`，环传比把整段激活 all-reduce 便宜一个数量级。计算和通信还能重叠：本块注意力在算时，下一块 KV 已经在路上。

**Ulysses（DeepSpeed；Megatron 里也叫 a2a）。** 两次 all-to-all 换轴：先按序列切开拿到 QKV，再换成按注意力头切开——每张卡拿到**完整序列、但只有一部分头**，注意力在本地做完，再 all-to-all 换回去。体积大约是 `2 · S · hidden / C`。一次集体通信，NVLink 上往往比一圈 P2P 更快；约束是头数得 ≥ CP 度。跨节点则常退回 Ring。

序列一长，该切 CP，而不是把 TP 再加大：Ring 吃的是 `d_kv`，TP 同步吃的是整个 `hidden`，GQA / MLA 下两者能差一个数量级。

推理还要再拆一次，因为 prefill 和 decode 的 KV 形状完全不同。

**Decode Context Parallel（DCP）。** 不增加 GPU 数。它复用现有 TP 组，按 token 交错把 KV 切开：`token i` 住在 `i % dcp` 那张卡上。GQA 头不够切时，TP 会把 KV 复制 `tp / H` 份；DCP 就是来砍掉这段复制的。vLLM 的开关是 `--decode-context-parallel-size`（也写 `-dcp`），上界大约是 `tp_size / num_kv_heads`。开得越大，KV 越省，通信越重。

**Prefill Context Parallel（PCP）。** 才真正加卡。一条超长 prompt 按序列切开，用来压 TTFT。world size 变成 `TP × PCP`。vLLM 对应 `--prefill-context-parallel-size`。它和 DCP 正交，不要共用一个开关：一个改的是"decode 时 KV 怎么摊在已有卡上"，一个改的是"prefill 要不要多叫几张卡来切序列"。

**影响与约束。** CP 打的是 KV：按 `C` 切开之后，每卡大约只留 `1/C` 的缓存，32k、128k 才装得下。权重几乎没变薄，所以它**补的是 TP 做不到的那一维**，不是 TP 的替代品。切错维，通信载荷会差一个数量级。

通信是每步都要付的。Ring 能和计算重叠，但 hop 随 `C` 涨；Ulysses 一次 all-to-all 在 NVLink 上更快，**头数必须 ≥ CP 度**，跨节点常退回 Ring。Decode 更苛刻：每步只有一个新 Q，却要扫很长的 KV。DCP 复用 TP 组、不加卡，上界大约 `tp / num_kv_heads`，开太大，省下的 HBM 会被通信吃回去。PCP 加卡压 TTFT，会乘进 world size。因果掩码没有被切掉：你分得再碎，逻辑上还是要看完整前缀。

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

## 八、all-to-all 贵 + 热点专家 —— DeepEP、EPLB 与 Dual-batch overlap

EP 把两个新问题同时放到台面上。它们不是先后发生的，而是同一套 all-to-all 上的两面：一面是**谁更忙**，一面是**通信本身怎么走**。

**问题一：热点专家。** 真实流量的路由几乎从来不是均匀的，更接近 zipf：少数专家吃掉大部分 token。你就算把 256 个专家整齐地每卡放 8 个，最热门的那几个如果碰巧住在同一张卡上，这张卡就会变成整层的 straggler。DeepSeek-V3 的对策是两步一起做。

第一步，先承认热点存在，给最热门的专家做 **冗余副本（redundant experts）**。统计一段时间里谁被选得最多，把这些专家**再复制一份**放到相对空闲的卡上；新来的 token 发给当前更轻的那份副本。第二步，用 **EPLB（Expert Parallelism Load Balancer）** 在节点内重排专家，尽量让热的和冷的搭在一起，并且不要为此增加跨节点 all-to-all——跨节点比节点内贵得多。

数字可以对照着看。Prefill 部署了 **32 个冗余专家**，EP32 上每卡从 8 个 routed 变成 **8+1**。Decode 更进一步，把共享专家也当成一个"永远被选中的热专家"，64 张卡专门托管冗余和共享。LMSYS 后来在 96×H100 上做消融，EPLB 带来大约 **1.49× prefill / 2.54× decode** 的吞吐——decode 受益更大，因为 EP 更大、不均被放大得更厉害。

![zipf 热点让单卡过载；EPLB 把最热门的专家复制到闲卡，墙钟跟最忙的卡走](/assets/posts/llm-inference-parallelism-moe/diagram-eplb.png)

**问题二：通信形态随阶段而变。** 就算专家放匀了，all-to-all 还在。Prefill 一次 dispatch 往往带着很长的一段序列，消息大，瓶颈在**带宽**；decode 每步只有一个新 token，消息小，瓶颈在**延迟**——启动一次通信、握手、绕过 CPU，本身就可能比有效载荷更贵。DeepEP 为此做了两种内核，而不是用一个内核应付两种流量：

- **高吞吐（normal）**：把管道打满，适合 prefill。代价是输出形状是 symbolic 的，和 CUDA Graph 合不来——CUDA Graph 要的是"这次和上次形状一样，才能把启动开销录下来重放"。
- **低延迟（low-latency）**：走 IB 点对点，再用 IBGDA（经由 NVSHMEM）把控制面也放到 GPU 上，适合 decode。形状稳定，能进 CUDA Graph。

关键限制写在 DeepEP / SGLang 的文档里，也很硬：**同一通信组不能同时跑两种内核。** 这就是[第四篇](/posts/llm-inference-pd-disaggregation/)的 PD 分离，在 MoE 上从"可选优化"变成**刚需**的原因。DeepEP 有一个 auto 模式，理论上能按负载切换；可一旦 prefill 和 decode 还挤在同一个引擎、同一组通信里，它选不了"prefill 走 normal、decode 走 low-latency"。拆成两个池之后，每个池绑死一种内核，这件事才做得到。

就算内核选对了，all-to-all 的时间往往仍和 grouped GEMM 同量级，藏不住。下一步是 **Dual-batch overlap / TBO（Two-Batch Overlap）**：把一个 batch 切成两个微批次，**一批在做 grouped GEMM 时，另一批正好在做 dispatch / combine**。墙钟从 `计算 + 通信` 变成大约 `max(计算, 通信)`。LMSYS 测到，prefill 在同等 token 数下 TBO 能再加 **27–35%** 吞吐；因为它把峰值激活也摊成了两半，单卡能吞的 token 上限还从 8k 抬到了 16k。decode 上 TBO 要更挑：batch 太小（例如每卡 32 token）时，切成两半会让 kernel 更不饱，甚至出现负收益。

![DeepEP 两种内核不能共存，必须 PD 分离；两个微批次交错后墙钟取 max(计算, 通信)](/assets/posts/llm-inference-parallelism-moe/diagram-deepep-overlap.png)

## 九、生产里怎么配：DeepSeek-V3 与开源框架

把前面的并行叠回去，就是 DeepSeek-V3 论文 §3.4 里的推理单元。最值得盯住的不是某一行的数字，而是：**prefill 和 decode 的并行度故意不一样。** 这正是第四篇说的"阶段专属优化"，落到 MoE 上的具体配法。

![Prefill：4 节点 / 32 GPU，attention TP4+SP+DP8，MoE EP32；Decode：40 节点 / 320 GPU，attention TP4+SP+DP80，MoE EP320](/assets/posts/llm-inference-parallelism-moe/diagram-pd-ep-scale.png)

|           | Prefill                            | Decode                        |
| --------- | ---------------------------------- | ----------------------------- |
| 最小单元  | 4 节点 / **32 GPU**                | 40 节点 / **320 GPU**         |
| Attention | **TP4 + SP + DP8**                 | **TP4 + SP + DP80**           |
| MoE       | **EP32**（每卡 8 routed + 1 冗余） | **EP320**（每卡 1 个 routed） |
| 冗余      | 32 个冗余专家                      | 64 张卡托管冗余 + 共享        |
| 通信      | 高吞吐内核 + 双微批次              | IB P2P + IBGDA                |

可以按行读一遍。Attention 两侧都钉死 **TP4 + SP**：节点内同步还便宜，再大就不划算。两边真正拉开的是 DP 和 EP。Prefill 算力密集，专家不必摊得太碎，**EP32** 就能喂饱计算，每卡还留得下 8 个 routed 加 1 个冗余。Decode 访存密集，权重和 KV 都在抢带宽，于是把专家摊到 **EP320**，每卡只住 1 个 routed——单卡上的权重更少，就能进更大的 batch，聚合起来的 HBM 带宽才够把 decode 喂饱。通信内核也跟着阶段走：prefill 打带宽，decode 走 IB 点对点和 IBGDA 压延迟。

数字本身不是教条。后来 DeepSeek 开源周的系统概述里，decode 单元收成过 **EP144 / 18 节点** 的写法，和论文的 EP320 不是同一版部署。硬件换一代、流量结构一变，具体 EP 会改。不变的是原则：**两阶段不要共用一套并行度。**

怎么选，其实就是开篇那条从简单到复杂的链子，加上约束之后可以写成检查清单：

1. **模型已经装得下、只是排队太长？** 先 DP。接受前缀缓存被摊薄，调度要做亲和。
2. **整网太大？** 按层切 PP。跨节点做 TP 往往更贵；PP 要接受 decode 的气泡。
3. **一层仍然太宽？** TP 切每个词那条向量的宽度，SP 切 LayerNorm 的序列。TP 停在 NVLink；头数整除不了就不要硬开。
4. **KV 比权重大、序列很长？** 不要再加大 TP——GQA 会复制 KV。改 CP；decode 用 DCP 砍复制，prefill 长 prompt 才用 PCP 加卡。
5. **FFN 已经是 MoE？** 专家改 EP，attention 留 TP+SP+DP。`EP = TP × DP` 时，放大 EP 等于放大 DP。
6. **all-to-all 和热点把墙钟吃掉？** EPLB + DeepEP 两种内核 + TBO；两种内核不能同居，必须 PD 分离。

叠的时候记住乘法：`#GPU = TP × PP × DP × PCP`，DCP 不乘进去。Attention 的 TP 和 MoE 的 EP 可以不是同一个数——V3 就是 attention 钉 TP4，MoE 在 prefill 走 EP32、decode 走 EP320。

开源侧能复现到什么程度？LMSYS / SGLang 在 **96×H100**（12 节点）上做了一个缩小版：prefill 仍是 **EP32**，decode 用 **EP72**（大约是论文 decode 规模的一半）。相对同一资源上的 vanilla TP16，输出吞吐最高大约 **5×**；2k 输入时，单节点大约 52.3k input tok/s、22.3k output tok/s。框架上对应的开关，可以按"开了它在解决哪一节的问题"来记：

- **SGLang**：`--enable-deepep-moe` 换上 DeepEP 的 all-to-all；`--deepep-mode {normal,low_latency}` 给两个池各绑一种内核；`--enable-eplb` 做冗余和重排；`--enable-two-batch-overlap` 打开 TBO；`--enable-dp-attention` 让 attention 按 DP 走。PD 分离本身用 `--disaggregation-mode {prefill,decode}`。
- **vLLM**：`--enable-expert-parallel` 把 MoE 从"用 TP 切专家"改成"按专家切开"，EP 规模等于 `TP × DP`（只开 PP、TP=1 且 DP=1 时，这个开关不会生效）；`--all2all-backend` 选 `deepep_high_throughput` 或 `deepep_low_latency`；`--enable-eplb` 和 `--enable-dbo` 分别对应 EPLB 与 Dual-batch overlap。长上下文再叠 CP：`--decode-context-parallel-size`（也写 `-dcp`）在现有 TP 组里交错切 KV，**不增加 GPU**；`--prefill-context-parallel-size` 才按序列加卡，world size 变成 `TP × PCP`。

有一件事两边都成立：没有 PD 分离就强行让同一通信组跑两种 DeepEP 内核，组会卡住。这不是哪个开关没打开，是通信组的语义不允许。

EP 把 FFN 按专家切开之后，还可以再走一步：**把 attention 和 MoE FFN 拆到两类机器上**。ByteDance 的 MegaScale-Infer 做的就是这件事——attention 机器盯 KV 和延迟，专家机器盯吞吐。它不是第七种切维，是模块分离：和[第四篇](/posts/llm-inference-pd-disaggregation/)的 PD 分离同一思路，只是切的对象从"两个阶段"换成"两种算子"。本篇不展开它的调度，只把它从这六种并行里拿出去，免得和 EP 叠在一起分不清。

## 十、训练里那些同名的并行（不展开）

训练也会切每个词那条向量的宽度、切层、切 batch，缩写经常和推理一样。问题却不是同一套：它要同步梯度、给反向留激活、用大 batch 填流水线。下面只列几种最容易混进来的名字，机制不展开。

- **ZeRO / FSDP**：按 DP 维切优化器状态、梯度，ZeRO-3 / FSDP 有时连参数也切开。推理没有优化器，也没有梯度；参数要切，走前面的 TP、PP 或 EP。
- **1F1B 与虚拟流水线 VPP**：用大 batch 和反向来填 PP 气泡。在线 decode 常常 `m≈1`、没有反向，这两招帮不上忙。所以第三节把 PP 写成推理里的备选，不是 decode 的主方案。
- **DualPipe**：训练里把前向和反向叠在同一条流水线上。第八节的 Dual-batch overlap 叠的是两个微批次的 grouped GEMM 和 all-to-all，名字像，不是同一件事。
- **训练 DP**：每步 all-reduce 梯度，副本之间同步很重。推理 DP 默认不通信，贵的是前缀缓存被摊薄。

Megatron SP 最初也是为训练激活发明的——激活要留给反向。推理 decode 激活本来就小；V3 写成 TP4+SP，是实现绑在一起，不是 decode 激活爆了。

这些都不改变前面的选择顺序。推理认的还是那六种并行，外加 DeepEP / EPLB / TBO。

## 十一、总结与延伸

推理并行的主线，还是那条从简单到复杂、"发现问题 → 引入优化 → 带出新问题"的链子：

- **模型已经装得下，人太多** → **DP / DP Attention**。推理不梯度同步。MoE 下 attention 按请求复制，专家按 EP 切。
- **整网太大** → **PP**。按层切开，气泡是 `(p−1)/(m+p−1)`；decode 常常 batch=1，几乎串行。
- **一层仍然太宽** → **TP（Megatron 列/行切分，含 vocab parallel）**。切的是每个词那条向量的宽度 H。层内 all-reduce，体积 `(t−1)/t`，最好停在 NVLink 域。
- **线性层切完，归一化还占着序列** → **SP**。只切 LayerNorm / Dropout，不切注意力；别和 Ulysses 同名混淆。
- **序列太长，KV 比权重大** → **CP**。Ring 传 KV，或 Ulysses 换轴；再拆 DCP（不加人）和 PCP（加卡切 prefill）。
- **稀疏 FFN 不值得再用 TP 切** → **EP**。dispatch / combine 两次 all-to-all。
- **all-to-all 贵，路由还不均匀** → **DeepEP 两种内核 + EPLB 冗余副本**。
- **通信还是和计算同量级** → **Dual-batch overlap**；两种内核不能住在同一通信组里 → **咬合第四篇的 PD 分离**。

一句话带走：**推理并行就是给每一段计算选一种切法——DP 切请求，PP 切层，TP 切每个词那条向量的宽度，SP 切归一化看到的那段序列，CP 切注意力看见的序列，EP 切专家。切完立刻要付账：DP 涨吞吐却摊薄缓存，PP 省层却买气泡，TP 省权重但不一定省 KV，SP 不切注意力，CP 才把 KV 切开，EP 把 straggler 和 all-to-all 放到台面上。同名不是同一种切法；DCP 不加卡，PCP 才加。训练里那些同名的并行，第十节点过就够。**

延伸阅读：下一篇会把负载换成 **long-CoT / 推理模型**。思维链把 decode 拉得很长，KV 占得更久，straggler 和调度的形状都会变——第六节的 CP 就是给那种负载预备的切法，本篇已经摊开。再往后是 GPU 架构与 attention kernel。若还想往 MoE 上再拆一步——把 attention 和 FFN 拆到两类机器上——见 MegaScale-Infer（ByteDance）：它是 EP 之后的另一次模块分离，第九节只点到为止。

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
