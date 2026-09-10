---
author: xiangel
pubDatetime: 2026-09-10T02:30:00Z
title: "30 GB/s 这条线：PD 分离什么时候比单机快，什么时候更慢"
slug: llm-inference-pd-disaggregation-gw
featured: false
draft: false
tags:
  - 大模型推理系统
  - PD分离
  - LLM
description: PD 分离不是"把两个阶段拆开"这么简单，它是拿一笔看得见的 KV 传输成本，去换单机里那笔看不见的 prefill–decode 干扰成本。这笔账只有当互连快过约 30 GB/s 时才为正，否则你会输给一台调好的单机。用"外包与运费"做类比，附一组无需 GPU、可复现的仿真，把那条盈亏平衡线画出来。
---

> 本文是用 great-writer 写作技能重写的第四篇「实验版」，和[原版《大模型推理的 PD 分离》](/posts/llm-inference-pd-disaggregation/)讲同一个主题、同一组实验数据，换一种写法，供对照。

你读完 DistServe，又读完 Splitwise，两篇都在说同一件事：把 prefill 和 decode 拆到两个 GPU 池，吞吐能翻倍。你信了，在集群上铺开 PD 分离，压测跑完——**比你换掉的那台单机还慢。**

哪儿错了？

一步没错。论文没骗你，翻倍是真的。只是它们默认了一个你没注意的前提：**机器之间有 RDMA 或 NVLink。** 而你那几台卡，是普通网络连的。

这就是 PD 分离最容易被忽略的一面：它不是一个"开了就快"的开关。它是一笔交易——而交易有可能赔本。这一篇，我只想把这笔账算清楚，把那条**赚与赔的分界线**画出来。

这是本系列第四篇。前三篇分别讲了[推理系统怎么长出来](/posts/from-causal-lm-to-inference-system/)、[KV Cache 怎么管](/posts/kv-cache-paged-attention-and-prefix-caching/)、[单机内部怎么调度](/posts/llm-inference-scheduling-single-node/)。这一篇往上走一步：**一张卡装不下、也伺候不好所有请求时，把 prefill 和 decode 彻底拆到两个池——这就是 PD 分离（Prefill/Decode Disaggregation）。**

## Table of contents

## 拆之前，先看清你到底在为什么买单

单机上，prefill 和 decode 挤在同一张卡。上一篇讲过的 `chunked prefill` 让它们轮流坐庄，别互相卡顿。但"轮流"不等于"不干扰"——它们的性格几乎在每个维度上都相反。

![prefill 吃算力、decode 吃显存带宽；一次重 prefill 之后跟着很多次轻 decode](/assets/posts/pd-disaggregation/diagram-two-phase-profiles.png)

prefill 一次并行吞掉整段 prompt，算力密集，一轮很重。decode 每步只算一个新 token，却要把**整个**不断变长的 KV Cache 读一遍，访存密集，一轮很轻。你把这两个塞进同一张卡，等于让一个人同时干两份性格相反的活：忙起来两头都别扭。

更要命的是配置。prefill 想要大算力、小并行度；decode 想要大显存带宽、大并行度（把 KV 摊薄）。一套 TP、一种卡，注定两头都不最优。要么牺牲 TTFT，要么牺牲 TPOT，要么两边都留余量、硬件利用率上不去。

**这就是你一直在付、却从没写进账单的那笔钱——干扰成本。** 它藏在延迟的抖动里，负载越高越贵，但你看不见它，也没法单独给它开一张发票。

PD 分离做的第一件事，就是把这笔隐性成本变成一件看得见的事。

## 第一刀：把两个阶段外包出去

既然两个阶段互相拖累，最直接的办法是别让它们共享 GPU 了，一人一个池。

![colocated：一张卡两阶段互相干扰、配置二选一；disaggregated：两个池各司其职，中间一条 KV 传输链路交接](/assets/posts/pd-disaggregation/diagram-colocated-vs-disagg.png)

- **Prefill 池**：只把 prompt 变成 KV Cache。
- **Decode 池**：只负责一个 token 一个 token 地稳定吐字。
- 一个请求先在 prefill 池生成 KV，**交接**给 decode 池，再续写到结束。

这就是 DistServe（OSDI'24）和 Splitwise（ISCA'24，微软）的基本盘。DistServe 的原话是：colocated 既带来 prefill–decode 干扰，又把两阶段的资源和并行度**耦死**了；拆开之后，干扰**直接消失**。

想象一家工厂，原来备料和出餐都在同一个灶台。现在你把备料这道工序**外包**给一个专业供应商：他有更专业的设备，能独立扩产，你的灶台也清净了。听起来全是好处——直到你发现，供应商在城另一头，每份半成品都得**运过来**。

那笔运费，就是下面所有故事的主角。

## 外包的红利：各用各的最优解

先说甜头，这也是大家愿意拆的原因。拆开之后，每个池都能被单独打磨到极致。

![prefill 池用小 TP、算力型 GPU；decode 池用大 TP、带宽型 GPU；两池按流量独立选择实例数量](/assets/posts/pd-disaggregation/diagram-phase-specific.png)

**各自的并行度。** prefill 用小 TP 就能喂饱，decode 用大 TP 把 KV 摊到更多卡、拿到更多聚合带宽。**各自的硬件。** Splitwise 的核心洞察是：decode 根本用不上最新最贵 GPU 的澎湃算力，扔到更便宜、更省电的卡上就行，把最强的卡留给 prefill——同样吞吐，成本降 20%；或者同样成本和功耗，吞吐提到 2.35×。**各自的扩缩容。** 线上 prefill 和 decode 的比例一直在变，长文档场景 prefill 重、闲聊场景 decode 重，两个池按 P:D 比例各加各的实例，不用为迁就一头而整体扩容。

一句话：备料车间配大功率绞肉机，出餐窗口多开几个、配保温台，中午给备料加人、晚上给窗口加人。各扩各的。

红利很实在。但它不是免费的。

## 那笔运费：KV 得搬过去，而且很重

拆开的代价，是凭空多出一条原本不存在的链路：prefill 池算出的 KV Cache，必须搬到 decode 池去。KV 可能有**好几个 GB**——长上下文尤其吓人。

这条链路能不能喂饱 decode，直接决定整个架构是赚还是赔。而它有多贵，取决于你用什么线来搬。互连带宽的差距，是**几个数量级**的：

![互连层级：NVLink（节点内，~100s GB/s）> InfiniBand/RoCE·RDMA（~10s GB/s）> PCIe（~10 GB/s）> TCP（~1 GB/s）](/assets/posts/pd-disaggregation/diagram-interconnect.png)

NVLink 是供应商就开在你隔壁厂区，叉车推过去几秒钟。InfiniBand/RoCE 是同城货运。PCIe 已经是慢船。到了普通 TCP，等于跨洋海运一盆会化的冰淇淋——等它到岸，decode 早凉透了。

搬的方式也分两派。**点对点**：prefill 的显存直接搬到 decode 的显存，Dynamo 用 NIXL 做非阻塞直传，vLLM 的 NixlConnector 走 RDMA。**池化**：把 KV 存进一个横跨 CPU/DRAM/SSD 的分布式池，谁要谁取——Mooncake（FAST'25 最佳论文，Kimi 的服务平台）就是这套 KVCache-centric 思路，顺手把集群里闲置的内存和 SSD 榨出来当 KV 池。

派别是工程选择。但**运费本身贵不贵，是物理决定的。** 所以真正的问题只有一个——

## 这篇的灵魂：30 GB/s 那条线

拆开省下的，是 colocated 里 prefill 对 decode 的干扰；多花的，是每个请求的 KV 运费。两项都随负载线性增长，所以负载会在等式两边同时约掉。剩下的，是一个只跟**互连带宽**有关的、干干净净的盈亏平衡问题。

把它算出来。下面这段仿真隔离地验证这一个效应，`SEED=42` 可复现，无需 GPU、模型、联网：

```python
import numpy as np

KV_BYTES_PER_TOKEN = 2 * 80 * 8 * 128 * 2   # ~320 KiB/token（70B 级、GQA）

def when_to_disaggregate():
    rng = np.random.default_rng(42)
    avg_P = int(np.mean([int(np.clip(rng.lognormal(6.6, 0.5), 64, 8000))
                         for _ in range(20000)]))   # 平均 prompt ≈ 835 token
    compute_ms_per_token = 0.011      # colocated 里，每个 prompt token 挤占 decode 的量
    for bw in np.array([1, 5, 12.5, 40, 150, 400]) * 1e9:   # TCP..PCIe..IB..NVLink
        transfer_ms = KV_BYTES_PER_TOKEN * avg_P / bw * 1e3
        net = avg_P * compute_ms_per_token - transfer_ms   # >0：分离净赚
        print(f"bw={bw/1e9:6.1f} GB/s  net/req={net:+.2f} ms -> "
              f"{'disagg 胜' if net > 0 else 'colo 胜'}")
    bw_star = KV_BYTES_PER_TOKEN * 1e3 / compute_ms_per_token / 1e9
    print(f"break-even ≈ {bw_star:.1f} GB/s")   # 与负载无关
```

![负载 × 互连带宽的净收益地图：绿区（高带宽）分离净赚，红区（低带宽）colocated 更优，分界线约在 30 GB/s](/assets/posts/pd-disaggregation/sim-when-to-disaggregate.png)

跑出来的数，是这篇文章最值得你记住的东西：

| 互连带宽 | 场景 | 每请求净收益 | 谁赢 |
|---------|------|------------|------|
| 1 GB/s | TCP | **−264 ms** 🔴 | colocated |
| 5 GB/s | 慢 PCIe | **−46 ms** 🔴 | colocated |
| 12.5 GB/s | PCIe | **−13 ms** 🔴 | colocated |
| 40 GB/s | InfiniBand | **+2.3 ms** 🟢 | disaggregated |
| 150 GB/s | NVLink | **+7.4 ms** 🟢 | disaggregated |
| 400 GB/s | NVLink 满配 | **+8.5 ms** 🟢 | disaggregated |

分界线落在**约 30 GB/s**。线以下，运费吃掉了全部红利，你辛辛苦苦拆出来的架构，跑不过一台调好的 colocated + chunked prefill——后者约 20~40% 收益，还零新基建。线以上，分离才开始赚，而且越往右越香。

回到开头那个"比单机还慢"的场景：现在你知道答案了。你的机器落在了线的左边。不是 PD 分离不行，是你的运费太贵。

## 把运费藏进生产线

分界线不是死的。工程上有两把撬棍，能把它往左压。

第一把，是**别等整批做完再发货**。最朴素的串行传输，是 prefill 把所有层都算完、再整体传 KV，decode 只能干等，白白多出一段 TTFT。聪明的做法是 layer-wise：某一层的 KV 一算完，立刻开始传，同时去算下一层——传输藏在计算背后。

![串行：算完所有层再传，decode 干等；layer-wise：每层算完即传、与下一层计算重叠](/assets/posts/pd-disaggregation/diagram-kv-transfer.png)

同样的可复现仿真（InfiniBand ≈ 40 GB/s、80 层）告诉我们：layer-wise 把交接开销**藏掉约 55%**。P=8000 时，串行要给 TTFT 多加 65.5 ms，layer-wise 只剩 29.6 ms；P=1024 时是 8.4 ms → 3.8 ms。Splitwise 就是这么干的：小 prompt 反正不大，串行传；大 prompt 用 layer-wise 把延迟塞进计算里，用户几乎无感。

剁好一部分就先递过去，等你剁完，前面的也送到了。搬运的时间，被剁馅的时间盖住了。

第二把撬棍，是**换一把尺子**。别再看原始吞吐——它会骗你。一个请求哪怕被"服务"了，只要 TTFT 或 TPOT 违反了 SLO，对用户来说这次服务就是废的。DistServe 因此主张用 **goodput**：在同时满足两个 SLO 的前提下，每张 GPU 能扛住的最大请求率。

![goodput vs 负载：colocated 因干扰很早崩塌；disaggregated 把 SLO 拐点往右推](/assets/posts/pd-disaggregation/sim-goodput-vs-load.png)

还是那组仿真：colocated 峰值 goodput 只有 88 req/s，负载一高就因干扰崩塌；disaggregated 达到 200 req/s，整条曲线明显更抗压，**峰值 2.27×**。DistServe 在真机上报告得更狠：满足 SLO 的前提下能服务 7.4× 的请求量，或在同样负载下满足 12.6× 更紧的 SLO。

一家餐厅一小时"接待"500 桌听着很猛，但一半人等太久摔门而去——真正吃上饭的才算数。goodput 数的就是这个。

## 那为什么大厂全在用？

说到这儿你可能想反驳：既然拆了可能赔本，为什么 2025 年主流推理栈——vLLM、Dynamo、Mooncake——全把 PD 分离当标配？

因为它们**恰好全站在线的右边**。

超大规模生产集群里，RDMA 和 NVLink 是标配，prompt 长、并发高、SLO 严。这正是干扰最凶、分离收益最大、而运费又最便宜的那一侧。Mooncake 把它推到极致，用 KVCache-centric 分离撑起 Kimi 日处理千亿 token，真实 trace 上有效容量 +59%~498%。他们还愿意为"把运费藏起来"砸重金——layer-wise、池化 KV、带宽感知放置，全是在把那条线往左推。

所以标配这件事，非但没有推翻"这是一笔可能赔本的交易"，反而是最硬的证据：**正因为它可能赔，值得赢的玩家才把互连、把传输重叠、把放置策略，一样一样做到极致，好让自己稳稳待在线的右边。** 你如果只有几台 TCP 连着的卡，最优解不是模仿他们拆，而是先把 chunked prefill 用满。

NVIDIA 官方也反复讲同一句话：最优工作点取决于模型、GPU、后端、请求长度分布，没有一个固定阈值。这篇给你的 30 GB/s，是个帮你建立直觉的量级，不是一把万能尺。

## 一句话，和一扇门

PD 分离不是"拆开两个阶段"这么简单。它是拿一笔看得见的 KV 运费，换单机里那笔看不见的干扰账单——**只有当你的互连快到让这笔交换净赚时，拆才成立。** 快互连、重负载、长 prompt，三者齐了才划算；否则先把 chunked prefill 用满，再谈拆。

现在假设你已经站在了线的右边，拆成了 prefill 池、decode 池、KV 池，散在整个集群上。新问题立刻冒出来：一个请求该去哪台 prefill、哪台 decode？怎么让路由知道它要的 KV 缓存**正躺在哪台机器上**（命中就省一次 prefill）？过载时，怎么提前拒绝那些注定超时的请求？

外包解决了"谁来干"，但没解决"货往哪送、谁来调度"。那是一整个集群级的问题——下一篇见。
