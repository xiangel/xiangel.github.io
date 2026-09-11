---
author: xiangel
pubDatetime: 2026-09-10T06:30:00Z
title: "大模型推理调度（分布式篇）：从缓存感知路由到全局准入控制"
slug: llm-inference-scheduling-distributed
featured: true
draft: false
tags:
  - 大模型推理系统
  - 推理调度
  - LLM
description: 一张卡装不下，就得把请求撒到一整个集群上。可一旦有了很多机器，"派单"本身就成了新问题——请求该去哪台 prefill、哪台 decode？用打车平台派单中心做类比，沿着"缓存命中 / 负载均衡 / 准入控制"三个互相打架的目标，讲清缓存感知路由、Preble E2、Mooncake Conductor、early rejection 和弹性扩缩容。附一组无需 GPU、可复现的仿真。
---

如果你把一个大模型服务从单机扩到了一整个集群，多半撞见过两个"反直觉"的现象：

1. 明明**加了机器**、总显存更大了，整体的 **KV 缓存命中率反而更低**、TTFT 更差了。
2. 流量高峰时，你想"**多接总比拒好**"，于是来者不拒——结果**成功的请求反而更少**了。

这两件事背后，是同一个模块在做决定：**集群调度器 / 路由器（router）**。它回答的问题，和单机调度是同一个母题，但换了个尺度——

> 有一堆用户请求、**很多台** GPU（还分成 prefill 池和 decode 池）、散落在各机器上的 KV 缓存，那么**每个请求该发给哪台 prefill、哪台 decode，过载了又该拒谁**？

这是本系列第五篇，也是"调度"这条线的收尾。[第一篇](/posts/from-causal-lm-to-inference-system/)讲推理系统的模块怎么长出来，[第二篇](/posts/kv-cache-paged-attention-and-prefix-caching/)讲 KV Cache 怎么管，[第三篇](/posts/llm-inference-scheduling-single-node/)讲**单机内部**怎么调度，[第四篇](/posts/llm-inference-pd-disaggregation/)讲怎么把 **prefill 和 decode 拆到两个 GPU 池（PD 分离）**。这一篇往上再走最后一步：**当 prefill 池、decode 池、KV 池散布在整个集群上，谁来给每个请求"派单"？**

> **说明**：第三篇聚焦的是**一个引擎内部**一次 forward 该算谁；本篇聚焦的是**跨实例**的那一层——请求进集群后先要被路由到某台机器，之后才轮到单机调度接手。两层是嵌套关系，不是替代关系。

全程我用一个贯穿的类比:**打车平台的派单中心**。请求就是**订单**，每台 GPU 实例是一位**司机**，前缀缓存命中就是**顺路 / 熟客**（省一大段重复路程），负载均衡就是**别让某个司机接爆、别人却空跑**，而 early rejection 就是**高峰期运力不够时干脆不接单**——而不是接了让乘客干等到崩溃。沿着系列一贯的"**每暴露一个问题，就引入一种优化，又带出新问题**"的主线往下讲。

## Table of contents

## 一、集群调度在调什么

先把问题的"形状"说清楚。单机调度纠结的是"这一轮 forward 里算谁、和谁拼批"；到了集群这层，问题变成一个**派单问题**：一个请求进来，要先决定它去哪。在 PD 分离的架构里，这个决定甚至是**一对**——选一台 prefill 实例、再选一台 decode 实例。

![一个派单中心，三个互相打架的目标：缓存命中、负载均衡、准入控制](/assets/posts/distributed-scheduling/diagram-dispatch-overview.png)

派单要同时兼顾三个目标，而它们**互相打架**：

1. **缓存命中（affinity）**：把带同一个长前缀（system prompt / RAG 文档 / 多轮对话历史）的请求，发给**已经缓存了这段 KV 的那台机器**，就能省下一整段 prefill。这要求"把相似的请求往一起送"。
2. **负载均衡（load balance）**：可如果所有热门请求都往同一台送，那台就会**过载**，其他机器空转。这要求"把请求摊开"。
3. **准入控制（admission）**：当整个集群都满了，还硬接单只会让**所有人**一起超时。这要求"该拒就拒"。

> **类比**：这就是派单中心每天的纠结。把订单都派给最熟路的司机（命中），那位司机会被接爆；雨露均沾地轮流派（均衡），又没人顺路、全程空驶；高峰期运力见底，还一直接单（不准入），最后是每一单都迟到。

单机调度的终极矛盾是"固定显存下平衡吞吐和延迟"；集群调度的终极矛盾则多了一维空间：**在缓存命中、负载均衡、准入控制之间选点，而请求去哪、要算多久，事先都不完全可知。** 下面几节，就是业界一步步逼近这个目标的过程。

## 二、第一个问题：加了机器，缓存却更凉了 —— 缓存感知路由

先看开头那个反直觉现象。上一篇讲过，prefix caching 能让"命中前缀"的请求跳过一大段 prefill。但缓存是**每台机器各存各的**（本地 KV）。如果路由器用最朴素的 **round-robin（轮询）**——不看内容、雨露均沾——会发生什么？

同一个长前缀的一批请求，被**均匀撒到了所有机器**上。于是每台机器都只见过这个前缀的一小部分流量，谁也没能把它"焐热"；机器越多，同一前缀的流量被稀释得越厉害，**命中率随集群规模不升反降**。你花钱加的机器，反而把缓存摊薄了。

![round-robin 把同前缀撒得到处都是（miss，重算）；cache-aware 把它送给已缓存的那台（hit，跳过 prefill）](/assets/posts/distributed-scheduling/diagram-cache-aware-routing.png)

解法很直接：**路由器要知道"谁手上有什么缓存"**，把请求送给**已经缓存了最长匹配前缀**的那台机器。这就是 **SGLang 的 cache-aware router**：每台 worker 维护一棵**近似基数树（approximate radix tree）**，记录自己近期缓存过哪些前缀；router 保存这些树的一份近似副本，来一个请求就查"谁的匹配前缀最长"，优先发给它。SGLang 报告这套路由把缓存命中率从约 20% 拉到约 75%，带来最高约 1.9× 的吞吐提升。

> **类比**：这就是派单从"就近轮流"升级成"**认熟客**"。老顾客打车，直接派给那个知道他家在哪、常走哪条路的司机——不用再从头问一遍地址。

代价是什么？router 得**维护并同步**每台机器的缓存状态。用精确的方式（每次缓存增删都上报）最准但开销大；用近似基数树 + 定期同步则便宜但会有偏差。这条"精确 vs 近似"的取舍，后面还会再遇到。

## 三、第二个问题：光追命中会把机器压垮 —— 亲和 vs 均衡

只按"最长前缀"派单，很快撞上第二个目标的反扑。设想有个**超热的前缀**（比如一份人人都在问的爆款文档）：纯亲和会把它的**所有**请求都怼给同一台机器。那台被接爆、排起长队，其他机器却在空转。**命中率是满了，尾延迟却炸了**——这就是**热点（hotspotting）**。

![亲和/均衡的两种失败模式，以及 Preble E2 的决策规则](/assets/posts/distributed-scheduling/diagram-affinity-vs-load.png)

反过来，纯**负载均衡**（比如经典的 **power-of-two-choices**：随机采两台、挑负载轻的那台）能让集群非常均匀，却完全无视缓存——每台都在重算前缀，命中率崩到地板。

于是问题变成：**什么时候该"黏"着缓存（exploit），什么时候该"摊开"去均衡（explore）？** **Preble** 给了一个很漂亮的判据 **E2（Exploitation + Exploration）**：

> 对每个请求，比较**"命中前缀能省下的重算量"**和**"这个请求本来就要新算的量"**。如果**省下的 > 新算的**（说明这次复用很值），就**黏**到缓存所在的机器；否则（复用没多少油水），就去**当前最轻**的机器。

一句话：**复用的收益大到值得为它排队，才亲和；否则老老实实均衡。** SGLang 的 router 用的是同一思想的另一种写法——**命中就亲和，但加一道负载护栏**：一旦缓存所在机器的负载超过均值的某个倍数（比如 1.5×），就放弃亲和、改投最轻的机器。两者殊途同归，都是在"亲和"和"均衡"之间放了一个**动态开关**。

> **一点渊源**：这个"按内容亲和、但给单机负载封顶"的思路，在分布式系统里早有经典对应——**consistent hashing with bounded loads**（一致性哈希 + 负载上限，Google 2017）：请求按 key 哈希到固定节点（亲和），可一旦该节点超过负载上限，就顺延到下一个节点（护栏）。LLM 路由不过是把"key"换成了"最长匹配前缀"、把"负载"换成了"KV / 队列压力"，本质是同一套"亲和为主、过载即溢出"的老配方。

> **类比**：熟客固然要认，但如果那位熟客司机手上已经压了五单，平台就该把新单派给旁边的空车——哪怕那位司机要重新问一次路。省下的"顺路"抵不过让乘客干等五单的代价。

这一节的取舍非常本质，后面的实验 B 会把它量出来：**纯亲和命中率最高但最不均衡，纯均衡最均衡但命中率最低，E2 / 带护栏的亲和落在两者之间的甜点上。**

## 四、第三个问题：KV 缓存散落在整个集群 —— KVCache 全局调度

PD 分离之后，事情又复杂了一层。现在一个请求要选的不是一台机器，而是一对：**(prefill 实例, decode 实例)**。而 KV 缓存既可能在某台 prefill 上，也可能被换出到一个**共享 KV 池**里。谁来统筹这一切？

**Mooncake** 的答案是一个全局调度器 **Conductor**（它服务的是 Kimi）。Conductor 的核心思想是 **KVCache-centric**——**一切围绕"KV 缓存在哪、搬过去多贵"来决策**。对每个请求，它会给候选的 (P, D) 对打分，估算一条**端到端的时间账**：

![Conductor 给每个 (P,D) 对打分，并管理一个热块复制、冷块换出的全局 KV 池](/assets/posts/distributed-scheduling/diagram-conductor.png)

> 选哪一对 = 让 **"前缀复用省下的时间" − "prefill 排队 + prefill 计算 + KV 传输"** 最优，且**满足 TTFT / TBT 的 SLO**。

注意它把**KV 传输的耗时也算进了账**——上一篇讲过，把 KV 从 prefill 搬到 decode 是要花互连带宽的，prompt 越长搬得越久。Conductor 会权衡"就近复用但要排队" vs "另找一台但要重算或长传输"，挑总时间最划算的那一对。

为了让"命中"更容易发生，Conductor 还主动经营这个全局 KV 池：**热块复制（hot-block replication）**——把最烫手的前缀多复制几份到不同实例，让更多机器都能命中；**冷块换出（cold swap）**——把很久没人用的块挪到更便宜的存储，给热块腾地方。

> **类比**：这已经不是单个调度员拍脑袋，而是一个**智能调度大脑**：它知道每位司机现在在哪、车上还有没有空座、把乘客的行李（KV）转交给另一辆车要多久，然后算一笔总时间账再派单。爆款目的地的"熟路司机"，它会**多培养几个**，免得全压在一个人身上。

## 五、第四个问题：过载时接了再丢 = 白烧算力 —— 全局准入控制

现在回到开头第二个反直觉现象：**过载时"多接单"为什么反而更糟？**

关键在于 LLM 请求的成本结构：一个请求要先花一大笔 **prefill**（算力），才能开始 decode 出字。如果集群已经满了，你还继续接单，请求就会在队列里越堆越久。等它终于轮到、prefill 也算完了，一看表——**TTFT 早就超了 SLO**，这个结果对用户已经没用了。**那笔 prefill 算力，纯纯白烧。** 更糟的是，这些注定超时的请求还在**和有救的请求抢** GPU，把本来能按时完成的也拖垮，形成雪崩。

![接了再丢：prefill 白烧算力；early rejection：入口就拒，零算力浪费](/assets/posts/distributed-scheduling/diagram-early-rejection.png)

解法是 Mooncake 提出的 **early rejection（提前拒绝）**：**在花任何 prefill 算力之前**，先估一下这个请求将要面对的负载——不只看 prefill 池，还要看它稍后要去的 **decode 池**（取两者负载的较大值），如果已经超过容量、注定无法在 SLO 内完成，就在**入口直接拒掉**，一个 token 都不算。

> **类比**：高峰期平台运力见底，与其接了单让乘客在路边等 40 分钟最后取消，不如**一开始就提示"当前无可用车辆"**。司机的油（算力）只花在能准时送达的订单上。被拒的乘客还能立刻改约别的平台（客户端重试 / 降级），而不是被你吊着。

这里有个微妙但重要的细节：**early rejection 要看的是请求将来会面对的负载，而不只是此刻的瞬时负载。** 只看当下容易误判——刚拒完一批、瞬时负载掉下来，又开始猛接，结果下一刻再次过载，来回抖动。Mooncake 因此用了基于预测的准入策略。实验 C 会把"接了再丢"和"入口就拒"的差距直接量出来。

## 六、第五个问题：请求结构一直在变 —— 弹性扩缩容

前面几节都假设 prefill 池和 decode 池的**大小是固定的**。但真实流量不是：白天是长文档分析（prefill 重），晚上是多轮闲聊（decode 重）；一次热点事件能让某类请求瞬间暴涨。**固定的 P/D 比例，迟早会一头堵死、另一头闲置。**

这就需要**弹性扩缩容**：根据实时负载，动态调整两个池各分多少 GPU。NVIDIA 的 **Dynamo** 里专门有个 **Planner** 干这件事——它是个 **SLA 感知**的自动扩缩容器，盯着两类信号做决定：

- **decode 侧**：decode 池的 **KV block 利用率**——快满了说明 decode 扛不住，该给 decode 池加卡。
- **prefill 侧**：**全局 prefill 队列深度**——排队太长说明 prefill 是瓶颈，该给 prefill 池加卡。

配合 Dynamo 的 **KV-aware router**（路由时同时看 KV 命中重叠度和实例负载）和 **NIXL**（做 KV 的高速传输），整套系统能在负载漂移时自动把资源往瓶颈那头调。Dynamo 报告 Planner 能在明显更低的成本下大幅减少 SLA 违约。

> **类比**：这就是平台根据早晚高峰**动态调度运力**——早高峰多派车去写字楼（prefill 重的长文档），夜里把车挪到酒吧街（decode 重的闲聊）。哪里排队长，就往哪里调车。

> **注意**：扩缩容不是免费的。新实例**冷启动**要时间、要**预热缓存**；抖动太频繁反而有害。所以 Planner 通常按较粗的时间粒度、留足余量地调，而不是逐秒抖动——这又是一个"响应速度 vs 稳定性"的取舍。

## 七、把它们拼起来：一张集群调度全景

前面五节的优化不是互相替代，而是**叠在一起**，构成一个真实的集群调度栈。今天主流的几套开源 / 生产系统，形状大同小异：

| 系统                              | 路由怎么做                                 | 特色                                  |
| --------------------------------- | ------------------------------------------ | ------------------------------------- |
| **SGLang router**                 | 近似基数树 + 最长前缀，带负载护栏          | 缓存感知路由的典型实现                |
| **Preble**                        | E2：省下的重算 > 新算才亲和，否则去最轻    | 把"亲和 vs 均衡"做成显式判据          |
| **Mooncake Conductor**            | KVCache-centric，为 (P,D) 对算端到端时间账 | 热块复制 / 冷块换出 + early rejection |
| **NVIDIA Dynamo**                 | KV-aware router（命中重叠 + 负载）+ NIXL   | Planner 做 SLA 感知弹性扩缩容         |
| **vLLM production-stack / llm-d** | Gateway API Inference Extension 的 **EPP** | 标准化的 Filter → Score → Pick 三段式 |

值得单独点一下 **llm-d / Gateway API Inference Extension** 的 **EPP（Endpoint Picker）**，因为它把前面所有讨论**收敛成了一个标准接口**——路由决策被拆成三步：

1. **Filter**：先排除不可用的实例（比如 KV 已满、健康检查不过）。
2. **Score**：给候选实例打分，分数综合了**前缀局部性、KV 利用率、队列深度**等——正是前几节那些信号。
3. **Pick**：按分数选最终目标（可带一点随机化避免羊群效应）。

你会发现，无论叫 router、Conductor 还是 EPP，它们做的都是同一件事：**用 KV 缓存位置、实例负载、SLO 这几个信号，在"命中 / 均衡 / 准入"之间选点。** 差别只在信号多精确、判据多复杂。

## 八、动手实验：无需 GPU 的集群调度仿真

下面用纯 Python 仿真**隔离**地验证前面几个结论。每个实验只放大**一个**效应，方便看清因果（因此不是端到端基准，倍数不能直接等同真机）。全部 `SEED=42` 可复现，无需 GPU / 模型 / 联网。

公共前置（三个实验共用；下面每段只贴仿真核心逻辑，`matplotlib` 绘图代码从略）——我们造一条**带共享前缀的请求流**：请求按 zipf 分布归入若干"前缀组"（共享同一段 system prompt / 文档），少数热组占大部分流量；每组有一段可复用的前缀，每个请求还有各自唯一的后缀。再加一个简单的 LRU 表模拟每台机器的本地缓存。

```python
import numpy as np

SEED = 42  # 固定随机种子，三个实验都可复现

# 每个请求属于某个“前缀组”，组按 zipf 分布 —— 少数热组占大多数流量
def make_requests(n, n_groups, rng, zipf_a=1.1):
    ranks = np.arange(1, n_groups + 1)
    p = 1.0 / ranks ** zipf_a
    p /= p.sum()
    groups = rng.choice(n_groups, size=n, p=p)
    group_prefix = np.clip(rng.lognormal(6.7, 0.5, size=n_groups), 64, 8000).astype(int)
    prefix_len = group_prefix[groups]                                  # 组共享的可复用前缀
    suffix_len = np.clip(rng.lognormal(4.5, 0.7, size=n), 8, 2000).astype(int)  # 各请求唯一后缀
    return groups, prefix_len, suffix_len

# 每台 worker 的本地缓存（按容量做 LRU 淘汰）
class LRU:
    def __init__(self, cap):
        self.cap, self.d = cap, {}
    def has(self, g):
        return g in self.d
    def touch(self, g, t):
        if g in self.d:
            del self.d[g]
        self.d[g] = t
        if len(self.d) > self.cap:
            del self.d[next(iter(self.d))]  # 淘汰最久未用
```

### 实验 A：round-robin vs cache-aware —— 命中率随集群规模

同一条请求流，分别用 round-robin（轮询）和 cache-aware（命中就发给它、否则发给最轻的）路由，看**集群整体的前缀命中率**如何随 worker 数变化：

```python
def exp_a_once(W, n=20000, n_groups=800, cap_groups=24):
    rng = np.random.default_rng(SEED)
    groups, prefix_len, suffix_len = make_requests(n, n_groups, rng)
    results = {}
    for policy in ("rr", "cache"):
        caches = [LRU(cap_groups) for _ in range(W)]
        load = np.zeros(W)                       # 累计 token 负载
        hit_tok, tot_tok = 0, 0
        for i in range(n):
            g, pl = int(groups[i]), int(prefix_len[i])
            if policy == "rr":
                w = i % W                         # 轮询：不看内容
            else:                                 # cache-aware：命中就发给它，否则发给最轻的
                owners = [k for k in range(W) if caches[k].has(g)]
                w = min(owners, key=lambda k: load[k]) if owners else int(np.argmin(load))
            if caches[w].has(g):
                hit_tok += pl                     # 前缀命中，省下这段 prefill
            tot_tok += pl
            caches[w].touch(g, i)
            load[w] += pl + int(suffix_len[i])
        results[policy] = hit_tok / tot_tok
    return results["rr"], results["cache"]        # 对每个 W ∈ {2,4,8,16,32} 各跑一次
```

![cache-aware 的命中率随集群变大稳步走高，round-robin 一直趴在原地](/assets/posts/distributed-scheduling/sim-routing-hitrate.png)

一手结果:`W=8` 时，round-robin 的集群命中率只有 **36.7%**，cache-aware 达到 **74.7%**；而且随着集群从 2 台涨到 32 台，cache-aware 从 **55% 一路升到 91%**（机器越多、越能把不同前缀分工缓存），round-robin 却始终**趴在 ~37%**、甚至略降。这就精确复现了开头那个反直觉现象——**加机器本身会稀释缓存，只有让路由"认缓存"，扩容才真正转化为命中率。**

### 实验 B：亲和 / 均衡 / E2 —— 命中率与负载不均的取舍

固定 8 台机器、更陡的热点分布，对比四种策略：纯亲和、power-of-two（纯均衡）、Preble E2、带负载护栏的亲和（SGLang 风格）。两个指标:前缀命中率（越高越好）和负载不均 `max/mean`（越低越好）：

```python
def exp_b(W=8, n=20000, n_groups=120, cap_groups=40):
    rng = np.random.default_rng(SEED)
    groups, prefix_len, suffix_len = make_requests(n, n_groups, rng, zipf_a=1.4)  # 更陡的热点

    def run(policy):
        caches = [LRU(cap_groups) for _ in range(W)]
        load = np.zeros(W)
        hit_tok, tot_tok = 0, 0
        rng2 = np.random.default_rng(SEED + 1)               # pow2 的随机采样
        for i in range(n):
            g, pl, sl = int(groups[i]), int(prefix_len[i]), int(suffix_len[i])
            owners = [k for k in range(W) if caches[k].has(g)]
            if policy == "affinity":                          # 纯亲和：同组永远同一台
                w = g % W
            elif policy == "balance":                         # power-of-two：采两台挑轻的
                a, b = rng2.integers(0, W, size=2)
                w = int(a) if load[a] <= load[b] else int(b)
            elif policy == "e2":                              # Preble E2：省下的 > 新算的才亲和
                w = min(owners, key=lambda k: load[k]) if (owners and pl > sl) else int(np.argmin(load))
            elif policy == "sglang":                          # 命中就亲和，但热点(>1.5×均值)时切最轻
                if owners and load[min(owners, key=lambda k: load[k])] <= 1.5 * (load.mean() + 1):
                    w = min(owners, key=lambda k: load[k])
                else:
                    w = int(np.argmin(load))
            if caches[w].has(g):
                hit_tok += pl
            tot_tok += pl
            caches[w].touch(g, i)
            load[w] += pl + sl
        return hit_tok / tot_tok * 100, load.max() / load.mean()   # 命中率%, 负载不均
```

![四种策略在“命中率 vs 负载不均”平面上的位置，甜点在右上偏左](/assets/posts/distributed-scheduling/sim-affinity-vs-load.png)

一手结果:

| 策略                   | 命中率    | 负载不均 (max/mean)   |
| ---------------------- | --------- | --------------------- |
| 纯亲和                 | **99.2%** | **2.03×**（热点严重） |
| power-of-two（纯均衡） | 84.0%     | **1.00×**（最均衡）   |
| Preble E2              | 99.1%     | 1.36×                 |
| 带护栏的亲和（SGLang） | **99.2%** | **1.03×**             |

这张表把第三节的取舍量得清清楚楚:**纯亲和命中率顶格但把一台机器压到 2 倍负载;纯均衡完美均衡却牺牲了 15 个百分点的命中;而 E2 和带护栏的亲和都落在"命中率几乎不掉、负载又拉回接近均衡"的甜点上**——这正是生产路由器想要的位置。

### 实验 C：accept-all vs early rejection —— 过载下的 goodput

模拟一个**明显过载**的集群（到达率约为 decode 容量的 1.4×）：decode 池只有 M 个并发槽位，prefill 有固定吞吐。对比两种策略——**接了再算**（accept-all）和**入口准入**（inflight ≥ 容量就拒）。指标是 **goodput**（在 SLO 内完成的 req/s）和**白烧的 prefill token**：

```python
def exp_c(horizon_ms=60000):
    M = 32                        # decode 并发槽位（整个 decode 池）
    prefill_tok_per_ms = 600.0    # prefill 池吞吐
    slo_ttft = 2000.0             # 首字 SLO(ms)，含排队 + prefill + 等 decode 槽位
    a_iter, b_iter = 0.4, 0.006   # decode 单轮耗时 ≈ a + b×并发
    rate = 0.44                   # 到达率(req/ms)，设到 decode 可持续吞吐的 ~1.4×，制造过载

    def simulate(early):
        rng = np.random.default_rng(SEED)      # 两种策略同一条到达流
        clk, nxt = 0.0, rng.exponential(1.0 / rate)
        prefill_free = 0.0
        pend, running = [], []                 # 等 decode 槽位的；decode 中的
        good, waste_tok, rejected = 0, 0, 0
        while clk < horizon_ms:
            while nxt <= clk:
                P = int(np.clip(rng.lognormal(6.6, 0.5), 64, 8000))
                inflight = len(running) + len(pend)
                if early and inflight >= M:     # 准入：没余量就在 prefill 前拒掉
                    rejected += 1
                    nxt += rng.exponential(1.0 / rate); continue
                start = max(prefill_free, nxt)  # 接受 → 占用 prefill 池
                prefill_free = start + P / prefill_tok_per_ms
                pend.append([prefill_free, nxt, P])
                nxt += rng.exponential(1.0 / rate)
            pend.sort(key=lambda r: r[0])
            while pend and pend[0][0] <= clk and len(running) < M:
                ready, arr, ptok = pend.pop(0)
                out = int(np.clip(rng.lognormal(4.8, 0.7), 4, 1000))
                ttft = clk - arr                # 首字延迟 = prefill 排队 + prefill + 等 decode 槽位
                ok = ttft <= slo_ttft
                if not ok:
                    waste_tok += ptok           # 已违反 TTFT：这次 prefill 白算了
                running.append([out, ok, ptok])
            clk += a_iter + b_iter * len(running)
            still = []
            for r in running:                   # 每轮所有 decode 推进一步
                r[0] -= 1
                if r[0] <= 0:
                    good += 1 if r[1] else 0
                else:
                    still.append(r)
            running = still
        for _, _, ptok in pend:                 # 还堵在队列、始终没进 decode 的也白花了
            waste_tok += ptok
        return good / (clk / 1000.0), waste_tok, rejected
```

![过载下：early rejection 的 goodput 远高于 accept-all，且白烧算力清零](/assets/posts/distributed-scheduling/sim-early-rejection.png)

一手结果（同一条过载到达流）:accept-all 的 goodput 只有 **51.1 req/s**，还白烧了 **19.35 M** 个 prefill token（这些请求 prefill 算完却已超时）；early rejection 的 goodput 达到 **326.8 req/s**、白烧 token **归零**——**goodput 6.4×，无效计算 100% 消除**。这就是开头第二个反直觉现象的解释：**过载时"来者不拒"会把算力浪费在注定超时的请求上，还拖垮有救的请求;入口就拒，反而让更多请求真正成功。**

## 九、总结与延伸

集群调度的主线，还是那条"发现问题 → 引入优化 → 带出新问题"的链子:

- **加机器反而稀释缓存** → **缓存感知路由（SGLang）**：路由认缓存，把同前缀汇聚到已缓存的机器。
- **光追命中会压垮机器** → **亲和 vs 均衡（Preble E2）**：省下的重算 > 新算才亲和，否则去最轻。
- **KV 散落在整个集群** → **KVCache 全局调度（Mooncake Conductor）**：给 (P,D) 对算端到端时间账 + 热块复制 / 冷块换出。
- **过载接了再丢 = 白烧** → **全局准入控制（early rejection）**：按将面对的负载在入口就拒。
- **请求结构一直在变** → **弹性扩缩容（Dynamo Planner）**：盯着 KV 利用率和 prefill 队列深度动态调 P/D 池。

把它们拼起来，无论叫 router、Conductor 还是 EPP，做的都是同一件事:**Filter → Score → Pick**，用 KV 缓存位置、实例负载、SLO 这几个信号在"命中 / 均衡 / 准入"之间选点。

一句话带走:**集群调度就是给每个请求"派单"——在缓存命中、负载均衡、准入控制这三个互相拉扯的目标之间，用一次决策同时选好去哪台 prefill、哪台 decode，以及到底接不接。**

延伸阅读:到这里，调度这条线收束了——模块怎么长出来（第一篇）、KV Cache 怎么管（第二篇）、单机怎么调度（第三篇）、prefill/decode 怎么拆（第四篇）、集群怎么派单（本篇）。下一篇[《大模型的各种并行：从张量切分到上下文并行》](/posts/llm-inference-parallelism-moe/)往模型内部走：TP / SP / PP / DP / CP / EP 各切哪一维，以及 MoE 把专家并行推到跨节点之后，通信和热点专家怎么接。

## 参考

1. Kwon et al., [_Efficient Memory Management for Large Language Model Serving with PagedAttention_](https://arxiv.org/abs/2309.06180)（vLLM）, SOSP 2023.
2. Zheng et al., [_SGLang: Efficient Execution of Structured Language Model Programs_](https://arxiv.org/abs/2312.07104), NeurIPS 2024.
3. Srivatsa et al., [_Preble: Efficient Distributed Prompt Scheduling for LLM Serving_](https://arxiv.org/abs/2407.00023), ICLR 2025.
4. Qin et al., [_Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving_](https://arxiv.org/abs/2407.00079), FAST 2025.
5. NVIDIA, [_Dynamo: A Datacenter Scale Distributed Inference Serving Framework_](https://github.com/ai-dynamo/dynamo)（KV-aware router + SLA-aware Planner + NIXL）, 2025.
6. Kubernetes SIG Network, [_Gateway API Inference Extension_](https://gateway-api-inference-extension.sigs.k8s.io/)（llm-d / vLLM production-stack 的 Endpoint Picker）, 2025.
7. Mitzenmacher, [_The Power of Two Choices in Randomized Load Balancing_](https://www.eecs.harvard.edu/~michaelm/postscripts/tpds2001.pdf), IEEE TPDS 2001.
8. Mirrokni et al., [_Consistent Hashing with Bounded Loads_](https://arxiv.org/abs/1608.01350), 2017.
