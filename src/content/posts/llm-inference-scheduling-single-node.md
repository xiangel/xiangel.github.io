---
author: xiangel
pubDatetime: 2026-09-09T02:30:00Z
title: "大模型推理调度（单机篇）：从 Continuous Batching 到缓存感知调度"
slug: llm-inference-scheduling-single-node
featured: true
draft: false
tags:
  - 大模型推理系统
  - 推理调度
  - LLM
description: 单卡引擎内部，一次 forward 里到底该算谁、和谁拼一批？用操作系统调度做类比，沿着"每暴露一个问题就引入一种优化"的主线，讲清 continuous batching、chunked prefill、抢占、排队与公平、缓存感知调度。附一组无需 GPU、可复现的调度仿真。
---

如果你压测过自己部署的大模型，多半见过两个"反直觉"的现象：

1. 并发一上来，`nvidia-smi` 显示 GPU 利用率明明没满，**延迟却开始飙升**。
2. 同样一句话，有时**秒回**，有时要**排很久**队才轮到。

这两件事背后，都是同一个模块在做决定：**调度器（scheduler）**。它回答的是一个听起来简单、做起来极难的问题——

> 有一堆用户请求、一张（或几张）GPU、一块装不下所有请求的 KV 显存，那么**每一步该算谁、和谁拼一批、谁先谁后、显存不够踢谁**？

这是本系列第三篇。[第一篇](/posts/from-causal-lm-to-inference-system/)讲了推理系统的模块是怎么从 Transformer 里"长"出来的，[第二篇](/posts/kv-cache-paged-attention-and-prefix-caching/)讲了最吃显存的 KV Cache 怎么管。调度这一层往上再走一步。

**本篇只聚焦单机（单卡 / 单引擎）内部的调度**——连续批处理、chunked prefill、抢占、排队与公平、缓存感知调度。至于跨实例的**分布式（集群）调度**（prefill/decode 分离、全局 KV 感知路由），得先把 **PD 分离**这块地基讲透，所以留到后面**单独成篇**再展开，本篇不涉及。

全程我用**操作系统的 CPU 调度**做类比（进程、时间片、抢占、优先级、饥饿——这些几十年前就研究透了），沿着"**每暴露一个问题，就引入一种优化**"的主线往下讲。

## Table of contents

## 一、调度到底在调什么

先把问题的"形状"说清楚。LLM 推理有三个特点，让它的调度和传统 Web 服务完全不同：

1. **一个请求要跑很多轮**。生成是自回归的：处理完 prompt（**prefill**，一次并行吃完整段输入），然后一个 token 一个 token 地吐（**decode**，每步只算一个新 token）。一个请求就是"1 次 prefill + 几十到几千次 decode"。
2. **每个请求要占着一块 KV 显存，而且越占越多**。KV Cache 随着生成不断变长（详见上一篇），显存是**硬约束**——装不下就得有人出局。
3. **两个延迟指标互相打架**。用户体感被拆成两段：

| 指标                                    | 含义                 | 由什么决定                 |
| --------------------------------------- | -------------------- | -------------------------- |
| **TTFT**（Time To First Token）         | 从发出到看到第一个字 | prefill 快不快、排队久不久 |
| **TPOT / TBT**（每个后续 token 的间隔） | 吐字顺不顺           | decode 每轮迭代的耗时      |

> **类比**：这几乎就是操作系统的 **CPU 调度**。请求 = 进程；GPU 的每一次 forward = 一个**时间片**；KV 显存 = 有限的物理内存；TTFT/TPOT = 响应时间与吞吐。OS 调度器几十年来纠结的"吞吐 vs 延迟、公平 vs 效率、要不要抢占"，在这里几乎原样重演——只不过"进程"会不断长大、还随时可能被换出。

调度的终极矛盾一句话：**在固定的 KV 显存预算下，同时把吞吐和两个延迟指标做好，而请求长度事先不可知。** 下面五节，就是业界一步步逼近这个目标的过程。

## 二、第一个问题：请求级批处理在浪费 GPU —— Continuous Batching

GPU 要"喂饱"才划算：一次算 1 个请求和一次算 32 个请求，耗时差不多，但吞吐差几十倍。所以必须**批处理（batching）**。

最朴素的做法是**静态批处理**：攒够一批请求，一起跑，**等整批都生成完**再返回、再收下一批。问题是——**同一批里请求的输出长度天差地别**。一个请求生成 5 个 token 就结束了，另一个要生成 500 个。短的早就算完了，却只能**空占着槽位干等**批里最慢的那个。

```text
静态批处理（batch=4）：短请求算完只能空等最慢的
  slot1: R1 R1 R1 ▢ ▢ ▢ ▢ ▢   ← 算完了，白占 5 轮
  slot2: R2 R2 R2 R2 R2 R2 R2 R2
  slot3: R3 R3 ▢ ▢ ▢ ▢ ▢ ▢
  slot4: R4 R4 R4 R4 ▢ ▢ ▢ ▢
         └── 整批跑完才能收新请求，新请求一直在门外等 ──┘
```

> **类比**：这就是没有时间片的**批处理系统**——一道作业跑完才轮到下一道。OS 早就抛弃了它，换成**分时（time-sharing）**。

**Orca（OSDI 2022）** 把这套搬了过来，提出 **iteration-level scheduling（迭代级调度）**，也就是今天人人都在用的 **continuous batching（连续批处理）**：**调度粒度从"整个请求"降到"一次 forward"**。每跑完一轮：谁生成完了立刻返回、腾出槽位；等待队列里的新请求**马上补进来**。批次的成分**每一轮都在变**。

![静态批处理里算完的槽位空转；连续批处理每轮把空出的槽位立即补满](/assets/posts/scheduling/diagram-static-vs-continuous.png)

这里有个绕不开的工程细节：不等长的请求怎么拼进同一个张量？Orca 的答案是 **selective batching（选择性批处理）**——对**逐 token 独立**的算子（Linear、LayerNorm、FFN），把所有请求的 token **拉平拼成一个大张量**一起算，吃满算力；只有 **attention** 这一步因为每个请求要看自己的 KV，才**逐请求**分开算。

收益有多大？Orca 论文对当年的 FasterTransformer 报告了最高 **36.9×** 的吞吐提升。**连续批处理是现代一切推理引擎的地基**——后面所有优化都建立在"每轮都能重新决策"这个能力之上。

## 三、第二个问题：prefill 和 decode 抢一张灶台 —— Chunked Prefill

连续批处理让"每轮补新请求"成为可能，但马上带出一个新麻烦：**新请求进来时要先做 prefill，而 prefill 和 decode 的"性格"完全相反。**

- **prefill**：一次要并行处理整段 prompt（可能几千 token），**算力密集**，一轮很重、很慢。
- **decode**：每个请求一轮只算 1 个 token，**访存密集**，一轮很轻、很快。

如果把一个长 prompt 的 prefill 塞进某一轮，这一轮就会变得**奇重无比**——而**同批所有正在 decode 的请求，都得陪着它一起等这一轮结束**。用户的体感就是：吐字吐得好好的，突然**卡顿一下**（TBT 尖刺）。Sarathi-Serve 论文测到，这种"naïve 混批"能让 TBT 恶化高达 **28×**。

```text
naïve 混批：一个长 prefill 独占一轮，拖垮所有人的 decode
  iter:  [dec] [dec] [====== PREFILL 整段 prompt ======] [dec] [dec]
                       ▲ 这一轮巨慢，所有 decode 一起卡顿
```

**Sarathi-Serve（OSDI 2024）** 的解法是两招组合：

1. **Chunked prefill（切块预填充）**：不再一轮吃完整段 prompt，而是把它**切成固定大小的小块**，每轮只喂一块，分摊到多轮。
2. **Stall-free scheduling（无停顿调度）**：给每一轮设一个 **token budget（令牌预算 τ）**。组批时**先把所有正在跑的 decode 塞进去**，再用剩下的预算额度**塞一块 prefill chunk**，保证每轮的总 token 数 ≤ τ。

![naïve 混批让 decode 卡顿；chunked prefill 把 prefill 切成预算大小的块，每轮形状接近一致](/assets/posts/scheduling/diagram-chunked-prefill.png)

这样每一轮的计算量都被"削平"到差不多大小，decode **再也不会因为邻座的 prefill 而卡顿**。

> **类比**：这正是 OS 的**时间片（time quantum）**。你不会让一个 CPU 密集的进程一口气跑 10 秒把别人饿死，而是切成一个个固定长度的时间片轮流跑。token budget 就是 LLM 版的时间片。

代价是什么？把长 prefill 切开会让它**总的 TTFT 稍微变长一点点**（多了一些分块的固定开销），换来的是**所有人的 TPOT 都稳了**。这是本篇第一次出现、但会反复出现的取舍：**TTFT ⇄ TPOT，按 SLO 调 τ 就是在这条线上选点。**

## 四、第三个问题：显存不够了怎么办 —— 抢占

连续批处理会把槽位尽量塞满，于是迟早撞上第二篇讲过的硬墙：**KV 显存不够了**。正在跑的请求每轮都在变长、要新的 KV block，可池子空了；等待队列里还有新请求想进来。

这时调度器必须做一件 OS 里最熟悉的事——**抢占（preemption）**：挑一个"受害者"请求，把它踢出去、回收它的 KV 显存，让位给别人；被踢的请求回到等待队列，之后再重新调度。

问题是：被踢的请求，它**已经算好的 KV Cache 怎么办**？两条路：

![显存打满时抢占的两条路：重算（丢弃 KV）与换出（KV 拷到 CPU 内存）](/assets/posts/scheduling/diagram-preemption.png)

- **重算（recompute）**：直接**丢掉** KV，回收显存最快最干脆。代价是这个请求重新调度时，得**从头再 prefill 一遍**——花的是**算力**。
- **换出（swap）**：把 KV **拷到 CPU 内存**里存着，之后再拷回来，不用重算。代价是**一来一回的 PCIe 带宽**，还占 CPU 内存。

> **类比**：这就是 OS 的 **swap / 换页**。recompute 像"丢掉缓存、要用再重建"，swap 像"把内存页换出到磁盘"。选哪个，取决于你更缺**算力**还是更缺**带宽**。

经验上，被抢占的请求往往**前缀不长、重算并不贵**，而 PCIe 带宽在高并发下很宝贵，所以现在不少引擎在显存压力下**默认优先重算**。抢占的存在也提醒我们：**一味把显存塞到 100% 并不是好事**——留一点余量，能显著减少来回抢占带来的抖动。

## 五、第四个问题：谁先算 —— 排队顺序与公平

前面都在讲"怎么把一批算好"，还没回答最开始那个问题：**等待队列里有一堆请求，先算谁？**

最简单、也是绝大多数引擎的默认策略是 **FCFS（先来先服务）**。它公平、无饥饿、实现简单，但有一个经典毛病——**队头阻塞（head-of-line blocking）**：如果队头是个要生成几千 token 的"长活儿"，后面一堆"几十个 token 就结束"的短请求，只能**干等它**。

> **类比**：超市里你只买一瓶水，前面那位推了满满一车正在结账。这就是 FCFS 的痛。OS 的答案是 **SJF/SRTF（短作业优先）**：让短的先走，**平均等待时间**能大幅下降。

但 LLM 有个特殊障碍：**输出多长，事先不知道。** 没有长度，就没法做 SJF。近两年的关键洞察是——**不需要知道准确长度，知道"谁比谁短"（相对排序）就够了**。**Learning to Rank（Fu et al., NeurIPS 2024）** 用一个很小的辅助模型（如 OPT-125M）预测请求输出长度的**相对排名**，据此近似 SJF/SRTF 调度，把聊天场景的延迟降了 **2.8–6.9×**。

不过 SJF 有代价，而且这个代价必须正视：**长请求会饥饿（starvation）。** 短请求不断插队，长请求可能迟迟排不上。这就引出**公平性**。

- **公平怎么定义？** 请求成本差好几个数量级（10 token 的闲聊 vs 10 万 token 的长文档），**按请求数**做限流毫无意义。**VTC（Virtual Token Counter，OSDI 2024）** 把网络里的 **WFQ（加权公平队列）** 搬了过来：给每个客户端维护一个**虚拟计数器**，按它已消耗的 **token 数**（输入、输出可加不同权重）往前走，每轮**优先服务计数器最小的客户端**。狂发请求的客户端只会**抬高自己的计数器**，不影响别人——天然做到隔离。
- **防饥饿**：给等待过久的请求**逐步提权**（类似 OS 的 aging），到阈值就强制让它上。

一句话收束这一节：**排队策略是在"效率"和"公平"之间选点。** FCFS 偏公平、SJF 偏效率、VTC/aging 在中间找平衡——没有免费的午餐。

## 六、第五个问题：排序还能顺便提命中率 —— 缓存感知调度

上一节的排序只看"长短"。但如果你还记得第二篇的 **Prefix Caching**，会发现排序里藏着**另一个维度**：**先算谁，直接决定了 KV 缓存的命中率。**

设想等待队列里，有些请求共享同一个长系统提示、同一份 RAG 文档（前缀能命中缓存），有些则毫无关联。如果调度器**在不相关的请求之间反复横跳**，缓存就会**颠簸（thrashing）**：刚缓存的前缀块还没被复用就被挤掉了，命中率暴跌。

**SGLang 的 cache-aware scheduling** 的做法很直接：**把等待队列按"能命中的前缀长度"排序，最长前缀优先（longest-prefix-first）。** 命中越多的请求先算，它需要重算的 prefill 就越少，还顺手把公共前缀"焐热"给后面的请求用。

![缓存感知调度：把等待队列按能命中的前缀长度排序，最长前缀优先，最大化 KV 复用](/assets/posts/scheduling/diagram-cache-aware.png)

SGLang 论文证明了一个漂亮的结论：**离线情况下，按最长公共前缀顺序处理，等价于对请求基数树做深度优先遍历（DFS），能取得最优命中率**；在线场景下这个贪心也能逼近离线最优（论文报告平均达到 96%）。

当然，老朋友又来了——**贪心会饥饿**：一个前缀谁都不匹配的请求，可能永远排不上。所以生产里通常要把它和公平策略揉在一起。**你会发现，"排序"这件事在 LLM 调度里被反复要求同时优化三个目标：延迟（SJF）、公平（VTC）、命中率（最长前缀）——它们经常互相打架，怎么权衡取决于你的业务。**

## 七、把它们拼起来：一个单引擎调度循环

前面五个优化不是互相替代，而是**叠在一起**。生产引擎（vLLM V1、SGLang）每一轮 `schedule()` 大致都是同一个形状：

![单引擎调度循环：先保运行中的 decode，显存不足就抢占，再按策略从等待队列 admit，最后组一个受 token 预算约束的混合批](/assets/posts/scheduling/diagram-schedule-loop.png)

1. **先保运行中的**：把正在 decode 的请求放进下一批（它们本来就得继续）。
2. **显存不够就抢占**：放不下就按策略挑受害者，recompute 或 swap 腾地方。
3. **从等待队列 admit**：按策略（FCFS / 优先级 / 最长前缀 / SJF）挑新请求，直到 token 预算或并发上限用尽。
4. **组一个混合批**：decode + prefill chunk 拼一起，总量 ≤ token budget，送去做一次 forward。
5. 回到第 1 步，**每一轮迭代都重来一遍**。

vLLM V1 甚至把 prompt token 和 output token **统一**成"一个请求这轮要算多少 token"，用一个 `{request_id: num_tokens}` 的预算字典统一管理——chunked prefill、prefix caching、speculative decoding 就都能自然地共存在同一个循环里。

## 八、动手实验：无需 GPU 的调度仿真

下面用纯 Python 仿真**隔离**地验证前面几个结论。每个实验只放大**一个**效应，方便看清因果（因此不是端到端基准，倍数不能直接等同真机）。全部 `SEED=42` 可复现，无需 GPU / 模型 / 联网。

公共前置（三个实验共用；下面每段只贴仿真核心逻辑，`matplotlib` 绘图代码从略）：

```python
import heapq
import numpy as np

SEED = 42  # 固定随机种子，三个实验都可复现

# 重尾的聊天输出长度：多数很短、少数极长（lognormal）
def decode_lengths(n, rng):
    x = rng.lognormal(mean=4.4, sigma=0.85, size=n)  # 中位数 ≈ 80
    return np.clip(x, 4, 2048).astype(int)
```

### 实验 A：静态批处理 vs 连续批处理

模拟真实的聊天长度分布（多数短、长尾），批大小 32，对比两种批处理：

```python
def exp_a():
    rng = np.random.default_rng(SEED)
    B, N = 32, 4000                       # 批大小、请求总数
    L = decode_lengths(N, rng)            # 每个请求要跑的 decode 步数
    used = int(L.sum())                   # 真正有用的“槽位·迭代”数

    # 静态批处理：固定 B 个一批，整批都要等批内最长的那个跑完
    static_iters = sum(int(L[i:i + B].max()) for i in range(0, N, B))
    static_util = used / (B * static_iters)

    # 连续批处理：B 个槽位始终填满，谁跑完立刻补进新请求
    heap, idx = [], 0
    while idx < B and idx < N:
        heap.append(int(L[idx])); idx += 1
    heapq.heapify(heap)
    cont_iters = 0
    while heap:
        step = heap[0]                    # 最先跑完的那个还差 step 步
        cont_iters += step
        heap = [r - step for r in heap]   # 所有在跑的推进 step 步
        newheap = []
        for r in heap:                    # 跑完的槽位立刻补队列里的新请求
            if r > 0:
                newheap.append(r)
            elif idx < N:
                newheap.append(int(L[idx])); idx += 1
        heap = newheap; heapq.heapify(heap)
    cont_util = used / (B * cont_iters)

    print(f"static util={static_util * 100:.1f}%  iters={static_iters}")
    print(f"cont   util={cont_util * 100:.1f}%  iters={cont_iters}")
    print(f"fewer iterations = {static_iters / cont_iters:.2f}x")
```

![静态 vs 连续批处理的 GPU 槽位利用率与总迭代数](/assets/posts/scheduling/sim-batching-utilization.png)

一手结果（`mean len ≈ 114`，`batch=32`）：静态批处理槽位利用率只有 **22%**（短请求都在空等批里最慢的），连续批处理达到 **97.6%**；清空同样的工作量，连续批处理只用了 **4.43×** 更少的迭代。这就是 continuous batching 的地基级收益——**把空转的槽位立刻填满**。

### 实验 B：FCFS vs SJF —— 队头阻塞的代价与反噬

在一个略微过载、长尾更重的到达流上（16 个槽位），对比按到达顺序（FCFS）与按短作业优先（SJF）：

```python
def simulate_queue(arrivals, service, order, c):
    """c 个并行槽位（连续批处理），非抢占。order ∈ {'fcfs','sjf'}，返回每个请求的延迟。"""
    n = len(arrivals)
    done = np.zeros(n)
    free_at = [0.0] * c                    # 每个槽位下次空闲的时间
    a_order = np.argsort(arrivals)         # 按到达时间排好序的下标
    pending, ai, finished, t = [], 0, 0, 0.0
    while finished < n:
        slot = int(np.argmin(free_at))     # 最早空出来的槽位
        slot_free = free_at[slot]
        while ai < n and arrivals[a_order[ai]] <= max(slot_free, t):
            pending.append(a_order[ai]); ai += 1        # 已到达的进入候选池
        if not pending:                    # 没人在等就跳到下一个到达时刻
            if ai < n:
                t = arrivals[a_order[ai]]
                pending.append(a_order[ai]); ai += 1
            else:
                break
        start = max(slot_free, min(arrivals[i] for i in pending))
        if order == "sjf":
            pick = min(pending, key=lambda i: service[i])   # 最短作业优先
        else:
            pick = min(pending, key=lambda i: arrivals[i])  # 先来先服务
        pending.remove(pick)
        finish = start + service[pick]
        done[pick] = finish - arrivals[pick]   # 延迟 = 排队 + 服务
        free_at[slot] = finish
        finished += 1
    return done


def exp_b():
    rng = np.random.default_rng(SEED)
    N, c = 3000, 16
    # 比聊天默认更重的长尾，让少数长请求足以堵住大量短请求
    service = np.clip(rng.lognormal(mean=4.4, sigma=1.1, size=N), 4, 4096).astype(float)
    rate = c / service.mean() * 1.15       # 略微过载，队列才会堆积、顺序才重要
    arrivals = np.cumsum(rng.exponential(1.0 / rate, size=N))

    lat_fcfs = simulate_queue(arrivals, service, "fcfs", c)
    lat_sjf = simulate_queue(arrivals, service, "sjf", c)
    for name, lat in [("FCFS", lat_fcfs), ("SJF", lat_sjf)]:
        print(f"{name}: mean={lat.mean():.0f}  p99={np.percentile(lat, 99):.0f}")
```

![FCFS vs SJF 的平均延迟与 p99 延迟](/assets/posts/scheduling/sim-fcfs-vs-sjf.png)

一手结果:SJF 把**平均延迟**从 2091 降到 427——**4.9× 的改善**；但**代价写在同一张图上**：p99 延迟从 4469 涨到 10611（**尾部反而恶化 2.4×**）。这精确复现了第五节的取舍——**SJF 让多数短请求飞快，却把少数长请求推向饥饿。** 生产里要用它，就必须叠加 aging / 公平机制来托住尾部。

### 实验 C：naïve prefill vs chunked prefill —— decode 的稳定性

模拟一条 decode 流，其间不时有新 prompt 需要 prefill。naïve 把整段 prompt 塞进一轮，chunked 则按 token budget（512）切块：

```python
def exp_c():
    rng = np.random.default_rng(SEED)
    n_decode, budget, steps = 32, 512, 4000
    a, b = 0.5, 0.008              # 单轮耗时 ≈ a + b × (本轮 token 数)
    prefill_prob = 0.06           # 每轮约 6% 概率来一个新 prompt 需要 prefill
    prompt_len = lambda: int(np.clip(rng.lognormal(7.0, 0.6), 128, 8192))  # 中位 ≈ 1100

    def run(chunked):
        tbt, pending = [], 0       # pending：还没 prefill 完的 prompt token 数
        for _ in range(steps):
            toks = n_decode        # 本轮先算上所有正在 decode 的请求
            if chunked:            # 切块：每轮只补 (budget - decode) 大小的一块 prefill
                if pending == 0 and rng.random() < prefill_prob:
                    pending = prompt_len()
                if pending > 0:
                    chunk = min(budget - n_decode, pending)
                    toks += chunk; pending -= chunk
            else:                  # naïve：整段 prompt 一次砸进同一轮
                if rng.random() < prefill_prob:
                    toks += prompt_len()
            tbt.append(a + b * toks)   # 这一轮所有 decode 用户都经历的 TBT
        return np.array(tbt)

    for name, tbt in [("naive", run(False)), ("chunked", run(True))]:
        print(f"{name}: p50={np.percentile(tbt, 50):.2f} "
              f"p99={np.percentile(tbt, 99):.2f} max={tbt.max():.2f}")
```

![naïve 混批的 TBT 尖刺 vs chunked prefill 的平稳 TBT](/assets/posts/scheduling/sim-chunked-prefill-tbt.png)

一手结果:两者的 **p50 TBT 一样**（平时没差别），但 naïve 的 **p99 TBT 是 chunked 的 3.7×**，**最糟的一次卡顿更是 8.7×**。左图的红色尖刺就是每次长 prefill 砸下来时，所有 decode 用户一起经历的卡顿；chunked（绿色）则被死死压在预算线以下。**token budget 用一点 TTFT 换来了 TPOT 的稳定。**

## 九、总结与延伸

单引擎调度的主线，就是一条"发现问题 → 引入优化"的链子：

- **批处理在浪费 GPU** → **Continuous Batching（Orca）**：调度粒度降到一次 forward，槽位随空随补。
- **prefill 拖垮 decode** → **Chunked Prefill（Sarathi-Serve）**：用 token budget 把每轮削平，decode 不再卡顿。
- **显存不够** → **抢占**：recompute（费算力）vs swap（费带宽），本质是资源之间的取舍。
- **该先算谁** → **排队策略**：FCFS 简单但队头阻塞；SJF/长度预测降延迟但会饥饿；VTC/aging 找回公平。
- **顺序还决定命中率** → **缓存感知调度**：最长前缀优先，最大化 KV 复用。

把它们拼起来，就是一个"**先保运行、再抢占、再 admit、组混合批**"的迭代循环。

一句话带走：**推理调度就是在固定 GPU-时间和 KV-显存下，反复在"吞吐 / TTFT / TPOT / 公平 / 命中率"这几个互相拉扯的目标之间选点。**

延伸阅读：单机之外，当一个实例扛不住时，就要把 **prefill 和 decode 拆到不同 GPU 池**（DistServe、Splitwise），并让一个**全局调度器**按"KV 缓存在哪、传输多贵、SLO 是否满足"来路由请求（Mooncake 的 KVCache-centric 架构 + early rejection）。这套**分布式调度**建立在 PD 分离这块地基上——所以接下来我会先单独写一篇 **PD 分离**，把它讲透之后，再单独开一篇讲**分布式（集群）调度**。

## 参考

1. Yu et al., [_Orca: A Distributed Serving System for Transformer-Based Generative Models_](https://www.usenix.org/conference/osdi22/presentation/yu)（continuous batching / selective batching），OSDI 2022.
2. Kwon et al., [_Efficient Memory Management for Large Language Model Serving with PagedAttention_](https://arxiv.org/abs/2309.06180)（vLLM，抢占：recompute / swap），SOSP 2023.
3. Agrawal et al., [_Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve_](https://arxiv.org/abs/2403.02310)（chunked prefill / stall-free batching），OSDI 2024.
4. Fu et al., [_Efficient LLM Scheduling by Learning to Rank_](https://arxiv.org/abs/2408.15792)（预测输出长度相对排名近似 SJF/SRTF），NeurIPS 2024.
5. Sheng et al., [_Fairness in Serving Large Language Models_](https://arxiv.org/abs/2401.00588)（VTC，Virtual Token Counter），OSDI 2024.
6. Zheng et al., [_SGLang: Efficient Execution of Structured Language Model Programs_](https://arxiv.org/abs/2312.07104)（RadixAttention / cache-aware scheduling），NeurIPS 2024.
7. Zhong et al., [_DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving_](https://arxiv.org/abs/2401.09670)，OSDI 2024.
8. Patel et al., [_Splitwise: Efficient Generative LLM Inference Using Phase Splitting_](https://arxiv.org/abs/2311.18677)，ISCA 2024.
9. Qin et al., [_Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving_](https://arxiv.org/abs/2407.00079)，FAST 2025.
