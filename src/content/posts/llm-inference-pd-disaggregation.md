---
author: xiangel
pubDatetime: 2026-09-10T01:30:00Z
title: "大模型推理的 PD 分离：把 Prefill 和 Decode 拆到不同 GPU 池"
slug: llm-inference-pd-disaggregation
featured: true
draft: false
tags:
  - 大模型推理系统
  - PD分离
  - LLM
description: 单机再怎么排班，prefill 和 decode 还是挤在同一张卡上。PD 分离把两阶段拆到两个 GPU 池，各自选硬件、调并行度、独立扩缩容——代价是要把 KV Cache 搬过去。用"专业化分工"做类比，讲清为什么拆、怎么搬、什么时候该拆，附一组无需 GPU、可复现的仿真。
---

上一篇[《大模型推理调度（单机篇）》](/posts/llm-inference-scheduling-single-node/)讲到，`chunked prefill` 把长 prompt 切块、和 decode 拼在一轮里，让 decode 不再被长 prefill 卡顿。但如果你盯着它多想一步会发现：**这只是让两个阶段"轮流坐庄"，并没有让它们"互不相干"。**

prefill 和 decode 仍然挤在**同一张 GPU** 上，共享同一套并行度、同一种硬件、同一块显存预算。它们的"性格"却截然相反——硬要用一套配置同时把两个都伺候好，本身就是个妥协。

这是本系列第四篇。[第一篇](/posts/from-causal-lm-to-inference-system/)讲推理系统的模块怎么长出来，[第二篇](/posts/kv-cache-paged-attention-and-prefix-caching/)讲 KV Cache 怎么管，[第三篇](/posts/llm-inference-scheduling-single-node/)讲单机内部怎么调度。这一篇往上再走一步：**当一张卡装不下、也伺候不好所有请求时，把 prefill 和 decode 彻底拆到两个 GPU 池——这就是 PD 分离（Prefill/Decode Disaggregation）。**

> **说明**：本篇只讲 **PD 分离这套机制本身**（为什么拆、怎么搬 KV、什么时候该拆）。至于拆开之后如何在**整个集群**上做全局的缓存感知路由、early rejection，那是**第五篇（分布式调度）**的主题。

全程我用一个贯穿的类比:**专业化分工**——就像数据中心会把"离线批处理集群"和"在线低延迟服务集群"分开，让各自用最合适的机器、各自扩缩容;而 KV Cache 的传输,就是在两个车间之间**搬运半成品的那条传送带**。

## Table of contents

## 一、单机的天花板:轮流,不等于不干扰

先把两个阶段的"性格"摆出来。它们几乎在每个维度上都相反:

- **prefill**:一次并行吞掉整段 prompt(可能几千 token),**算力密集(compute-bound)**,一轮很重。
- **decode**:每步只算一个新 token、但要读取**整个**不断变长的 KV Cache,**访存密集(memory-bound)**,一轮很轻。

![prefill 与 decode 的资源画像相反:一个吃算力、一个吃显存带宽;一次重 prefill 之后跟着很多次轻 decode](/assets/posts/pd-disaggregation/diagram-two-phase-profiles.png)

`chunked prefill` 解决的是**时间上**的冲突(别让长 prefill 独占一轮),但它解决不了两件事:

1. **它们仍在抢同一张卡**。哪怕切了块,prefill 的算力尖峰和 decode 的访存需求还是在同一个 GPU 上此消彼长;高并发下,这种来回切换本身就有开销。
2. **配置被迫二选一**。prefill 想要**大算力、小并行度**;decode 想要**大显存带宽、大并行度(把 KV 摊薄)**。一套 TP、一种卡,注定两头都不最优——要么牺牲 TTFT,要么牺牲 TPOT,要么两边都留余量、硬件利用率上不去。

> **类比**:这就像让**一个师傅**既当"备料工"(prefill:一次性剁一大盆馅,力气活)又当"出餐员"(decode:一份份稳定出餐,讲究节奏)。就算你让他"剁一会儿、出一份"地穿插着干(chunked prefill),他还是一个人、一套工具、在同一个灶台前——**忙起来两头都别扭**。

## 二、第一刀:把两阶段拆到两个池 —— PD 分离

既然两个阶段性格相反、又互相干扰,最直接的办法就是:**别让它们共享 GPU 了,一人一个池。**

- **Prefill 池**:只做 prefill,专心把 prompt 变成 KV Cache。
- **Decode 池**:只做 decode,专心一个 token 一个 token 地稳定吐字。
- 一个请求先在 prefill 池生成 KV,**把 KV 交接给** decode 池,再由后者续写到结束。

![colocated:一张卡两阶段互相干扰、配置被迫二选一;disaggregated:两个池各司其职,中间用一条 KV 传输链路交接](/assets/posts/pd-disaggregation/diagram-colocated-vs-disagg.png)

这就是 **DistServe(OSDI'24)** 和 **Splitwise(ISCA'24,微软)** 的基本盘。DistServe 把它总结成一句话:**colocated 既带来 prefill-decode 干扰,又把两阶段的资源分配和并行度策略耦死了**;拆开之后,**干扰直接消失**。Splitwise 还多加了一个**mixed pool(混合池)**,随负载弹性伸缩,吸收 prefill/decode 比例的临时波动。

一个请求在分离系统里的完整旅程是这样的:

![一个请求的生命周期:client → router/frontend → prefill worker 算 KV → KV 传输 → decode worker 生成 → 流式返回](/assets/posts/pd-disaggregation/diagram-request-lifecycle.png)

> **类比**:把"一个师傅两头忙"改成**开两个车间**——备料车间和出餐车间,各自招各自的人、配各自的设备。互不打扰,各自还能按订单量单独扩招。

## 三、红利一:阶段专属优化

拆开之后,最大的红利不是"少了干扰"这么简单,而是**每个池都能被单独打磨到极致**:

![prefill 池用小 TP、算力型 GPU;decode 池用大 TP、带宽型 GPU;两池按流量独立选择实例数量(图中 P:D = 2:3)](/assets/posts/pd-disaggregation/diagram-phase-specific.png)

1. **各自的并行度**。prefill 算力密集,用**小 TP**就能喂饱;decode 访存密集,用**大 TP**把 KV Cache 摊到更多卡上、拿到更多聚合显存带宽。Dynamo 的文档说得很直白:decode 用大 TP、prefill 用小 TP,两阶段都能算得更高效。
2. **各自的硬件**。Splitwise 的核心洞察之一:decode**用不上**最新最贵 GPU 的澎湃算力,完全可以放在**更便宜、更省电**的卡上;把最强的卡留给 prefill。于是**同样的吞吐,成本降 20%**,或者**同样的成本和功耗,吞吐提到 2.35×**。
3. **各自的扩缩容**。线上流量的 prefill:decode 比例随时在变(长文档场景 prefill 重、闲聊场景 decode 重)。两个池**独立伸缩**,按 P:D 比例各自加减实例,不用再为了迁就一头而整体扩容。

> **类比**:备料车间配大功率绞肉机(算力型硬件),出餐窗口多开几个、配保温台(带宽型硬件);中午备料忙就给备料车间加人,晚上出餐忙就给窗口加人——**各扩各的**。

## 四、新问题:KV Cache 得搬过去

天下没有免费的分工。拆开的代价是引入了一条原本不存在的链路:**prefill 池算出的 KV Cache,必须搬到 decode 池去。** 而 KV 可能有**好几个 GB**(长上下文尤甚)。**这条传输链路能不能喂饱 decode,直接决定整个架构是赚还是赔。**

第一个问题是**怎么搬得快**——KV 走什么线。互连带宽差了几个数量级:

![互连层级:NVLink(节点内,~100s GB/s)> InfiniBand/RoCE·RDMA(~10s GB/s)> PCIe(~10 GB/s)> TCP(~1 GB/s);越快的线,交接越便宜。右侧:点对点 VRAM→VRAM vs 池化 KV Store](/assets/posts/pd-disaggregation/diagram-interconnect.png)

- **点对点(P2P)**:prefill 的显存直接搬到 decode 的显存。DistServe 用**带宽感知放置**,尽量让配对的两阶段落在**同节点、走 NVLink**;NVIDIA **Dynamo** 用 **NIXL** 做**非阻塞**的 GPU→GPU 直传(传输时 GPU 还能继续算别的);vLLM 的 **NixlConnector** 也是这条路(默认 UCX,走 RDMA/InfiniBand)。
- **池化(pooled)**:把 KV 存进一个横跨 CPU/DRAM/SSD 的**分布式 KV 池**,谁要谁取。**Mooncake(FAST'25 最佳论文,Kimi 的服务平台)** 就是这套 **KVCache-centric** 思路,顺便把 GPU 集群里闲置的 CPU/DRAM/SSD 利用起来当 KV 池。

第二个问题是**怎么把搬的延迟藏起来**。最朴素的**串行传输**:等 prefill 把所有层都算完,**再**整体传 KV——decode 只能干等,白白多出一段 TTFT。聪明的做法是 **layer-wise / chunked 传输**:某一层的 KV 一算完,**立刻**开始传,同时去算下一层——传输**藏在计算背后**,等 prefill 算完时,KV 也基本传完了。

![串行传输:算完所有层再传,decode 干等;layer-wise 传输:每层算完即传、与下一层计算重叠,交接几乎免费](/assets/posts/pd-disaggregation/diagram-kv-transfer.png)

Splitwise 就是这么做的:**小 prompt 用串行传输**(反正也不大),**大 prompt 用 layer-wise 传输**把延迟藏进计算里,用户几乎无感。

> **类比**:传送带就是互连(NVLink 是高速传送带,TCP 是老式手推车)。串行传输 = 把整盆馅全剁完才推给窗口;layer-wise = 剁好一部分就先递一part过去,等你剁完,前面的也送到了——**搬运的时间被剁馅的时间盖住了**。

## 五、用什么度量:goodput,而不是吞吐

拆不拆、拆成什么样,得有个标尺。这里有个容易踩的坑:**光看吞吐(throughput)是会骗人的。**

一个请求哪怕被"服务"了,但如果它的 TTFT 或 TPOT 违反了 SLO(首字等太久、吐字太卡),对用户来说**这次服务就是废的**。DistServe 因此主张用 **goodput** 来度量:**在同时满足 TTFT 和 TPOT 两个 SLO 的前提下,每张 GPU 能扛住的最大请求率。**

![goodput vs 负载:colocated 因干扰很早就到达 SLO 拐点、goodput 崩塌;disaggregated 把拐点推得更远,能在 SLO 内扛住更高负载](/assets/posts/pd-disaggregation/diagram-goodput.png)

colocated 因为有干扰,负载一高,TPOT 就开始违反 SLO,**有效**吞吐(goodput)很早就见顶甚至崩塌;disaggregated 消除了干扰,**把 SLO 拐点往右推**,同样的卡能在 SLO 内扛住更高的负载。DistServe 报告在满足 SLO 的前提下,相比 colocated 能服务 **7.4× 的请求量**,或在同样负载下满足 **12.6× 更紧的 SLO**(且 >90% 请求达标)。

> **类比**:一家餐厅一小时"接待"了 500 桌,听着很猛;但如果一半人等太久摔门而去,**真正吃上饭的**才是有效产出。goodput 数的就是"真正吃上饭的桌数"。

## 六、到底该不该拆:一条盈亏平衡线

PD 分离不是免费午餐,也不是万能药。它需要**两个池、一个 router、一条 KV 传输链路**,运维复杂度陡增。什么时候值得?一句话:**当"省下的干扰" > "多花的传输"时。**

- **拆得值**:prompt 长、并发高、SLO 严,**而且已经有 RDMA/NVLink 高速互连**。此时干扰严重、分离收益大,传输又便宜。业界经验值:full disaggregation 约 **1.5–2.5×** 吞吐。
- **拆了亏**:prompt 短、负载低,或者**只有普通网络(TCP)**。传输开销会吃掉收益,甚至**跑不过一台调好的 colocated + chunked prefill**(后者约 20–40% 收益、零新基建)。

所以正确的路径是:**先把 chunked prefill 用满,不够了再上 PD 分离。** NVIDIA 官方也反复强调:最优工作点取决于**模型、GPU、后端、请求长度分布**,没有一个固定阈值。下面的实验 B 会把这条盈亏平衡线画出来。

## 七、生产全景:谁在用,怎么用

PD 分离已经是 2025 年主流推理栈的标配能力,几套系统各有侧重:

- **DistServe**:学术奠基,提出 goodput 目标 + 两阶段独立协同优化 + 带宽感知放置。
- **Splitwise**:强调**异构硬件分工**与三池(prompt/token/mixed)结构,MSCCL++ 走 InfiniBand。
- **Mooncake**:把它推到**生产超大规模**——KVCache-centric 分离 + 分布式 KV 池,支撑 Kimi 日处理 1000 亿+ token,真实 trace 上 **+59%~498%** 有效容量。
- **vLLM**:`--kv-transfer-config` 配 `kv_producer`/`kv_consumer` 两类实例,`NixlConnector` / `MooncakeConnector` 负责 KV 传输,支持 **xPyD**(x 个 prefill、y 个 decode)。
- **NVIDIA Dynamo**:`PrefillRouter` + 全局 prefill 队列(NATS)+ NIXL 直传,**运行时可重配** xPyD;还会**动态决定**某个请求的 prefill 是本地算还是甩给远端 prefill 池(按 prefill 长度和队列状态)。

> 注意:这些系统里的**全局路由、缓存命中调度、early rejection**——比如 Mooncake 的 Conductor 全局调度器——严格说属于**集群级调度**,是第五篇的主角。本篇到"两个池 + 一条传输链路"为止。

## 八、动手实验:无需 GPU 的 PD 分离仿真

老规矩,用纯 Python 仿真**隔离**地验证前面几个结论。每个实验只放大**一个**效应(因此不是端到端基准,倍数不能直接等同真机)。全部 `SEED=42` 可复现,无需 GPU / 模型 / 联网。

公共前置(三个实验共用;下面每段只贴核心逻辑,`matplotlib` 绘图从略):

```python
import heapq
import numpy as np

SEED = 42
A_ITER, B_ITER = 0.5, 0.005                 # 单轮耗时 ≈ A_ITER + B_ITER × 本轮 token 数 (ms)
KV_BYTES_PER_TOKEN = 2 * 80 * 8 * 128 * 2   # ~320 KiB/token（70B 级、GQA：2×层×KV头×head维×dtype字节）

def prompt_len(rng):  # 输入长度：多数中等、长尾（中位 ~735）
    return int(np.clip(rng.lognormal(6.6, 0.5), 64, 8000))

def output_len(rng):  # 输出长度：多数短、少数长（中位 ~120）
    return int(np.clip(rng.lognormal(4.8, 0.7), 4, 1000))
```

### 实验 A:colocated vs disaggregated —— goodput vs 负载

同一条到达流,对比两种部署:colocated 把 prefill chunk 混进 decode 轮次(互相干扰),disaggregated 的 decode 引擎**只做 decode**、prefill 交给独立的 prefill 服务器 + KV 传输。度量:**同时满足 TTFT 与 TPOT 两个 SLO** 的有效请求率(goodput)。

```python
def simulate(rate, disagg, horizon_ms=30000):
    rng = np.random.default_rng(SEED)
    M, budget = 48, 1024              # 引擎最大并发（KV 上限）、每轮 token 预算
    slo_tpot, slo_ttft = 1.6, 250.0  # 每 token 延迟 / 首字延迟 的 SLO
    prefill_tok_per_ms = 220.0       # disagg：独立 prefill 服务器的吞吐
    bw = 150e9 / 1e3                 # disagg：KV 传输带宽（NVLink 级 ~150 GB/s，bytes/ms）
    clock, prefill_free = 0.0, 0.0
    nxt = rng.exponential(1.0 / rate)
    waiting, ready, running, done = [], [], [], []   # ready: disagg 的 (就绪时刻, req) 小顶堆
    seq = 0
    while clock < horizon_ms:
        while nxt <= clock:                       # 到达
            P, O = prompt_len(rng), output_len(rng)
            r = {"arr": nxt, "P": P, "out": O, "pre": P, "ts": 0.0, "tn": 0, "ttft": None}
            if disagg:                            # 独立 prefill 服务器（串行）+ KV 传输 → 决定 TTFT
                start = max(prefill_free, nxt); prefill_free = start + P / prefill_tok_per_ms
                xfer = KV_BYTES_PER_TOKEN * P / bw
                r["ttft"] = prefill_free + xfer - nxt
                seq += 1; heapq.heappush(ready, (prefill_free + xfer, seq, r))
            else:
                waiting.append(r)
            nxt += rng.exponential(1.0 / rate)
        if disagg:                                # KV 到齐、有空位 → 进 decode 引擎
            while ready and ready[0][0] <= clock and len(running) < M:
                running.append(heapq.heappop(ready)[2])
        decode_tokens, prefill_tokens = len(running), 0
        if not disagg and waiting and len(running) < M:   # colocated：预算里塞一块 prefill
            head = waiting[0]; room = budget - decode_tokens
            if room > 0:
                chunk = min(room, head["pre"]); head["pre"] -= chunk; prefill_tokens = chunk
                if head["pre"] == 0:
                    head["ttft"] = clock - head["arr"]; running.append(waiting.pop(0))
        dt = A_ITER + B_ITER * (decode_tokens + prefill_tokens)   # 关键差别就在这一行的 token 数
        clock += dt
        still = []
        for r in running:                         # 给每个在 decode 的请求记一次 TPOT，并推进一步
            r["ts"] += dt; r["tn"] += 1; r["out"] -= 1
            (done if r["out"] <= 0 else still).append(r)
        running = still
    tpot = np.array([r["ts"] / r["tn"] for r in done])
    ttft = np.array([r["ttft"] if r["ttft"] is not None else 1e9 for r in done])
    good = (tpot <= slo_tpot) & (ttft <= slo_ttft)
    return good.sum() / (clock / 1000.0)          # 每秒 good 请求数（goodput）
```

（完整脚本还加了过载时的拒绝上限与队列压缩等工程细节，只影响性能不影响结论。）

![colocated 因干扰很早见顶、goodput 崩塌;disaggregated 把拐点推得更远、峰值更高](/assets/posts/pd-disaggregation/sim-goodput-vs-load.png)

一手结果:colocated 的峰值 goodput 只有 **88 req/s**(SLO 拐点约在 127 req/s 的负载),disaggregated 达到 **200 req/s**(拐点推到约 205 req/s),**峰值 goodput 2.27×**。负载再往上,colocated 因干扰迅速崩塌,disaggregated 也会过载,但整条曲线**明显更抗压**——这就是"消除干扰、把 SLO 拐点往右推"的直接体现。

### 实验 B:到底该不该拆 —— 盈亏平衡线(本篇的灵魂图)

分离**省下**的是 colocated 里 prefill 对 decode 的干扰(随负载线性增长);**多花**的是每个请求的 KV 传输(取决于互连带宽)。把二者相减,扫一遍(负载 × 带宽),就能看出**什么时候拆才划算**。

```python
def when_to_disaggregate():
    rng = np.random.default_rng(SEED)
    avg_P = int(np.mean([prompt_len(rng) for _ in range(20000)]))   # 平均 prompt 长度
    compute_ms_per_token = 0.011        # colocated 里，每个 prompt token 的 prefill 会挤占 decode 的量
    for bw in np.array([1, 5, 12.5, 40, 150, 400]) * 1e9:           # bytes/s：TCP..PCIe..IB..NVLink
        transfer_ms = KV_BYTES_PER_TOKEN * avg_P / bw * 1e3         # 每请求 KV 传输耗时
        net = avg_P * compute_ms_per_token - transfer_ms           # >0：分离净赚（省的干扰 > 多花的传输）
        print(f"bw={bw/1e9:6.1f} GB/s  transfer={transfer_ms:6.2f} ms/req  "
              f"net/req={net:+.2f} ms -> {'disagg 胜' if net > 0 else 'colo 胜'}")
    # 盈亏平衡带宽（净收益=0，与负载无关，因为两项都随负载线性）
    bw_star = KV_BYTES_PER_TOKEN * 1e3 / compute_ms_per_token / 1e9
    print(f"break-even ≈ {bw_star:.1f} GB/s")
```

![负载 × 互连带宽的净收益地图:绿区(高带宽)分离净赚,红区(低带宽)colocated 更优,盈亏平衡线约在 30 GB/s](/assets/posts/pd-disaggregation/sim-when-to-disaggregate.png)

一手结果(平均 prompt ≈ 835 token、KV ≈ 320 KiB/token):**盈亏平衡互连约在 30 GB/s**。在 1 / 5 / 12.5 GB/s(TCP、慢 PCIe)下,每请求净收益是 **−264 / −46 / −13 ms**——**colocated 反而更好**;到 40 / 150 / 400 GB/s(InfiniBand、NVLink)下,净收益转正为 **+2.3 / +7.4 / +8.5 ms**——**分离才开始赚**。这精确复现了那句话:**没有高速互连,PD 分离可能跑不过一台调好的单机。**

### 实验 C:layer-wise 传输 —— 把交接延迟藏起来

分离的传输开销并非只能硬扛。对比两种传法:**串行**(prefill 全算完再传整份 KV)与 **layer-wise**(每层算完即传、与后续层计算重叠)。

```python
def layerwise_vs_serialized(bw=40e9, L=80, prefill_tok_per_ms=220.0):
    prompt = np.arange(256, 8001, 64)
    transfer_ms = KV_BYTES_PER_TOKEN * prompt / bw * 1e3        # 传完整份 KV 的耗时
    prefill_ms = prompt / prefill_tok_per_ms                    # prefill 计算耗时
    serialized = transfer_ms                                    # 串行：整份传输全暴露在 TTFT 上
    # layer-wise：传输藏进 L-1 层的计算窗口里，只暴露超出计算的那部分（至少一层的分片）
    layerwise = np.maximum(transfer_ms / L, transfer_ms - prefill_ms * (L - 1) / L)
    for P in (1024, 4096, 8000):
        i = int(np.argmin(np.abs(prompt - P)))
        print(f"P={P:5d}  serialized={serialized[i]:6.2f} ms  "
              f"layer-wise={layerwise[i]:6.2f} ms  hidden={100*(1-layerwise[i]/serialized[i]):.0f}%")
```

![串行传输的 TTFT 开销随 prompt 线性上升;layer-wise 把大部分藏进计算,开销显著更低](/assets/posts/pd-disaggregation/sim-layerwise-transfer.png)

一手结果(InfiniBand ≈ 40 GB/s、80 层):layer-wise 把交接开销**藏掉约 55%**。P=8000 时,串行要给 TTFT 多加 **65.5 ms**,layer-wise 只剩 **29.6 ms**;P=1024 时是 **8.4 ms → 3.8 ms**。换句话说,**重叠传输把"搬 KV"这件事的大半成本盖进了 prefill 计算里**——这也是实验 B 里盈亏平衡线能往左压的关键工程手段。

## 九、总结与延伸

PD 分离的主线,还是那条"发现问题 → 引入优化 → 带出新问题"的链子:

- **单机排班治标不治本** → **PD 分离**:prefill、decode 各一个池,干扰彻底消失。
- **红利:阶段专属优化** → 各自的并行度、各自的硬件、各自的扩缩容(DistServe / Splitwise)。
- **新代价:KV 得搬过去** → 互连带宽是命脉;**layer-wise 传输**把延迟藏进计算;点对点 vs 池化 KV Store(Mooncake)。
- **正确的标尺:goodput** → 在双 SLO 内的有效吞吐,而不是原始吞吐。
- **该不该拆:一条盈亏平衡线** → 长 prompt + 高并发 + 快互连才划算;否则先用满 chunked prefill。

一句话带走:**PD 分离是用"一次 KV 搬运的成本",换来"两个阶段各自最优 + 互不干扰"——这笔账只有在互连够快、负载够重、prompt 够长时才划算。**

延伸阅读(也是**第五篇**的主题):当 prefill 池、decode 池、KV 池散布在**整个集群**上,谁来决定一个请求去哪台 prefill、哪台 decode?怎么让路由**感知 KV 缓存在哪**(命中就省一次 prefill)、怎么在过载时**提前拒绝**注定超时的请求?这就是 **Mooncake Conductor** 那套 **KVCache-centric 全局调度**要回答的——下一篇见。

## 参考

1. Zhong et al., [_DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving_](https://arxiv.org/abs/2401.09670)（goodput 目标 + 带宽感知放置），OSDI 2024.
2. Patel et al., [_Splitwise: Efficient Generative LLM Inference Using Phase Splitting_](https://arxiv.org/abs/2311.18677)（异构硬件分工 + 三池 + layer-wise 传输），ISCA 2024.
3. Qin et al., [_Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving_](https://arxiv.org/abs/2407.00079)（分布式 KV 池 + Conductor 全局调度），FAST 2025.
4. Agrawal et al., [_Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve_](https://arxiv.org/abs/2403.02310)（chunked prefill，单机对照基线），OSDI 2024.
5. vLLM 文档，[_Disaggregated Prefilling_](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)（`kv_producer`/`kv_consumer`、NixlConnector / MooncakeConnector、xPyD）.
6. NVIDIA, [_Dynamo_](https://github.com/ai-dynamo/dynamo)（PrefillRouter + 全局 prefill 队列 + 运行时可重配 xPyD）；配套介绍见 [NVIDIA 技术博客](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/).
7. NVIDIA, [_NIXL: NVIDIA Inference Xfer Library_](https://github.com/ai-dynamo/nixl)（非阻塞点对点 KV 传输，RDMA/IB/UCX/NVMe/S3 后端）.
