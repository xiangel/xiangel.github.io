#!/usr/bin/env python3
"""无需 GPU 的分布式（集群级）调度仿真，三个实验，SEED=42 可复现。

A: round-robin vs cache-aware routing —— 集群 KV 命中率（随 worker 数）
B: 亲和 vs 均衡 vs 混合（E2/pow2）—— 命中率与负载不均的取舍（帕累托）
C: accept-all vs early rejection —— 过载下的 goodput 与无效计算

图表用 diagram-design 调色板，与本系列其它图保持一致。
输出 PNG 到 public/assets/posts/distributed-scheduling/。
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SEED = 42
OUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "public", "assets", "posts", "distributed-scheduling"
)
OUT = os.path.abspath(OUT)

# diagram-design palette
PAPER = "#f5f5f5"
INK = "#2d3142"
MUTED = "#4f5d75"
SOFT = "#7a8399"
ACCENT = "#eb6c36"
RULE = "#d7d9de"

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": RULE,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)


# ----------------------------------------------------------------------------
# 公共前置：带共享前缀的请求流
# ----------------------------------------------------------------------------
def make_requests(n, n_groups, rng, zipf_a=1.1):
    """每个请求属于某个"前缀组"（共享 system prompt / 文档），组按 zipf 分布，
    少数热组占大多数流量。prefix_len 是组共享的可复用前缀长度，suffix_len 唯一。"""
    ranks = np.arange(1, n_groups + 1)
    p = 1.0 / ranks**zipf_a
    p /= p.sum()
    groups = rng.choice(n_groups, size=n, p=p)
    # 每个组一个前缀长度（长系统提示/文档），多为中长
    group_prefix = np.clip(rng.lognormal(6.7, 0.5, size=n_groups), 64, 8000).astype(int)
    prefix_len = group_prefix[groups]
    suffix_len = np.clip(rng.lognormal(4.5, 0.7, size=n), 8, 2000).astype(int)  # 唯一后缀
    return groups, prefix_len, suffix_len


# ----------------------------------------------------------------------------
# 实验 A：round-robin vs cache-aware —— 集群命中率随 worker 数
# ----------------------------------------------------------------------------
class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.d = {}  # group -> last_use（用插入顺序近似 LRU）

    def has(self, g):
        return g in self.d

    def touch(self, g, t):
        if g in self.d:
            del self.d[g]
        self.d[g] = t
        if len(self.d) > self.cap:
            # 淘汰最久未用
            oldest = next(iter(self.d))
            del self.d[oldest]


def exp_a_once(W, n=20000, n_groups=800, cap_groups=24):
    rng = np.random.default_rng(SEED)
    groups, prefix_len, suffix_len = make_requests(n, n_groups, rng)
    results = {}
    for policy in ("rr", "cache"):
        caches = [LRU(cap_groups) for _ in range(W)]
        load = np.zeros(W)  # 累计 token 负载（用于 cache 策略的"最空"回退）
        hit_tok, tot_tok = 0, 0
        for i in range(n):
            g, pl = int(groups[i]), int(prefix_len[i])
            if policy == "rr":
                w = i % W
            else:  # cache-aware：命中就发给它，否则发给负载最轻的
                owners = [k for k in range(W) if caches[k].has(g)]
                w = min(owners, key=lambda k: load[k]) if owners else int(np.argmin(load))
            if caches[w].has(g):
                hit_tok += pl  # 前缀命中，省下这段 prefill
            tot_tok += pl
            caches[w].touch(g, i)
            load[w] += pl + int(suffix_len[i])
        results[policy] = hit_tok / tot_tok
    return results["rr"], results["cache"]


def exp_a():
    Ws = [2, 4, 8, 16, 32]
    rr, ca = [], []
    for W in Ws:
        a, b = exp_a_once(W)
        rr.append(a * 100)
        ca.append(b * 100)
    print("[A] cluster prefix hit rate (%)")
    for i, W in enumerate(Ws):
        print(f"    W={W:2d}  round-robin={rr[i]:5.1f}  cache-aware={ca[i]:5.1f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(Ws, ca, "-o", color=ACCENT, lw=2.4, label="cache-aware routing")
    ax.plot(Ws, rr, "-o", color=MUTED, lw=2.0, label="round-robin")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ws)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("number of workers")
    ax.set_ylabel("cluster prefix-cache hit rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Cache-aware routing keeps hit rate high as the cluster grows",
                 color=INK, fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sim-routing-hitrate.png"), dpi=150)
    plt.close(fig)
    return Ws, rr, ca


# ----------------------------------------------------------------------------
# 实验 B：亲和 vs 均衡 vs 混合 —— 命中率与负载不均的取舍
# ----------------------------------------------------------------------------
def exp_b(W=8, n=20000, n_groups=120, cap_groups=40):
    rng = np.random.default_rng(SEED)
    groups, prefix_len, suffix_len = make_requests(n, n_groups, rng, zipf_a=1.4)

    def run(policy):
        caches = [LRU(cap_groups) for _ in range(W)]
        load = np.zeros(W)
        hit_tok, tot_tok = 0, 0
        rng2 = np.random.default_rng(SEED + 1)  # pow2 的随机采样
        for i in range(n):
            g, pl, sl = int(groups[i]), int(prefix_len[i]), int(suffix_len[i])
            owners = [k for k in range(W) if caches[k].has(g)]
            if policy == "affinity":  # 一致性哈希：同组永远同一台
                w = g % W
            elif policy == "balance":  # power-of-two-choices：采两台挑轻的
                a, b = rng2.integers(0, W, size=2)
                w = int(a) if load[a] <= load[b] else int(b)
            elif policy == "e2":  # Preble E2：省下的重算 > 新算才亲和，否则去最轻
                if owners and pl > sl:
                    w = min(owners, key=lambda k: load[k])
                else:
                    w = int(np.argmin(load))
            elif policy == "sglang":  # 命中就亲和，但热点(负载>1.5×均值)时切最轻
                if owners and load[min(owners, key=lambda k: load[k])] <= 1.5 * (load.mean() + 1):
                    w = min(owners, key=lambda k: load[k])
                else:
                    w = int(np.argmin(load))
            if caches[w].has(g):
                hit_tok += pl
            tot_tok += pl
            caches[w].touch(g, i)
            load[w] += pl + sl
        imbalance = load.max() / load.mean()
        return hit_tok / tot_tok * 100, imbalance

    names = {"affinity": "pure affinity", "balance": "power-of-two (balance)",
             "e2": "E2 (Preble)", "sglang": "cache-aware + balance (SGLang)"}
    res = {}
    print("[B] hit rate vs load imbalance")
    for pol in ("affinity", "balance", "e2", "sglang"):
        h, imb = run(pol)
        res[pol] = (h, imb)
        print(f"    {names[pol]:32s} hit={h:5.1f}%  imbalance(max/mean)={imb:.2f}x")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"affinity": SOFT, "balance": MUTED, "e2": INK, "sglang": ACCENT}
    # 手工微调各点标签偏移，避免高命中率的三点标签互相叠压
    label_off = {
        "affinity": (-8, 10, "right"),
        "balance": (10, 4, "left"),
        "e2": (6, -20, "left"),
        "sglang": (0, -22, "center"),
    }
    for pol in ("affinity", "balance", "e2", "sglang"):
        h, imb = res[pol]
        focal = pol == "sglang"
        ax.scatter(imb, h, s=240 if focal else 150, color=colors[pol],
                   zorder=3, edgecolors=PAPER, linewidths=1.5)
        dx, dy, ha = label_off[pol]
        ax.annotate(names[pol], (imb, h), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, color=INK, fontsize=9.5)
    ax.set_xlabel("load imbalance  (max worker load / mean)  →  worse")
    ax.set_ylabel("cluster prefix-cache hit rate (%)  →  better")
    ax.set_title("Affinity vs balance: the sweet spot is a blend",
                 color=INK, fontsize=12, pad=14)
    ax.set_ylim(78, 104)
    ax.set_xlim(0.9, 2.15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sim-affinity-vs-load.png"), dpi=150)
    plt.close(fig)
    return res


# ----------------------------------------------------------------------------
# 实验 C：accept-all vs early rejection —— 过载下的 goodput 与无效计算
# ----------------------------------------------------------------------------
def exp_c(horizon_ms=60000):
    M = 32  # decode 并发槽位（整个 decode 池）
    prefill_tok_per_ms = 600.0  # prefill 池吞吐
    slo_ttft = 2000.0  # 首字 SLO（ms），含排队 + prefill + 等 decode 槽位
    a_iter, b_iter = 0.4, 0.006  # decode 单轮耗时 ≈ a + b×并发
    # decode 可持续吞吐 ≈ M / (avg_out × 单轮耗时)；把到达率设到它的 ~1.4×，制造过载
    rate = 0.44  # 到达率（req/ms）

    def simulate(early):
        rng = np.random.default_rng(SEED)  # 两种策略同一条到达流
        clk = 0.0
        nxt = rng.exponential(1.0 / rate)
        prefill_free = 0.0
        pend = []  # prefill 完成、等 decode 槽位：[ready_ms, arrival_ms, ptok]
        running = []  # decode 中：[剩余步数, ok, ptok]
        good, waste_tok, arrivals, rejected = 0, 0, 0, 0
        while clk < horizon_ms:
            while nxt <= clk:
                arrivals += 1
                P = int(np.clip(rng.lognormal(6.6, 0.5), 64, 8000))
                inflight = len(running) + len(pend)
                if early and inflight >= M:  # 准入：只在有余量时才接，否则 prefill 前就拒
                    rejected += 1
                    nxt += rng.exponential(1.0 / rate)
                    continue
                start = max(prefill_free, nxt)  # 接受 → 占用 prefill 池
                prefill_free = start + P / prefill_tok_per_ms
                pend.append([prefill_free, nxt, P])
                nxt += rng.exponential(1.0 / rate)
            pend.sort(key=lambda r: r[0])
            while pend and pend[0][0] <= clk and len(running) < M:
                ready, arr, ptok = pend.pop(0)
                out = int(np.clip(rng.lognormal(4.8, 0.7), 4, 1000))
                ttft = clk - arr  # 首字延迟 = prefill 排队 + prefill + 等 decode 槽位
                ok = ttft <= slo_ttft
                if not ok:
                    waste_tok += ptok  # 已违反 TTFT：这次 prefill 白算了
                running.append([out, ok, ptok])
            dt = a_iter + b_iter * len(running)
            clk += dt
            still = []
            for r in running:
                r[0] -= 1
                if r[0] <= 0:
                    if r[1]:
                        good += 1
                else:
                    still.append(r)
            running = still
        for ready, arr, ptok in pend:  # 还堵在 prefill 队列、始终没进 decode 的，也白花了
            waste_tok += ptok
        return good / (clk / 1000.0), waste_tok, arrivals, rejected

    g0, w0, arr0, rj0 = simulate(early=False)
    g1, w1, arr1, rj1 = simulate(early=True)
    print("[C] overload: accept-all vs early rejection")
    print(f"    accept-all    goodput={g0:6.1f} req/s  wasted prefill={w0/1e6:6.2f} M tokens  rejected={rj0}")
    print(f"    early-reject  goodput={g1:6.1f} req/s  wasted prefill={w1/1e6:6.2f} M tokens  rejected={rj1}")
    print(f"    goodput {g1/max(g0,1e-9):.2f}x, wasted compute {(1 - w1/max(w0,1))*100:.0f}% lower")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.2))
    labels = ["accept-all", "early-reject"]
    ax1.bar(labels, [g0, g1], color=[MUTED, ACCENT], width=0.6)
    ax1.set_ylabel("goodput (SLO-meeting req/s)")
    ax1.set_title("Goodput under overload", color=INK, fontsize=11)
    ax2.bar(labels, [w0 / 1e6, w1 / 1e6], color=[MUTED, ACCENT], width=0.6)
    ax2.set_ylabel("wasted prefill (M tokens)")
    ax2.set_title("Compute wasted on doomed requests", color=INK, fontsize=11)
    for ax in (ax1, ax2):
        ax.grid(axis="x", visible=False)
    fig.suptitle("Early rejection: spend compute only on requests that can meet SLO",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "sim-early-rejection.png"), dpi=150)
    plt.close(fig)
    return (g0, w0), (g1, w1)


if __name__ == "__main__":
    import matplotlib.ticker  # noqa
    exp_a()
    print()
    exp_b()
    print()
    exp_c()
    print(f"\ncharts -> {OUT}")
