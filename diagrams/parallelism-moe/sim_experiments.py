#!/usr/bin/env python3
"""无需 GPU 的并行策略 / MoE 仿真，四个实验，SEED=42 可复现。

A: TP 一层墙钟 = 计算/tp + ring all-reduce（NVLink vs IB）
B: PP 气泡利用率 = m / (m + p - 1)，小 m / 大 p 时崩掉
C: zipf 专家路由 vs EPLB 冗余副本 —— 负载不均与 makespan
D: 长序列下 TP 激活同步 vs Ring CP / Ulysses 通信载荷

图表用 diagram-design 调色板。输出 PNG 到
public/assets/posts/llm-inference-parallelism-moe/。
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SEED = 42
OUT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "public",
    "assets",
    "posts",
    "llm-inference-parallelism-moe",
)
OUT = os.path.abspath(OUT)

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

for _fp in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
):
    if os.path.exists(_fp):
        try:
            font_manager.fontManager.addfont(_fp)
        except Exception:
            pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def ring_allreduce_ms(nbytes, tp, bw_GBps, hop_lat_us):
    """Ring all-reduce：2*(tp-1) 跳，每跳发送 nbytes/tp。"""
    if tp <= 1:
        return 0.0
    hops = 2 * (tp - 1)
    chunk = nbytes / tp
    bw_B_per_ms = bw_GBps * 1e6  # GB/s → B/ms
    return hops * hop_lat_us / 1000.0 + hops * chunk / bw_B_per_ms


# ----------------------------------------------------------------------------
# 实验 A：TP 计算 vs all-reduce，NVLink vs IB
# ----------------------------------------------------------------------------
def exp_a():
    tps = [1, 2, 4, 8, 16]
    hidden, dtype = 8192, 2
    n_allreduce = 2  # 一层：attention out + MLP out
    # Prefill：长序列，算力主导；Decode：小消息，延迟更刺
    scenarios = {
        "prefill": dict(tokens=4096, compute_tp1=6.0, title="Prefill（4096 token）"),
        "decode": dict(tokens=64, compute_tp1=0.55, title="Decode（64 并发）"),
    }
    fabrics = {
        "NVLink": dict(bw=400.0, lat_us=2.0, color=ACCENT),
        "IB": dict(bw=50.0, lat_us=10.0, color=MUTED),
    }

    print("[A] one decoder layer wall-clock (ms)")
    results = {}
    for name, sc in scenarios.items():
        nbytes = hidden * sc["tokens"] * dtype
        results[name] = {}
        for fab, spec in fabrics.items():
            totals, comps, comms = [], [], []
            for tp in tps:
                compute = sc["compute_tp1"] / tp
                comm = n_allreduce * ring_allreduce_ms(nbytes, tp, spec["bw"], spec["lat_us"])
                totals.append(compute + comm)
                comps.append(compute)
                comms.append(comm)
            results[name][fab] = (totals, comps, comms)
            print(f"    {name:7s} {fab:6s}", " ".join(f"TP{tp}={t:.3f}ms" for tp, t in zip(tps, totals)))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=False)
    for ax, (name, sc) in zip(axes, scenarios.items()):
        for fab, spec in fabrics.items():
            totals = results[name][fab][0]
            ax.plot(tps, totals, "-o", color=spec["color"], lw=2.2, label=fab)
        ax.set_xscale("log", base=2)
        ax.set_xticks(tps)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("张量并行度 TP")
        ax.set_title(sc["title"], color=INK, fontsize=12)
        ax.legend(frameon=False)
    axes[0].set_ylabel("一层 decoder 墙钟 (ms)")
    fig.suptitle("节点内 NVLink 上 TP 仍在加速；跨节点 IB 上通信很快反超", color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "sim-tp-compute-vs-comm.png"), dpi=150)
    plt.close(fig)
    return results


# ----------------------------------------------------------------------------
# 实验 B：PP 气泡
# ----------------------------------------------------------------------------
def exp_b():
    def util(p, m):
        return m / (m + p - 1) * 100.0

    ps = [2, 4, 8, 16]
    ms = [1, 2, 4, 8, 16, 32]
    print("[B] pipeline utilization (%) = m / (m + p - 1)")
    grid = {}
    for p in ps:
        grid[p] = [util(p, m) for m in ms]
        print(f"    p={p:2d}", " ".join(f"m={m}:{u:5.1f}%" for m, u in zip(ms, grid[p])))

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = {2: SOFT, 4: MUTED, 8: INK, 16: ACCENT}
    for p in ps:
        ax.plot(ms, grid[p], "-o", color=colors[p], lw=2.2, label=f"PP={p}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("微批次数量 m")
    ax.set_ylabel("理想利用率 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("batch=1（m=1）时，PP 几乎退化成串行", color=INK, fontsize=12)
    ax.legend(frameon=False, title="流水线阶段")
    ax.axhline(50, color=RULE, lw=1.0, ls="--")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sim-pp-bubble.png"), dpi=150)
    plt.close(fig)
    return grid


# ----------------------------------------------------------------------------
# 实验 C：zipf 专家路由 vs EPLB
# ----------------------------------------------------------------------------
def exp_c(n_experts=256, n_gpus=32, top_k=8, n_tokens=80000, n_redundant=32, zipf_a=1.15):
    rng = np.random.default_rng(SEED)
    ranks = np.arange(1, n_experts + 1)
    p = 1.0 / ranks**zipf_a
    p /= p.sum()

    # 每个 token 抽 top-k 个互异专家（按受欢迎程度加权，近似路由）
    # 用两次选择：先按 zipf 抽 k*3 再 unique 截断，保证可复现且够快
    raw = rng.choice(n_experts, size=(n_tokens, top_k * 3), p=p)
    routes = np.empty((n_tokens, top_k), dtype=int)
    for i in range(n_tokens):
        seen = []
        for e in raw[i]:
            if e not in seen:
                seen.append(int(e))
            if len(seen) == top_k:
                break
        while len(seen) < top_k:  # 极罕见补齐
            extra = int(rng.integers(0, n_experts))
            if extra not in seen:
                seen.append(extra)
        routes[i] = seen
    expert_load = np.bincount(routes.ravel(), minlength=n_experts).astype(np.float64)

    experts_per_gpu = n_experts // n_gpus  # 8

    def gpu_load_from_map(owner_lists):
        """owner_lists[e] = 持有专家 e 的 GPU 列表。token 发给当前最轻的副本。"""
        load = np.zeros(n_gpus)
        # 为了速度：按专家聚合后再贪心摊到副本（同一专家的 token 均分到副本会低估热点；
        # 这里按到达顺序逐 token 摊，保持 straggler 语义）
        for e in routes.ravel():
            owners = owner_lists[e]
            g = owners[int(np.argmin(load[owners]))]
            load[g] += 1
        return load

    # 1) 连续放置：GPU i 拿专家 [8i, 8i+8)
    contig = [[] for _ in range(n_experts)]
    for e in range(n_experts):
        contig[e] = [e // experts_per_gpu]

    # 2) 贪心装箱：热专家优先放到当前最轻的 GPU（仍每专家一份）
    order = np.argsort(-expert_load)
    packed_owner = [[] for _ in range(n_experts)]
    gpu_of = np.full(n_experts, -1)
    packed_load_est = np.zeros(n_gpus)
    slot = np.zeros(n_gpus, dtype=int)
    for e in order:
        # 只往还没装满 8 个专家的卡上放
        cand = [g for g in range(n_gpus) if slot[g] < experts_per_gpu]
        g = min(cand, key=lambda x: packed_load_est[x])
        gpu_of[e] = g
        packed_owner[e] = [g]
        packed_load_est[g] += expert_load[e]
        slot[g] += 1

    # 3) EPLB：在装箱基础上，给最热的专家加冗余副本，放到最轻的卡
    eplb_owner = [lst[:] for lst in packed_owner]
    extra_slot = np.zeros(n_gpus, dtype=int)  # 每卡额外可放若干冗余
    # 估算当前 GPU 负载
    est = packed_load_est.copy()
    hot = order[:n_redundant]
    for e in hot:
        g = int(np.argmin(est))
        if g not in eplb_owner[e]:
            eplb_owner[e].append(g)
            extra_slot[g] += 1
            # 副本会分走一部分负载
            est[g] += expert_load[e] / len(eplb_owner[e])
            est[gpu_of[e]] -= expert_load[e] / (len(eplb_owner[e]) * (len(eplb_owner[e]) - 1) or 1)
            est[gpu_of[e]] = max(est[gpu_of[e]], 0)

    print("[C] expert routing imbalance (max/mean GPU load)")
    names = [
        ("连续放置", contig),
        ("按热度装箱", packed_owner),
        (f"装箱 + {n_redundant} 冗余", eplb_owner),
    ]
    stats = {}
    for name, owners in names:
        load = gpu_load_from_map(owners)
        imb = load.max() / load.mean()
        stats[name] = (load, imb)
        print(f"    {name:16s} max={load.max():.0f}  mean={load.mean():.0f}  max/mean={imb:.2f}x  min={load.min():.0f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.4))
    # 左：专家热度
    topn = 40
    ax1.bar(np.arange(topn), expert_load[order[:topn]] / expert_load.sum() * 100, color=ACCENT, width=0.85)
    ax1.set_xlabel("专家（按热度排序的前 40 个）")
    ax1.set_ylabel("被路由到的 token 占比 (%)")
    ax1.set_title("路由是 zipf，不是均匀", color=INK, fontsize=12)

    labels = list(stats.keys())
    imbs = [stats[n][1] for n in labels]
    colors = [SOFT, MUTED, ACCENT]
    ax2.bar(labels, imbs, color=colors, width=0.6)
    ax2.set_ylabel("GPU 负载不均（max / mean）")
    ax2.set_title("冗余副本把 straggler 压下来", color=INK, fontsize=12)
    ax2.set_ylim(0, max(imbs) * 1.18)
    ax2.grid(axis="x", visible=False)
    for lab, v in zip(labels, imbs):
        ax2.text(lab, v + 0.03, f"{v:.2f}×", ha="center", color=INK, fontsize=10)
    fig.suptitle("EPLB：把最烫的专家复制到闲卡上", color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "sim-eplb-imbalance.png"), dpi=150)
    plt.close(fig)
    return stats


# ----------------------------------------------------------------------------
# 实验 D：长序列下 TP 激活同步 vs CP 传 KV
# ----------------------------------------------------------------------------
def exp_d():
    """一层注意力附近的通信载荷随序列长度怎么涨。

    TP：两次 all-reduce 的激活 ≈ 2 · S · hidden · 2B
    Ring CP：每卡环传 KV ≈ (1 − 1/C) · S · d_kv · 2(K+V) · 2B
    Ulysses：两次 all-to-all ≈ 2 · S · hidden · 2B / C
    """
    hidden, d_kv, dtype, C = 8192, 1024, 2, 8
    seqs = np.array([2048, 4096, 8192, 16384, 32768, 65536, 131072])

    tp_mb = 2 * seqs * hidden * dtype / 1e6
    ring_mb = (1 - 1 / C) * seqs * d_kv * 2 * dtype / 1e6
    ulysses_mb = 2 * seqs * hidden * dtype / C / 1e6

    print("[D] prefill comm payload per layer (MB), C=8, GQA d_kv=1024")
    for s, a, b, c in zip(seqs, tp_mb, ring_mb, ulysses_mb):
        print(f"    S={s:6d}  TP-allreduce={a:8.1f}  Ring-CP={b:7.1f}  Ulysses={c:7.1f}")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(seqs, tp_mb, "-o", color=MUTED, lw=2.2, label="TP 激活 all-reduce")
    ax.plot(seqs, ring_mb, "-o", color=ACCENT, lw=2.2, label="Ring CP 传 KV（GQA）")
    ax.plot(seqs, ulysses_mb, "-o", color=INK, lw=2.2, label="Ulysses all-to-all")
    ax.set_xscale("log", base=2)
    ax.set_xticks(seqs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: f"{int(v)//1024}k"))
    ax.set_xlabel("序列长度 S")
    ax.set_ylabel("一层通信载荷 (MB / GPU)")
    ax.set_title("长上下文该切 CP，而不是把 TP 再加大", color=INK, fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sim-cp-vs-tp-comm.png"), dpi=150)
    plt.close(fig)
    return seqs, tp_mb, ring_mb, ulysses_mb


if __name__ == "__main__":
    import matplotlib.ticker  # noqa

    os.makedirs(OUT, exist_ok=True)
    exp_a()
    print()
    exp_b()
    print()
    exp_c()
    print()
    exp_d()
    print(f"\ncharts -> {OUT}")
