#!/usr/bin/env python3
"""第 3 篇：无需 GPU 的单机调度仿真，三个实验，SEED=42 可复现。

A: 静态批处理 vs 连续批处理 —— 槽位利用率与总迭代数
B: FCFS vs SJF —— 平均延迟与 p99
C: naïve 混批 vs chunked prefill —— TBT 稳定性

图表用 diagram-design 调色板 + 中文标注。
输出 PNG 到 public/assets/posts/scheduling/。
"""

import heapq
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SEED = 42
OUT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "public", "assets", "posts", "scheduling")
)

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


def decode_lengths(n, rng):
    x = rng.lognormal(mean=4.4, sigma=0.85, size=n)
    return np.clip(x, 4, 2048).astype(int)


def exp_a():
    rng = np.random.default_rng(SEED)
    B, N = 32, 4000
    L = decode_lengths(N, rng)
    used = int(L.sum())

    static_iters = sum(int(L[i : i + B].max()) for i in range(0, N, B))
    static_util = used / (B * static_iters)

    heap, idx = [], 0
    while idx < B and idx < N:
        heap.append(int(L[idx]))
        idx += 1
    heapq.heapify(heap)
    cont_iters = 0
    while heap:
        step = heap[0]
        cont_iters += step
        heap = [r - step for r in heap]
        newheap = []
        for r in heap:
            if r > 0:
                newheap.append(r)
            elif idx < N:
                newheap.append(int(L[idx]))
                idx += 1
        heap = newheap
        heapq.heapify(heap)
    cont_util = used / (B * cont_iters)

    print(f"[A] static util={static_util * 100:.1f}%  iters={static_iters}")
    print(f"    cont   util={cont_util * 100:.1f}%  iters={cont_iters}")
    print(f"    fewer iterations = {static_iters / cont_iters:.2f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.2))
    labels = ["静态批处理", "连续批处理"]
    ax1.bar(labels, [static_util * 100, cont_util * 100], color=[MUTED, ACCENT], width=0.6)
    ax1.set_ylabel("GPU 槽位利用率 (%)")
    ax1.set_title("利用率：空转 vs 立刻补满", color=INK, fontsize=11)
    ax1.set_ylim(0, 110)
    ax1.bar_label(
        ax1.containers[0],
        labels=[f"{static_util * 100:.1f}%", f"{cont_util * 100:.1f}%"],
        padding=3,
        color=INK,
        fontsize=10,
    )

    ax2.bar(labels, [static_iters, cont_iters], color=[MUTED, ACCENT], width=0.6)
    ax2.set_ylabel("清空同等工作量所需的总迭代数")
    ax2.set_title(f"总时长：连续批处理少 {static_iters / cont_iters:.2f}× 迭代", color=INK, fontsize=11)
    ax2.bar_label(ax2.containers[0], fmt="{:,.0f}", padding=3, color=INK, fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(axis="x", visible=False)
    fig.suptitle("实验 A：静态 vs 连续批处理（同一条工作负载）", color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "sim-batching-utilization.png"), dpi=150)
    plt.close(fig)


def simulate_queue(arrivals, service, order, c):
    n = len(arrivals)
    done = np.zeros(n)
    free_at = [0.0] * c
    a_order = np.argsort(arrivals)
    pending, ai, finished, t = [], 0, 0, 0.0
    while finished < n:
        slot = int(np.argmin(free_at))
        slot_free = free_at[slot]
        while ai < n and arrivals[a_order[ai]] <= max(slot_free, t):
            pending.append(a_order[ai])
            ai += 1
        if not pending:
            if ai < n:
                t = arrivals[a_order[ai]]
                pending.append(a_order[ai])
                ai += 1
            else:
                break
        start = max(slot_free, min(arrivals[i] for i in pending))
        if order == "sjf":
            pick = min(pending, key=lambda i: service[i])
        else:
            pick = min(pending, key=lambda i: arrivals[i])
        pending.remove(pick)
        finish = start + service[pick]
        done[pick] = finish - arrivals[pick]
        free_at[slot] = finish
        finished += 1
    return done


def exp_b():
    rng = np.random.default_rng(SEED)
    N, c = 3000, 16
    service = np.clip(rng.lognormal(mean=4.4, sigma=1.1, size=N), 4, 4096).astype(float)
    rate = c / service.mean() * 1.15
    arrivals = np.cumsum(rng.exponential(1.0 / rate, size=N))

    lat_fcfs = simulate_queue(arrivals, service, "fcfs", c)
    lat_sjf = simulate_queue(arrivals, service, "sjf", c)
    for name, lat in [("FCFS", lat_fcfs), ("SJF", lat_sjf)]:
        print(f"[B] {name}: mean={lat.mean():.0f}  p99={np.percentile(lat, 99):.0f}")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(2)
    w = 0.34
    means = [lat_fcfs.mean(), np.percentile(lat_fcfs, 99)]
    sjfs = [lat_sjf.mean(), np.percentile(lat_sjf, 99)]
    b1 = ax.bar(x - w / 2, means, w, color=MUTED, label="FCFS")
    b2 = ax.bar(x + w / 2, sjfs, w, color=ACCENT, label="SJF")
    ax.set_xticks(x, ["平均延迟", "p99 延迟"])
    ax.set_ylabel("延迟（decode 步）")
    ax.set_title(
        f"实验 B：FCFS vs SJF —— 平均延迟低 {lat_fcfs.mean() / lat_sjf.mean():.1f}×\n（同一到达流与服务时间，16 个槽位）",
        color=INK,
        fontsize=12,
    )
    ax.legend(frameon=False)
    ax.bar_label(b1, fmt="{:,.0f}", padding=3, color=INK, fontsize=9)
    ax.bar_label(b2, fmt="{:,.0f}", padding=3, color=INK, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sim-fcfs-vs-sjf.png"), dpi=150)
    plt.close(fig)


def exp_c():
    rng = np.random.default_rng(SEED)
    n_decode, budget, steps = 32, 512, 4000
    a, b = 0.5, 0.008
    prefill_prob = 0.06

    def prompt_len():
        return int(np.clip(rng.lognormal(7.0, 0.6), 128, 8192))

    def run(chunked):
        tbt, pending = [], 0
        for _ in range(steps):
            toks = n_decode
            if chunked:
                if pending == 0 and rng.random() < prefill_prob:
                    pending = prompt_len()
                if pending > 0:
                    chunk = min(budget - n_decode, pending)
                    toks += chunk
                    pending -= chunk
            else:
                if rng.random() < prefill_prob:
                    toks += prompt_len()
            tbt.append(a + b * toks)
        return np.array(tbt)

    # 正文里两次 run() 共用同一只 rng，顺序：先 naive 再 chunked
    naive = run(False)
    chunked = run(True)

    print(
        f"[C] naive:   p50={np.percentile(naive, 50):.2f} "
        f"p99={np.percentile(naive, 99):.2f} max={naive.max():.2f}"
    )
    print(
        f"    chunked: p50={np.percentile(chunked, 50):.2f} "
        f"p99={np.percentile(chunked, 99):.2f} max={chunked.max():.2f}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.2))
    ax1.plot(naive, color=MUTED, lw=0.8, label="naïve 混批")
    ax1.plot(chunked, color=ACCENT, lw=0.8, label="chunked prefill")
    ax1.axhline(a + b * budget, color=SOFT, ls="--", lw=1, label="预算上限")
    ax1.set_xlabel("迭代")
    ax1.set_ylabel("TBT（相对时间）")
    ax1.set_title("TBT 轨迹：naïve 尖刺 vs chunked 平稳", color=INK, fontsize=11)
    ax1.legend(frameon=False, fontsize=9)

    cats = ["p99 TBT", "最大 TBT（最糟尖刺）"]
    x = np.arange(2)
    w = 0.34
    b1 = ax2.bar(
        x - w / 2,
        [np.percentile(naive, 99), naive.max()],
        w,
        color=MUTED,
        label="naïve",
    )
    b2 = ax2.bar(
        x + w / 2,
        [np.percentile(chunked, 99), chunked.max()],
        w,
        color=ACCENT,
        label="chunked",
    )
    ax2.set_xticks(x, cats)
    ax2.set_ylabel("TBT（相对时间）")
    p99_ratio = np.percentile(naive, 99) / max(np.percentile(chunked, 99), 1e-9)
    max_ratio = naive.max() / max(chunked.max(), 1e-9)
    ax2.set_title(f"尾部 TBT：p99 好 {p99_ratio:.1f}×，最糟尖刺好 {max_ratio:.1f}×", color=INK, fontsize=11)
    ax2.legend(frameon=False, fontsize=9)
    ax2.bar_label(b1, fmt="%.1f", padding=3, color=INK, fontsize=9)
    ax2.bar_label(b2, fmt="%.1f", padding=3, color=INK, fontsize=9)
    ax2.grid(axis="x", visible=False)

    fig.suptitle("实验 C：naïve vs chunked prefill —— decode 延迟稳定性", color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "sim-chunked-prefill-tbt.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    exp_a()
    print()
    exp_b()
    print()
    exp_c()
    print(f"\ncharts -> {OUT}")
