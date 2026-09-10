"""Restyle of the PD-disaggregation experiment plots to the diagram-design palette.

Simulation logic is byte-for-byte identical to the original pd_experiments.py, so
the numbers reproduce exactly (SEED=42). Only colors, fonts, background and axis
chrome change, to match the diagram-design editorial skin used by the redrawn
schematic figures (paper #f5f5f5, ink #2d3142, accent #eb6c36, muted #4f5d75).
"""
import os
import heapq
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
OUT = "/workspace/public/assets/posts/pd-disaggregation"
os.makedirs(OUT, exist_ok=True)

# ---- diagram-design palette ----
PAPER = "#f5f5f5"
INK = "#2d3142"
MUTED = "#4f5d75"
SOFT = "#7a8399"
ACCENT = "#eb6c36"
RULE = "#d7d9de"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.dpi": 150,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": RULE,
    "grid.linewidth": 0.8,
})

A_ITER, B_ITER = 0.5, 0.005


def prompt_lengths(n, rng):
    return np.clip(rng.lognormal(6.6, 0.5, n), 64, 8000).astype(int)


def output_lengths(n, rng):
    return np.clip(rng.lognormal(4.8, 0.7, n), 4, 1000).astype(int)


def _clean(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------------------
# A. colocated vs disaggregated: goodput vs offered load
# ---------------------------------------------------------------------------
def simulate(rate, disagg, horizon_ms=30000):
    rng = np.random.default_rng(SEED)
    M = 48
    budget = 1024
    slo_tpot = 1.6
    slo_ttft = 250.0
    max_pending = 512
    prefill_tok_per_ms = 220.0
    bw_bytes_per_ms = 150e9 / 1e3
    kv_bytes_per_token = 2 * 80 * 8 * 128 * 2

    clock = 0.0
    next_arrival = rng.exponential(1.0 / rate)
    waiting = []
    wi = 0
    ready = []
    seq = 0
    running = []
    done_tpot = []
    done_ttft = []

    def new_req(arr):
        P = int(np.clip(rng.lognormal(6.6, 0.5), 64, 8000))
        O = int(np.clip(rng.lognormal(4.8, 0.7), 4, 1000))
        return {"P": P, "O": O, "arr": arr, "out_remaining": O,
                "prefill_remaining": P, "tpot_sum": 0.0, "tpot_n": 0, "ttft": None}

    prefill_free = 0.0
    while clock < horizon_ms:
        while next_arrival <= clock:
            arr = next_arrival
            next_arrival += rng.exponential(1.0 / rate)
            if disagg:
                if len(ready) - 0 >= max_pending:
                    continue
                r = new_req(arr)
                start = max(prefill_free, arr)
                prefill_free = start + r["P"] / prefill_tok_per_ms
                xfer = (kv_bytes_per_token * r["P"]) / bw_bytes_per_ms
                r["ttft"] = (prefill_free + xfer) - arr
                seq += 1
                heapq.heappush(ready, (prefill_free + xfer, seq, r))
            else:
                if (len(waiting) - wi) >= max_pending:
                    continue
                waiting.append(new_req(arr))

        if disagg:
            while ready and ready[0][0] <= clock and len(running) < M:
                running.append(heapq.heappop(ready)[2])

        decode_tokens = len(running)
        prefill_tokens = 0
        if not disagg and wi < len(waiting) and len(running) < M:
            head = waiting[wi]
            room = budget - decode_tokens
            if room > 0:
                chunk = min(room, head["prefill_remaining"])
                head["prefill_remaining"] -= chunk
                prefill_tokens = chunk
                if head["prefill_remaining"] == 0:
                    head["ttft"] = clock - head["arr"]
                    running.append(head)
                    wi += 1
                    if wi > 4096:
                        waiting = waiting[wi:]; wi = 0

        iter_tokens = decode_tokens + prefill_tokens
        dt = A_ITER + B_ITER * iter_tokens
        clock += dt

        still = []
        for r in running:
            r["tpot_sum"] += dt
            r["tpot_n"] += 1
            r["out_remaining"] -= 1
            if r["out_remaining"] <= 0:
                done_tpot.append(r["tpot_sum"] / r["tpot_n"])
                done_ttft.append(r["ttft"] if r["ttft"] is not None else 1e9)
            else:
                still.append(r)
        running = still

        if not running and wi >= len(waiting) and not ready and next_arrival > clock:
            clock = next_arrival

    if not done_tpot:
        return 0.0, float("inf"), float("inf")
    tpots = np.array(done_tpot)
    ttfts = np.array(done_ttft)
    good = (tpots <= slo_tpot) & (ttfts <= slo_ttft)
    good_rate = good.sum() / (clock / 1000.0)
    return good_rate, float(np.percentile(tpots, 99)), float(ttfts.mean())


def exp_a():
    rates = np.linspace(0.05, 0.9, 12)
    colo_g, dis_g = [], []
    for rate in rates:
        g_c, _, _ = simulate(rate, disagg=False)
        g_d, _, _ = simulate(rate, disagg=True)
        colo_g.append(g_c)
        dis_g.append(g_d)
    colo_g, dis_g = np.array(colo_g), np.array(dis_g)
    offered = rates * 1000.0
    print(f"[A] colo  peak goodput={colo_g.max():.0f} req/s")
    print(f"[A] disagg peak goodput={dis_g.max():.0f} req/s")
    print(f"[A] peak-goodput gain = {dis_g.max()/max(colo_g.max(),1e-9):.2f}x")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(offered, offered, "--", color=INK, lw=1.0, alpha=0.4, label="ideal (all served)")
    ax.plot(offered, colo_g, "-o", color=MUTED, lw=2.0, ms=4, label="colocated (interference)")
    ax.plot(offered, dis_g, "-o", color=ACCENT, lw=2.4, ms=5, label="disaggregated")
    ax.set_xlabel("offered load (requests / s)")
    ax.set_ylabel("goodput\n(req/s within TTFT & TPOT SLOs)")
    ax.set_ylim(0, max(dis_g.max(), colo_g.max()) * 1.25)
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    _clean(ax)
    ax.set_title(f"Experiment A: goodput vs load — disaggregation pushes the SLO knee out\n"
                 f"(peak goodput {dis_g.max()/max(colo_g.max(),1e-9):.1f}x higher)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sim-goodput-vs-load.png"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B. when to disaggregate
# ---------------------------------------------------------------------------
def exp_b():
    rng = np.random.default_rng(SEED)
    avg_P = int(prompt_lengths(20000, rng).mean())
    kv_bytes_per_token = 2 * 80 * 8 * 128 * 2
    compute_ms_per_token = 0.011
    loads = np.linspace(2, 40, 160)
    bws_g = np.logspace(np.log10(0.8), np.log10(500), 160) * 1e9
    per_req = avg_P * compute_ms_per_token - (kv_bytes_per_token * avg_P) / bws_g * 1e3
    Z = np.outer(per_req, loads)
    print(f"[B] avg prompt = {avg_P} tokens")
    bw_star = kv_bytes_per_token * 1e3 / compute_ms_per_token / 1e9
    print(f"[B] break-even interconnect ≈ {bw_star:.1f} GB/s")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    LL, BB = np.meshgrid(loads, bws_g / 1e9)
    clip = 600.0
    pcm = ax.pcolormesh(LL, BB, np.clip(Z, -clip, clip), cmap="RdYlGn",
                        vmin=-clip, vmax=clip, shading="auto")
    ax.axhline(bw_star, color=INK, lw=1.6, ls="--")
    ax.text(3, bw_star * 1.12, f"break-even ≈ {bw_star:.0f} GB/s", fontsize=8.5, color=INK)
    ax.set_yscale("log")
    ax.set_yticks([1, 5, 12.5, 40, 150, 400])
    ax.set_yticklabels(["1", "5", "12.5", "40", "150", "400"])
    ax.set_ylabel("interconnect bandwidth (GB/s, log)")
    ax.set_xlabel("offered load (requests / s)")
    ax.set_title("Net benefit of disaggregation\n(green = disagg wins, red = colocated wins)", fontsize=10.5)
    cb = fig.colorbar(pcm, ax=ax, label="ms/s saved (decode stall) − spent (KV xfer)")
    cb.outline.set_edgecolor(MUTED)

    ax = axes[1]
    ramp = [(1e9, INK, 0.85), (12.5e9, MUTED, 0.85), (150e9, SOFT, 0.9)]
    for bw, col, a in ramp:
        transfer_ms = (kv_bytes_per_token * avg_P) / bw * 1e3
        ax.plot(loads, loads * transfer_ms, lw=2.0, color=col, alpha=a,
                label=f"KV xfer @ {bw/1e9:.0f} GB/s")
    ax.plot(loads, loads * avg_P * compute_ms_per_token, "--", color=ACCENT, lw=2.4,
            label="interference avoided (colo cost)")
    ax.set_xlabel("offered load (requests / s)")
    ax.set_ylabel("cost (ms of engine time / s)")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False)
    _clean(ax)
    ax.set_title("Disagg wins when 'interference avoided' > 'KV transfer'", fontsize=10.5)
    fig.suptitle("Experiment B: when is PD disaggregation worth it? (avg prompt ≈ %d tokens)" % avg_P,
                 fontsize=12, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sim-when-to-disaggregate.png"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


# ---------------------------------------------------------------------------
# C. serialized vs layer-wise transfer
# ---------------------------------------------------------------------------
def exp_c():
    L = 80
    kv_bytes_per_token = 2 * 80 * 8 * 128 * 2
    prompt = np.arange(256, 8001, 64)
    bw = 40e9
    prefill_tok_per_ms = 220.0
    transfer_ms = (kv_bytes_per_token * prompt) / bw * 1e3
    serialized_overhead = transfer_ms
    per_layer_xfer = transfer_ms / L
    prefill_ms = prompt / prefill_tok_per_ms
    per_layer_compute = prefill_ms / L
    layerwise_overhead = np.maximum(per_layer_xfer, transfer_ms - per_layer_compute * (L - 1))

    for P in [1024, 4096, 8000]:
        idx = int(np.argmin(np.abs(prompt - P)))
        s, l = serialized_overhead[idx], layerwise_overhead[idx]
        print(f"[C] P={prompt[idx]:5d}  serialized={s:7.2f} ms  layer-wise={l:6.2f} ms  hidden={100*(1-l/s):.0f}%")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(prompt, serialized_overhead, color=MUTED, lw=2.2, label="serialized (transfer after prefill)")
    ax.plot(prompt, layerwise_overhead, color=ACCENT, lw=2.6, label="layer-wise (overlap with compute)")
    ax.fill_between(prompt, layerwise_overhead, serialized_overhead, color=ACCENT, alpha=0.10)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("KV-transfer overhead added to TTFT (ms)")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    _clean(ax)
    ax.set_title("Experiment C: layer-wise transfer hides most of the KV handoff\n"
                 "(InfiniBand ≈ 40 GB/s, 80 layers)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sim-layerwise-transfer.png"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    print("=== Experiment A ==="); exp_a()
    print("=== Experiment B ==="); exp_b()
    print("=== Experiment C ==="); exp_c()
    print("all restyled experiments done")
