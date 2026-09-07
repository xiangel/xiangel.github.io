---
author: xiangel
pubDatetime: 2026-09-04T02:30:00Z
title: "KV Cache 详解：从 PagedAttention 到 Prefix Caching"
slug: kv-cache-paged-attention-and-prefix-caching
featured: true
draft: false
tags:
  - LLM
  - 推理系统
  - KV-Cache
description: 用操作系统的类比，把 LLM 推理里最吃显存的 KV Cache 讲清楚：PagedAttention 如何像虚拟内存一样管理显存，Prefix Caching 如何让重复的前缀不再重算。附一组无需 GPU、可复现的仿真实验与图表。
---

如果你自己部署过大模型，多半遇到过两件怪事：

1. 模型权重明明放得下，可上下文一长、并发一高，显存立刻 **OOM**。
2. 第二次问一个相似的问题，"首字"（首个 token）明显更快就蹦出来了。

这两件事背后，是同一个主角：**KV Cache**。它是自回归推理的"记忆"，也是长上下文、高并发场景下最贵的那块显存。这篇文章用一条主线把它讲透——

- **KV Cache 怎么才放得下？** → `PagedAttention`
- **重复的前缀能不能不重算？** → `Prefix Caching`

全程我会用**操作系统**做类比（这也是 PagedAttention 论文的灵感来源），并在最后给出一组**无需 GPU、可复现**的仿真实验，用一手数据印证每一个结论。

## Table of contents

## 一、先搞懂 KV Cache 是什么

Transformer 生成文本是**逐 token**的。生成第 t 个 token 时，它的 Query 要和前面**所有** token 的 Key / Value 做注意力：

```text
Attention(Q_t, K_1:t, V_1:t) = softmax( Q_t · K_1:tᵀ / √d ) · V_1:t
                                             ▲                 ▲
                                        需要所有历史 K      需要所有历史 V
```

关键洞察：**K 和 V 只取决于各自 token 的输入和（冻结的）权重**，一旦算出来就永远不变。既然如此，何必每步都重算前面所有 token 的 K/V？**算一次，缓存起来，反复读**——这就是 KV Cache。它把解码从每步 O(n²) 降到每步 O(n)。

> **类比**：KV Cache 就像你做长阅读理解时的**笔记**。每读一句就记下要点，回答问题时翻笔记即可，而不是把整篇文章重读一遍。

推理因此分成两个阶段，它们的成本结构完全不同：

| 阶段        | 做什么                           | 瓶颈             | 用户感知              |
| ----------- | -------------------------------- | ---------------- | --------------------- |
| **Prefill** | 一次性并行处理整个 prompt        | 计算密集（算力） | TTFT（首 token 延迟） |
| **Decode**  | 逐个生成 token，每步读整个 cache | 访存密集（带宽） | TPOT（每 token 延迟） |

![Prefill 与 Decode 两阶段示意：Prefill 一次性并行填充 KV Cache，Decode 逐 token 读取不断增长的 KV Cache](/assets/posts/kv-cache/diagram-prefill-decode.png)

记住这张表：**Prefix Caching 优化的是 Prefill/TTFT**，而 KV Cache 的显存压力主要来自 Decode 阶段不断增长的历史。

KV Cache 有多大？粗略地：`Bytes ≈ 2(K和V) × 层数 × KV头数 × head维度 × token数 × dtype字节数`。

它随 **序列长度、层数、KV 头数** 线性增长。这也是为什么 `GQA`（Grouped-Query Attention）、`MQA`、`MLA` 这些"减少 KV 头"的技巧如此重要——它们直接砍掉 KV Cache 的体积。但即便压缩过，KV Cache 依然是长上下文下的显存大户，于是就有了下一个问题：**这么大的东西，到底该怎么在显存里摆放？**

## 二、PagedAttention：把 KV Cache 当"虚拟内存"来管

### 老办法为什么浪费

在 PagedAttention 出现之前，主流做法（FasterTransformer 一类）是给每个请求**预留一整块连续显存**，大小按 `max_model_len`（比如 2048）来算。问题是一个请求可能只生成 300 个 token，剩下的全是**空占着**的浪费。

这会造成三种浪费：

```text
连续预留 max_model_len = 2048 的一个请求：

  [■■■■■■■■□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□]
   ↑ 实际用了 300     ↑ 预留却没用（内部碎片 + 预留浪费）

多个变长请求挤在一起，中间还留下用不上的空洞（外部碎片）：

  [req A ████][ 空洞 ][req B ██][   空洞   ][req C ███]
              ↑ 放不下新请求，即使总空闲量够
```

- **内部碎片**：预留了 2048，只用 300。
- **预留浪费**：为"还没生成"的 token 提前占位。
- **外部碎片**：变长的连续块之间留下用不上的空隙。

PagedAttention 的论文实测：这类系统 **60–80% 的 KV 显存都被浪费掉了**。

### 核心思想：分页

> **类比**：这正是操作系统几十年前就解决过的问题。程序以为自己拥有一整段连续内存（虚拟地址），实际上 OS 把它切成固定大小的**页（page）**，散落在物理内存各处，用**页表**做映射。进程根本不知道背后是零散的。

PagedAttention 把这套**原封不动**搬到 KV Cache 上：

| 操作系统        | PagedAttention             |
| --------------- | -------------------------- |
| 页 page         | KV block（默认 16 token）  |
| 页表 page table | block table（每序列一张）  |
| 虚拟地址        | 逻辑块号                   |
| 物理帧 frame    | 物理块（显存池里任意位置） |

```text
逻辑视图（模型以为的连续序列）        物理视图（显存块池，任意摆放）
Seq A: [blk0][blk1][blk2]             [B0][B1][B2][B3][B4][B5]...
          |     |     |                ▲         ▲    ▲
block table(A): 0→B2, 1→B5, 2→B0 ─────┘─────────┘────┘   （非连续，零外部碎片）
```

![PagedAttention 用 block table 把序列的逻辑块映射到显存池里任意位置的物理块](/assets/posts/kv-cache/diagram-block-table.png)

每个序列有一张 **block table**，把"逻辑上连续"的块号，映射到显存池里**任意位置**的物理块。要增长就从空闲池再取一块，结束就还回去。于是：

- 外部碎片 **归零**（所有块一样大，随便放）。
- 内部碎片最多只剩**一个半满的尾块**（≤ 15 个 token 的浪费）。

**block size 的取舍**（等价于 OS 选页大小）：块太小，碎片少但元数据/索引开销大；块太大，反之且尾块浪费多。常见实现取 **16 token** 左右，是 A100/H100 上的经验最优。

### 一个绕不开的工程细节：定制 kernel

标准的 FlashAttention kernel 假设 K/V 在显存里是**连续**的。PagedAttention 的块是散落的，所以必须写**定制 CUDA kernel**：按 block table 去各个物理块**gather**（收集）K/V。关键洞察是——**逐序列看是分散的，但逐块看是连续的**（一块内 16 个 token 紧凑排列），所以这层间接寻址的开销在 GPU 上几乎可忽略。

### 白捡的红利：Copy-on-Write

有了"物理块 + 引用计数"这层抽象，**共享**就变得极其自然。

> **类比**：就像 `fork()` 出子进程时，父子共享同一份内存页，**谁要写才复制那一页**（copy-on-write）。

并行采样（`best_of=4`）或 beam search 时，多个候选**共享同一份前缀块**，只在各自**分叉、真正写入新 token**的地方才复制那一块。被淘汰的候选立刻减引用计数、归还块，零拷贝开销。

![Copy-on-Write：整块的共享前缀块从不复制，只有被两个候选共同写入的那个半满尾块被复制一次](/assets/posts/kv-cache/diagram-cow.png)

这个"共享前缀块"的能力，正是下一章 **Prefix Caching** 的地基。

## 三、Prefix Caching：重复的前缀，凭什么算两遍？

真实流量里，**前缀高度重复**：同一个系统提示、同一批 few-shot 示例、RAG 里同一份检索文档、多轮对话里不断累积的历史……如果每个请求都把这些**从头 prefill 一遍**，纯属浪费。

Prefix Caching 的思想一句话：**把已经算好的前缀 KV 块留着，下个请求前缀相同就直接复用，跳过重算。** 它几乎是"免费的午餐"，而且**不改变模型输出**。

### 实现思路一：块级哈希链（chained block hashing）

一种常见做法是用**哈希**来判断"这一块是否已经算过"。每个块的哈希是**链式**的（类似 Merkle chain）：

```text
block hash_i = hash( hash_{i-1},  本块的 token,  额外key )
                        ▲              ▲             ▲
                     父块哈希       内容精确匹配    可选的隔离标识
```

```text
请求1: [ system prompt .......... ][ 用户问题 A ]
        h0   h1   h2   h3            （算完缓存 h0..h3）

请求2: [ system prompt .......... ][ 用户问题 B ]
        h0   h1   h2   h3   ✔命中     ← 前缀块哈希全中，直接复用！
                                      只需为"问题 B"新算
```

![Prefix Caching：两个请求共享的前缀块哈希全部命中，只有末尾独特部分需要重算](/assets/posts/kv-cache/diagram-hash-chain.png)

几个要点：

- **块级、只缓存整块**：一个前缀如果只填了新块的一部分（不足一个块，如 16 token），那半块**不会**被缓存。
- **精确匹配**：哈希链意味着命中要求**逐 token 完全一致**，而不是"语义相似"。第一个对不上的块，就是重算的起点。
- **驱逐**：空闲块按 **LRU** 回收；正在被引用的块（引用计数 > 0）不会被驱逐；请求结束时按"尾块先驱逐"的逆序归还（尾块哈希了最多 token，最不可能被别人复用）。

（哈希算法的选择、多租户 salt 隔离、跨进程哈希一致性等**实现细节**论文并未展开，留到后续分析代码时再谈。）

### 实现思路二：基数树（radix tree / RadixAttention）

除了块级哈希，另一类做法把 KV Cache 组织成一棵 **radix tree（基数树 / 压缩前缀树）**：

```text
              (root)
                │  "You are a helpful assistant. "   ← 系统提示，只存一份
             ┌──┴──┐
   "Doc A…"  │     │  "Doc B…"        ← RAG 不同文档在此分叉
          ┌──┘     └──┐
   "…问题1"          "…问题2"          ← 各自会话继续分叉
```

![RadixAttention：把共享前缀组织成基数树，系统提示作为根节点被所有请求共享](/assets/posts/kv-cache/diagram-radix-tree.png)

- 粒度是 **token 级**（页大小 = 1 token），边可以标注**变长**的 token 序列。
- 用 **LRU 驱逐叶子**：系统提示这种根节点被每个请求访问，永远活着；一小时前某次对话的细节则作为叶子被淘汰。
- 配合 **cache-aware 调度**提高命中率。
- 最大差异**不是**"能共享任意中间子串"——它同样是**从根共享前缀**。真正的优势是 **token 级粒度 + 多级（树状分叉）前缀共享**：对"系统提示 → 检索文档 → 对话历史"这种层层分叉的结构，一棵树能自然表达并做高效的匹配/插入/驱逐，命中率通常明显更高。

一句话对比：

| 维度     | 块级哈希链                       | 基数树（RadixAttention）  |
| -------- | ------------------------------- | ------------------------ |
| 数据结构 | 全局哈希表 + 链式块哈希         | radix tree               |
| 粒度     | 固定块（如 16 token），仅整块   | token 级                 |
| 共享结构 | 单层前导前缀（块对齐）          | 前导前缀 + 多级树状分叉（token 级） |
| 强项     | 实现简单、块独立、易分配回收    | RAG/多级共享命中率更高   |

### ChunkAttention：共享之后，如何在 Attention Kernel 里"高效复用"

前面两种做法（块级哈希、基数树）解决的是"**KV 怎么共享、怎么省显存**"。但还有一个常被忽略的缝隙：**共享之后，attention kernel 真的高效利用了这份共享吗？** 传统实现里，即便前缀 KV 在显存里只存了一份，解码时每个请求仍会**各自把这段共享 KV 从显存读一遍**、各自跑一次 attention——访存被重复放大了。

**ChunkAttention**（微软，ACL 2024）正盯着这个缝隙。它做两件事：

1. **Prefix-Aware KV Cache（PAKV）**：把单块的 K/V 张量切成固定大小的 **chunk**（示例 chunk size = 256），组织成一棵**前缀树**。相同前缀的请求在树上**共用同一批 chunk**，前缀 KV 只存一份，后续按请求分叉出各自的私有尾块。

![ChunkAttention 把共享前缀切成 chunk 存进前缀树，只存一份](/assets/posts/kv-cache/diagram-chunkattention-memory.png)

> 显存账（论文示例）：`16 个 chunk × 256 = 4096` 的共享前缀只存一份，3 个请求各自 100 token 的私有尾巴另计，相比"每请求各存整段前缀"约省 **3×** 显存。

2. **Two-Phase Partition（TPP）**：真正的新意在 **attention kernel**。解码时把自注意力拆成两个阶段：
   - **Chunk-first（共享阶段）**：只处理被多个请求共享的 chunk。把这些请求的 query **batch 在一起**去和共享 chunk 做注意力——于是变成 **矩阵 × 矩阵**、能吃满 tensor core；用 **online softmax** 存下部分结果 `O^(C), m^(C), n^(C)`，分区之间无需同步。
   - **Sequence-first（私有阶段）**：再逐请求处理各自的私有尾块（**向量 × 矩阵**），并把上一阶段的共享部分结果**合并**（online softmax）得到最终输出。

![ChunkAttention 的 two-phase partition：共享 chunk 批量算一次，私有尾块逐请求算再合并](/assets/posts/kv-cache/diagram-chunkattention-twophase.png)

关键收益是**数据局部性**：共享前缀的 KV 在 chunk-first 阶段**只从显存读一次**、被所有请求的 query 复用，而不是每请求各读一遍。论文报告：系统提示长度 1024–4096 时，自注意力 kernel 相比 SOTA 实现加速 **3.2–4.8×**。

> **一句话定位**：前面的做法回答了"**KV 怎么共享**"，ChunkAttention 追问"**共享之后，怎么让 attention kernel 真正复用这份共享 KV**"——把"共享"从省显存延伸到了省访存、提升算力利用率。

## 四、动手实验：无需 GPU 的可复现仿真

### 实验 A：显存碎片——contiguous vs paged

模拟真实的聊天长度分布（多数短、长尾），对比"连续预留 `max_model_len`"与"分页（block=16）"：

![Contiguous vs PagedAttention 的显存浪费与并发容量](/assets/posts/kv-cache/fragmentation.png)

一手结果（`mean seq length ≈ 358`，`max_model_len = 2048`）：

- 连续预留浪费 **82.5%**，同样显存预算只能装 **97** 个并发序列；
- 分页（block=16）浪费仅 **2.0%**，能装 **546** 个——**5.6× 的容量提升**。

这与论文"旧系统浪费 60–80%、PagedAttention 把浪费降到 <4%"完全吻合。（注意：容量提升 ≠ 吞吐提升；论文的端到端吞吐是 2–4×，因为吞吐还受算力/调度制约。）

### 实验 B1：省下的 Prefill 随共享前缀比例线性增长

![Prefill 省下的计算量随共享前缀比例变化](/assets/posts/kv-cache/prefix-saving-vs-ratio.png)

共享前缀占 prompt 的比例越高，省下的 prefill 计算就越多，几乎贴着理想直线——因为省掉的正是那段共享前缀的 prefill。

### 实验 B2：同样的内容，"易变字段"放头还是放尾，天壤之别

这是生产里**最容易踩的坑**。假设 prompt 里有个每次都变的字段（时间戳 / 请求 ID，仅 5 个 token）：

![易变字段放头 vs 放尾对命中率和 TTFT 的影响](/assets/posts/kv-cache/prefix-head-vs-tail.png)

一手结果：

- **易变字段放在开头** → 第一个块每次都不同 → 哈希链从第 0 块就断了 → 命中率 **0%**，TTFT 和没缓存一样。
- **同样的字段挪到末尾** → 共享前缀全部命中 → 命中率 **78%**，TTFT 代理值从 1024 → **224**。

> **一句话铁律：静态内容前置，易变字段后置。** 这跟社区实测（某租户把易变字段从头挪到尾，命中率 0.3% → 87%）是同一个故事。

### 实验 B3：显存不够时，LRU 会把共享块挤掉，命中率随之崩塌

用一个真实的**块级链式哈希 + LRU 驱逐**缓存，模拟 8 个租户（8 个不同系统提示）竞争：

![多租户下命中率随缓存容量的变化](/assets/posts/kv-cache/prefix-eviction.png)

一手结果：缓存太小（装不下所有租户前缀，需 256 块）时，前缀块在被复用前就被 LRU 挤掉，命中率从结构上限 **80%** 一路**崩到接近 0**。这解释了为什么生产里**显存占用逼近满载**时，前缀命中率会莫名很低——**不是模板不对，是显存被榨干了**。

## 五、延伸阅读（本文未展开的进阶话题）

这些方向能进一步压榨 KV Cache，留给后续文章或读者自行深入：

- **cache-aware 调度**：在基数树之上，按"最长公共前缀优先"排序等待队列，进一步抬高命中率，值得单独一篇。
- **MLA（Multi-head Latent Attention，DeepSeek）**：把 KV 压成低秩潜向量，从"架构层"直接缩小 KV Cache。
- **KV 量化**：`FP8 / INT8` KV Cache，用精度换一半甚至四分之一的显存。
- **Token 驱逐 / 稀疏**：`H2O`、`SnapKV` 等，只保留"重要"的历史 token。
- **分布式 KV 与 PD 分离**：`LMCache` + `Mooncake` 把多节点内存聚合成共享 KV 池，配合 prefill/decode 分离做**跨实例前缀共享**。

## 六、总结

- **KV Cache** 是自回归推理的记忆，也是长上下文/高并发的显存大户。
- **PagedAttention** 借操作系统的**分页 + copy-on-write**，把显存浪费从 60–80% 降到个位数，并提供"可共享物理块"这一关键抽象。
- **Prefix Caching** 建立在其上，让重复前缀**免于重算**；块级哈希与基数树是两种常见组织方式，殊途同归。
- 由哈希链原理直接得到的一条规律：**静态内容前置、易变字段后置**，才能让前缀稳定命中。

---

## 附录 A：完整仿真脚本

保存为 `kv_experiment.py`，`pip install numpy matplotlib` 后 `python kv_experiment.py` 即可复现本文全部图表（无需 GPU / 模型 / 联网，`SEED=42` 确定性）。

```python
import hashlib
import os
from collections import OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
BLOCK = 16  # tokens per KV block
OUT = os.environ.get("OUT_DIR", "./figs")
os.makedirs(OUT, exist_ok=True)


def sample_lengths(n, rng, max_len):
    """Realistic chat length mix: many short prompts, long tail."""
    x = rng.lognormal(mean=5.6, sigma=0.75, size=n)  # median ~ 270
    return np.clip(x, 8, max_len).astype(int)


# ---- Experiment A: fragmentation (contiguous vs paged) ----
def experiment_fragmentation():
    rng = np.random.default_rng(SEED)
    max_len, budget = 2048, 200_000
    lengths = sample_lengths(20_000, rng, max_len)
    mean_len = lengths.mean()

    contig_capacity = budget // max_len
    contig_waste = 1 - mean_len / max_len

    padded = np.ceil(lengths / BLOCK) * BLOCK
    paged_capacity = int(budget // padded.mean())
    paged_waste = float((padded - lengths).mean() / padded.mean())

    print(f"mean len={mean_len:.1f}  contig waste={contig_waste*100:.1f}% "
          f"cap={contig_capacity}  paged waste={paged_waste*100:.1f}% "
          f"cap={paged_capacity}  gain={paged_capacity/contig_capacity:.1f}x")


# ---- A real block-level chained-hash prefix cache with LRU ----
class PrefixCache:
    def __init__(self, capacity_blocks):
        self.cap = capacity_blocks
        self.store = OrderedDict()

    @staticmethod
    def _h(parent, tokens):
        m = hashlib.sha256()
        m.update(repr(parent).encode())
        m.update(repr(tuple(tokens)).encode())
        return m.hexdigest()

    def process(self, tokens):
        n_full = len(tokens) // BLOCK
        reused, parent, hitting = 0, None, True
        for i in range(n_full):
            blk = tokens[i * BLOCK:(i + 1) * BLOCK]
            h = self._h(parent, blk)
            parent = h
            if hitting and h in self.store:
                self.store.move_to_end(h)
                reused += BLOCK
            else:
                hitting = False
                self.store[h] = True
                self.store.move_to_end(h)
                while len(self.store) > self.cap:
                    self.store.popitem(last=False)
        return reused, len(tokens) - reused


# The remaining experiments (B1 saved-vs-ratio, B2 head-vs-tail,
# B3 eviction) follow the same pattern; the full plotting version is in
# the post's companion figures. See the article body for each result.
if __name__ == "__main__":
    experiment_fragmentation()
```

> 完整含绘图的版本（4 张图）与本文一致，核心逻辑就是上面的 `PrefixCache`：**链式哈希 + LRU**。把 `experiment_*` 按文中描述补齐即可产出全部图表。
