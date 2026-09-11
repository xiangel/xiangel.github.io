# 大模型推理系统系列 · 线路图

本系列以「一条数据在推理系统里如何流动」为主线，从单请求的前向计算，一路讲到分布式集群调度、并行策略、硬件与各类进阶优化。每篇都保持统一的写作签名：

- 一个贯穿全文的生活化**类比**；
- 保留英文技术术语；
- 原创重绘的示意图（diagram-design 风格，**中文标注**，英文技术术语保留）；
- **无需 GPU、`SEED=42` 可复现**的仿真实验；
- 结尾的 `## 参考` 一手来源。

状态图例：✅ 已发布　🚧 评审中　⬜ 规划中

## 总览

| #   | 篇目                                       | 覆盖的框架模块                                                | 贯穿类比                | 无 GPU 实验                                   | 状态 |
| --- | ------------------------------------------ | ------------------------------------------------------------ | ----------------------- | --------------------------------------------- | ---- |
| 1   | 从 Transformer 出发来看推理系统            | 模块全景 / Engine 入口 / API Server                          | —                       | 前向计算与 KV 增长可视化                      | ✅   |
| 2   | KV Cache 详解：PagedAttention 到 Prefix Caching | KV Cache Manager / BlockPool / RadixAttention           | 图书馆借还书            | 分页/前缀命中率与碎片仿真                      | ✅   |
| 3   | 推理调度（单机篇）：Continuous Batching 到缓存感知调度 | Scheduler（单机 step 循环）                          | OS 的 CPU 调度          | 排队/批处理吞吐与时延仿真                      | ✅   |
| 4   | PD 分离：把 Prefill 和 Decode 拆到不同 GPU 池 | KVConnector / KV transfer                                  | 工厂车间的专业化分工    | P/D 配比与传输开销仿真                         | ✅   |
| 5   | 推理调度（分布式篇）：缓存感知路由到全局准入控制 | Router / 准入控制 / 扩缩容                                | 打车平台派单中心        | 缓存感知路由 / 亲和 vs 均衡 / 早拒绝仿真       | ✅   |
| 6   | 并行策略 + MoE 推理                         | Model Executor / TP·PP·DP·EP / EPLB·DeepEP                   | 大厨团队的分工          | 各并行维度的通信/气泡开销仿真                  | 🚧   |
| 7   | 推理模型（long-CoT）服务                    | 调度 + KV 生命周期在长思维链下的新负载                       | 考场里的大考            | 长 CoT 下 KV 占用与 straggler 仿真            | ⬜   |
| 8   | GPU 架构与现有 GPU（含 Attention Kernel）   | 硬件 / Roofline / Attention Kernels（FlashInfer/FA3）        | GPU = 一座工厂          | Roofline 与 kernel 访存/算力仿真              | ⬜   |
| 9   | 推测解码（Speculative Decoding）            | Speculative proposer（EAGLE / MTP / n-gram）                 | 抢答 + 复核             | 接受率/加速比与高并发失效仿真                  | ⬜   |
| 10  | 采样与结构化解码                           | Sampler + StructuredOutputManager / XGrammar                | 掷骰子 + 填表格模板     | 采样分布与语法掩码开销仿真                     | ⬜   |
| 11  | 量化与低精度                               | AWQ / GPTQ / FP8 / NVFP4 + KV/激活量化                       | 压缩打包行李            | 精度-显存-吞吐权衡仿真                         | ⬜   |
| 12  | 多模态推理                                 | Multimodal 输入处理 / VLM / mm cache                         | 多国语翻译入关          | 多模态 token 预算与预处理开销仿真             | ⬜   |
| 13  | 多 LoRA / 适配器多租户                     | LoRA（S-LoRA / Punica）/ 适配器内存池                        | 同底盘换外壳            | 多适配器切换与内存池命中仿真                   | ⬜   |
| 14  | 训练-推理一体化 / RL rollout               | Weight loading + 权重热更新 API                             | 边训边比赛换装备        | 权重热更新时延与 rollout 吞吐仿真            | ⬜   |
| 15  | 可观测、benchmark 与 SLO 方法论            | Metrics / benchmark_serving                                 | 汽车仪表盘              | SLO 达成率与压测曲线复现                       | ⬜   |

## 结构说明

- **数据流主干（1–8）**：从单请求前向 → KV → 单机调度 → PD 分离 → 分布式调度 → 并行/MoE → 长 CoT 负载 → 硬件与 kernel，构成一条「一条请求怎样穿过整个系统」的主线。
- **解码整形簇（9–10）**：推测解码、采样与结构化解码，都是在 decode 阶段「怎么产出下一个 token」上做文章。
- **模型装载与适配（11、13、14）**：量化、多 LoRA、训练-推理一体化，围绕「模型权重如何被压缩、切换、更新」。
- **输入侧（12）**：多模态把非文本输入接入同一条流水线。
- **横切方法论（15）**：可观测与 benchmark，回访并串联全系列的 SLO 主题。
- Attention kernel（FlashInfer / FA3）并入第 8 篇，与 GPU 架构一起讲，避免单开一篇过窄。

## 框架模块覆盖矩阵

以 vLLM V1 与 SGLang 的实际模块划分为参照，核对系列覆盖情况；标注每个模块由哪一篇覆盖，`GAP` 表示原计划遗漏、后补入 9–15 的模块。

| 框架模块（vLLM V1 / SGLang）                      | 覆盖篇目      | 备注                                   |
| ------------------------------------------------- | ------------- | -------------------------------------- |
| API Server（tokenize / mm load / streaming）      | 1             | Engine 入口全景                        |
| Engine Core（busy loop / step）                   | 1、3          | 单机 step 循环在第 3 篇               |
| Scheduler                                         | 3、5          | 单机 / 分布式两篇                     |
| KVCacheManager / BlockPool / RadixCache           | 2             | —                                      |
| KVConnector / KV transfer                         | 4             | PD 分离                                |
| Router / 准入 / 扩缩容                            | 5             | 分布式篇                              |
| Model Executor（TP·PP·DP·EP）                      | 6             | 并行 + MoE                            |
| Attention Kernels（FlashInfer / FA3）             | 8             | GAP → 并入 GPU 架构篇                 |
| Speculative proposer（EAGLE / MTP / n-gram）      | 9             | GAP                                    |
| Sampler                                           | 10            | GAP                                    |
| StructuredOutputManager / XGrammar                | 10            | GAP                                    |
| Quantization（AWQ/GPTQ/FP8/NVFP4）                | 11            | GAP                                    |
| Multimodal 输入 / mm_receiver_cache               | 12            | GAP                                    |
| LoRA（S-LoRA / Punica）                           | 13            | GAP                                    |
| Weight loading / 权重热更新 API                   | 14            | GAP（RL rollout 权重同步）           |
| Metrics / benchmark_serving                       | 15            | GAP（横切方法论）                     |

## 写作约定（每篇通用）

- 开篇用一个生活化类比锚定问题，正文保留英文术语。
- 采用「每暴露一个问题就引入一种优化」的问题→优化→新问题链条。
- 图表用 diagram-design 风格重绘（**中文标注**，英文技术术语保留），源文件置于 `diagrams/<topic>/`，导出 PNG 到 `public/assets/posts/<slug>/`。
- 实验一律无需 GPU、`SEED=42` 可复现，正文内联核心 Python，绘图代码从略。
- 结尾固定 `## 总结与延伸`（含「一句话带走」+「延伸阅读」）与正式的 `## 参考`（markdown 链接）。
