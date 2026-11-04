# Deusmith-MemorySystem · 架构（01）

类人脑分层记忆管理库：浅层工作记忆 / 中层情景语义记忆 / 深层核心记忆，
双路检索（pgvector HNSW + 图游走）、证据驱动的可信度、写读分离、离线治理。

> 红线：生物隐喻只提供曲线形状与变量选择，参数由评估指标决定（见 docs/eval-report.md）。

## 0. 不变量（违反任何一条 = 实现错误）

| # | 不变量 | 落点 |
|---|---|---|
| I1 | 缺 scope 的读写直接抛错 | `MemoryNode/MemoryEdge` 构造校验 + schema `CHECK (scope <> '')` |
| I2 | 深层只可废止不可删除/内容变更；变更走版本链+审计 | 触发器 `protect_deep` + `deprecate_deep()` + `memory_version`/`audit_log` |
| I3 | 冲突按槽位身份+极性判定；单值谓语换宾语=冲突，多值谓语只看同宾语极性翻转 | `algos.is_contradiction` + 冲突仲裁路径 |
| I4 | 打分因子 clip[0.01,1]；rel 用绝对参考点归一 | `retrieval_score` + `rrf_fuse` |
| I5 | 边强化饱和 w←1-(1-w)(1-Δ)，同 (src,dst,type) 合并幂等 | schema 触发器 `merge_edge`（客户端无法绕过） |
| I6 | importance/emotion/confidence/activation ↑ ⇒ τ_eff ↑ | `test_tau_eff_monotonic` |
| I7 | 注入 Token ≤ 预算、浅层 ≤30% 窗口；超窗走降级阶梯 | `inject_context` |
| I8 | 写入幂等：(scope, content_hash, embedding_model, active) 唯一；重复写转强化 | 部分唯一索引 + `find_active_by_hash` → `reinforce` |
| I9 | 图游走在同构 ID 空间：单表+类型判别 | schema 设计 |
| I10 | sensitivity 超过允许级别 → 门控过滤或脱敏，记 filtered_reason | 检索门控 + event_log |

## 1. 三条链路（对应 §五.8 调用顺序）

```
写入：解析抽取 → 可信度评估 → 幂等/冲突处理 → 节点落库 → 即时建边 → 晋升评估 → 日志
读取：查询 → 五路召回(向量/图/关键词/最近/显式) → 融合去重 → 绝对刻度打分 → MMR → 门控 → 注入
离线：边衰减重算 → 环路消解 → 压缩合并 → 层级流转 → 一致性校验 → 归档审计
```

## 2. 分层模型

| 层 | 定位 | 实现 | 容量 | 生命周期 |
|---|---|---|---|---|
| 浅层 | 当前会话工作记忆：最近 N=10 轮原文、工具结果、滚动摘要 | `shallow.ShallowSession`（进程内；Redis 为演进适配位） | ≤上下文窗口 30% | 会话结束 30 分钟后原文可清理；摘要进中层候选 |
| 中层 | 跨会话偏好/事实/事件/任务经验 | PG `memory_node` + `memory_edge`（pgvector HNSW + 边表图） | 20 万节点/用户量级（实测见 04§H） | recency 连续衰减 + 归档期限导出公式 + 离线压缩 |
| 深层 | 身份/价值观/安全策略/核心画像 | 同库 `tier='deep'` + `memory_version` 版本链 + 审计 | 1 万级 | 无 TTL；只可废止（I2），identity_policy 入层必须人工确认 |

层间流转（`algos.tier_transition`）：
- 浅→中：显式记住 ∨ 跨会话重复≥2 ∨ importance≥0.7
- 中→深：confidence≥0.9 ∧ source_auth≥0.8 ∧ 跨会话≥5；identity_policy 需 human_confirmed
- 归档：超过导出归档期限 `-τ_eff·ln(0.2)` ∧ weight<0.2 ∧ 非争议
- 深层退出：仅 `deprecate_deep()`（版本快照+审计+状态废止），或高置信冲突走人工仲裁

## 3. 数据模型（完整 DDL 见 docs/02-schema.sql，check_sql 22/22 真引擎验证）

**memory_node（单表+类型判别，I9）**：身份/内容、槽位（subject/predicate/object_value/polarity，
冲突判定前提）、时间（created/updated/valid_from/valid_to/last_access）、来源可信
（source/source_auth/confidence/consistency/evidence_refs）、权重使用（weight/activation_count/
importance/emotion）、治理（sensitivity/status/version/supersedes/contested_by/content_hash）、
流转统计（session_id/cross_session_count/explicit_remember/human_confirmed）、摘要谱系（generation）。
> decay_rate/ttl 不入库——由参数导出（归档期限公式），避免双轨不一致。

**memory_edge**：九种类型（semantic_similar/temporal_next/causal/co_occur/coref/supports/
contradicts/derives_from/summary_of），饱和合并由 DB 触发器强制（I5），UNIQUE(src,dst,type)。

**日志**：`event_log`（统一事件，trace_id 贯穿写入/检索/注入/冲突/流转）+
`audit_log`（深层变更/仲裁/废止/人工确认，只增不改）+ `memory_version`（深层版本快照）+
`evidence`（对象存储指针：本地目录或 s3:// URI）。

关键索引：HNSW(vector_cosine_ops)、(scope,status)、槽位部分索引、(scope,last_access DESC)、
幂等部分唯一索引、边 src/dst 部分索引。

## 4. 参数体系（docs/03-params.yaml，check_params 53 项零漂移）

耦合关系（全部有单测）：
```
activation_count↑ → weight↑ → score↑
last_access 越近 → recency↑ → score↑
importance/emotion/confidence↑ → S↑ → τ_eff↑ → 有效TTL↑
source_auth+consistency+evidence↑ → confidence↑（sigmoid，gain=3 修正）
conflict_penalty↑ → confidence↓ → 节点 contested
sensitivity↑ → 门控收紧 → 候选可能被过滤
```
可调维度收敛为 7 个（τ_base / 打分指数组 / 强化组 / MMR λ / 召回组 / 晋升阈值 / 聚类阈值），
其余为结构常数——<300 探针时更多自由参数必然过拟合。

## 5. 核心算法（六件套齐全；数值与边界见 docs/04）

| 算法 | 签名 | 复杂度 |
|---|---|---|
| 遗忘衰减 | `decay_recency(node, now)` = exp(-Δt/(τ_base·S)) | O(1) |
| 激活强化 | `reinforce(node, s, now)`：w←clip(w+η(1-w)s)，置信度随一致性上探 | O(1) |
| 层级流转 | `tier_transition(node, now)` → promote/archive/keep/needs_confirmation | O(1) |
| 链路更新 | `merge_edge_weight`（饱和）/`decay_edge_weight`/`should_prune_edge`/`resolve_cycles`（Tarjan） | O(E) |
| 可信度 | `estimate_confidence(node, evidence)` = sigmoid(k·(五因子)) | O(E) |
| 检索打分 | `retrieval_score(rel, node, now)` = rel^α·w^β·rec^γ·conf^δ·imp^ε·tier_boost·gate | O(1)/条 |
| 压缩合并 | `cluster_nodes`（type+scope+cos≥0.9 贪心聚类，generation 封顶防级联） | O(N·K) |

## 6. 读取路径细节（pipeline.retrieve）

1. **五路召回**：向量（HNSW top-50）/ 图（向量 top-5 种子 → 有界 2 跳 BFS，100/跳，
   walk 分 = max 0.85^hop·Πw）/ 关键词（CJK 二元组切词 + 拉丁词）/ 最近访问 / 显式+深层常驻；
2. **敏感门控**（I10）：sensitivity > allowed → 过滤 + filtered_reason；
3. **融合**：RRF（k=60）除以绝对参考点（全通道第一名）→ rel∈[0,1] 与查询无关（I4）；
4. **打分**：`retrieval_score`，每因子 clip[0.01,1]；
5. **MMR**：λ=0.7 多样性裁剪 topk=8；
6. **命中强化**：selected 节点 reinforce（TTL 滑动续期）；
7. **日志**：trace_id / recall 通道计数 / 候选与选中 / 分数 / 冲突标记 / 过滤原因。

**注入**（`inject_context`）：system 段=深层身份与策略（常驻）；user 段=相关记忆
（争议只注入冲突提示，按槽位去重）+ 浅层投影（≤30% 预算）；tool 段=证据指针。
超窗降级阶梯：裁低分候选 → 裁 tool → 裁浅层 → 请求用户分步检索（I7）。

## 7. 写入路径细节（pipeline.write）

抽取（RuleBasedLLM 正则 / OpenAI 兼容 LLM 可换）→ 构造节点（槽位+极性）→
幂等检查（同 hash 活跃 → reinforce，跨会话计数）→ 冲突检查（槽位身份，I3）→
仲裁（深层红线 → conflict_deep 审计；权威差距≥0.15 → supersedes 版本链；
否则双版本 contested）→ 落库 → 即时建边（kNN top-20 语义边 ≥0.3 + 同会话时序边）→
晋升评估（显式/重要偏好直接入中层）→ event_log。

## 8. 离线治理（offline.OfflineGovernor.run，建议每日或每 6 小时）

边衰减（λ_e=0.01/天）与三条件剪枝（w<0.1 ∧ 60 天未强化 ∧ conf<0.3）→
Tarjan 环路消解（去最弱边）→ 压缩合并（聚类≥2 → 摘要节点 generation+1、
summary_of 边、原文归档；摘要置信度按聚合证据数估计）→ 层级流转（含中→深晋升审计）→
深层一致性校验（无证据 → 审计告警）。

## 9. 可替换组件（用户前提：本地模型或 API 均可）

| 组件 | 默认 | 替换 |
|---|---|---|
| Embedding | `HashEmbedding`（零依赖词法占位，离线保底） | `OpenAICompatibleEmbedding`（/embeddings API，env `DEUSMITH_EMBEDDING_API_KEY`）/ `LocalModelEmbedding`（sentence-transformers，可选依赖） |
| LLM 抽取/摘要/仲裁 | `RuleBasedLLM`（正则+规则，离线保底） | `OpenAICompatibleLLM`（/chat/completions） |
| 存储 | `PostgresStore`（PG17+pgvector） | `InMemoryStore`（单测/演示） |
| 图库 | PG 边表 + 逐跳索引 BFS（实测 p50=10.4ms@20 万边） | Neo4j/FalkorDB 按演进触发引入 |

配置：`config.json`（见 config.example.json）+ env `DEUSMITH_PG_DSN`。

## 10. 评估闭环

- `scripts/eval_probes.py`：50 节点七类探针门禁（陈旧/纠正/连续/聚合/常驻/精度/改写）
  **7/7 PASS** + 相似度判据坏变体对照（否定更新丢失 5–6/8 @θ=0.5–0.6 vs 槽位判定 0/8）；
- `scripts/check_params.py` 13/13；`scripts/check_sql.py` 22/22（①真引擎）；
- 单测 164/164；`demo.py --selftest` 25/25；
- 污染声明与留出集要求见 docs/eval-report.md。

## 11. 演进触发（原计划 §8.5，保留）

| 触发条件 | 动作 |
|---|---|
| 向量 >5000 万 或过滤/延迟成瓶颈 | pgvector → Qdrant |
| 图查询复杂度/Cypher 生态需求上升 | PG 边表 → Neo4j/FalkorDB |
| 中层事实量级/冲突仲裁复杂度超出规则 | 自建管线 → Mem0/Graphiti（契约层不变） |
| 需要双时间推理（有效窗口 vs 摄取时间） | 引入 Graphiti |
| 多用户并发/分布式 | 浅层 Redis 适配、只读副本 |

## 12. 本次交付状态（对应原计划 §9.5 三阶段）

| 阶段 | 内容 | 状态 |
|---|---|---|
| 一 | PG+pgvector 存储、节点/边 Schema、向量检索与注入 | ✅ 完成+实测 |
| 二 | 衰减强化、可信度、冲突仲裁、图游走与 MMR | ✅ 完成+实测（图库以 PG 边表替代，见演进触发） |
| 三 | 离线压缩合并、层级流转、深层版本审计、可观测性、评估闭环 | ✅ 完成（评估闭环含七类门禁） |
