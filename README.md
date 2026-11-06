# Deusmith-MemorySystem

类人脑的智能体记忆管理数据库：**浅层工作记忆 / 中层情景语义记忆 / 深层核心记忆** 三层流转，
pgvector 语义召回 + 图游走双路检索 + 可选 cross-encoder 精排，证据驱动的可信度与冲突仲裁，
写读分离 + 离线治理，自带 CLI/REPL 控制台。

- 后端：**PostgreSQL 17 + pgvector**（单库起步栈；psycopg_pool 连接池默认开启）
- Embedding：本地模型（sentence-transformers / bge）/ OpenAI 兼容 API / 零依赖词法哈希 三选一
- Rerank：本地 cross-encoder（bge-reranker 系）/ OpenAI 兼容 /rerank API / 默认关闭 三选一
- LLM 挂钩：抽取/摘要/仲裁可用规则引擎离线跑（在线 LLM 失败自动降级），也可换 OpenAI 兼容 API
- 控制台：`deusmith` 命令（CLI 一次性命令 + 交互 REPL，零第三方依赖）

## 安装

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e .
# 本地语义模型（可选）：.venv/Scripts/python -m pip install sentence-transformers
```

需要 PostgreSQL ≥16 + pgvector 扩展（建库后 `CREATE EXTENSION vector`），DSN 走
`DEUSMITH_PG_DSN` 环境变量或 `config.json`（模板见 `config.example.json`）。
无 PG 环境时可用 `InMemoryStore` 跑通全流程。

## 快速开始

```python
from deusmith.embeddings import HashEmbedding          # 或 LocalModelEmbedding / OpenAICompatibleEmbedding
from deusmith.llm import RuleBasedLLM                  # 或 OpenAICompatibleLLM
from deusmith.pipeline import DeusmithMemory
from deusmith.store import InMemoryStore               # 生产换 PostgresStore(dsn=...)

mem = DeusmithMemory(store=InMemoryStore(), embedder=HashEmbedding(dim=256),
                     llm=RuleBasedLLM(), scope="tenant0/user1")

s = mem.session("chat-001")
mem.write(s, "请记住：我对花粉过敏。")
mem.write(s, "我喜欢摄影，周末常去拍夜景。")
mem.write(s, "我现在住在北京。")

out = mem.inject_context(s, "用户的居住地和爱好", token_budget=4000)
print(out["system"])   # 深层身份/策略（常驻）
print(out["user"])     # 相关记忆 + 冲突提示 + 最近会话投影
print(out["tokens"], out["degraded"])

mem.end_session(s)     # 摘要 + 晋升评估 + 会话持久化

from deusmith.offline import OfflineGovernor
report = OfflineGovernor(mem.store, mem.embedder, mem.llm).run("tenant0/user1")
# 离线治理：边衰减/剪枝/环路消解/压缩合并/层级流转/权重防垄断(Gini)/一致性审计
```

## 控制台（CLI / REPL / Web 可视化）

```bash
pip install -e .            # 提供 deusmith / deusmith-web 入口点（也可 python -m deusmith.console）

deusmith                    # 交互 REPL（无 config.json 时内存模式，零依赖可跑）
deusmith --memory write "请记住：我对花粉过敏。"      # 一次性命令
deusmith --scope tenant0/user1 search "过敏" -k 5    # PG 后端跨进程检索
deusmith web                # REPL 内启动可视化控制台（后台线程）
deusmith --web --port 9000  # 直接启动可视化控制台
deusmith-web                # 同上（独立入口点）
deusmith --smoke            # 自检（CI 用）
```

REPL 命令：`write` 写入 → `search` 检索（得分/rel/精排明细）→ `inject` 上下文注入
→ `tiers` 三层分布 → `node/slot/why/explain` 全参数与可解释性 → `stats` 基尼与事件
→ `forget <id>` 单条遗忘曲线与归档期限 → `govern` 离线治理 → `web` 可视化控制台
→ `config` 生效配置。配置文件：`--config` / `$DEUSMITH_CONFIG` / `./config.json`
（模板 `config.example.json`）。

**Web 可视化控制台**（`deusmith-web`，零第三方依赖，默认 `http://127.0.0.1:8765`）：

- 写入（来源权威可调）/ 检索（score·rel·精排明细）/ 注入预览（system·user·tool 三段 + 预算降级）
- 记忆浏览器：tier/status/文本过滤，节点全参数抽屉 + explain 因子分解
- 遗忘曲线：`exp(−Δt/τ_eff)` 实测曲线 + 冷/热参考 + θ 归档阈值 + 归档期限标记
- 记忆图谱：力导向布局（颜色=层级，大小=权重，边色=类型），点选高亮邻居
- 槽位版本链（双时间窗口）/ 事件流 / 离线治理报告 / scope 在线切换

## 能力一览

- **三层记忆**：浅层会话（滚动窗口+摘要，PG 持久化、重启恢复）→ 中层情景语义
  （pgvector HNSW + 边表图 + 时序版本链）→ 深层核心（版本链+审计，只可废止不可篡改）。
- **双路检索 + 可选精排**：HNSW 向量 + 有界 2 跳图游走 + 稀有词关键词通道，RRF 绝对刻度融合，
  可选 cross-encoder/API 精排（`rel' = (1-w)·rel_rrf + w·rerank_score`，失败自动降级回 RRF），
  MMR 多样性裁剪；常驻记忆单独成段注入，不挤占相关性排序。
- **冲突仲裁**：槽位身份 + 极性判定（否定句不会被相似度去重吞掉）；高权威纠正走
  supersedes 版本链 + 双时间窗口（valid_from/valid_to）；深层记忆永不静默覆盖。
- **证据与可解释性**：写入自动挂内容寻址证据（sha256 防篡改）；
  `explain_injection(trace_id)` / `why_selected(node_id)` / `explain_score(node_id)` /
  `slot_history(subject, predicate)`。
- **在线 API 健壮性**：指数退避重试、批量上限、成本计数；LLM 失败自动降级规则引擎。
- **权重防垄断**：离线治理对节点权重做惰性慢衰减（半衰期≈347 天，floor 0.2）+
  基尼系数监控告警。
- **遗忘曲线**：`recency = exp(-Δt/(τ_base·S))`，S 由 importance/emotion/confidence/activation
  拉伸；时效因子有界弱化，语义相关性主导排序。

## 关键设计

- **冲突判定用槽位身份+极性，不用相似度**：否定句（"我不再喜欢摄影了"）与原句余弦
  相似度 0.56–0.74，高于多数良性改写——任何能合并良性改写的相似度阈值必然吞掉否定
  更新；槽位+极性判定零丢失。多值谓语（喜欢）换宾语是两条并存记忆，单值谓语（住在）
  换宾语才是冲突。
- **绝对刻度打分**：RRF 融合除以"全通道第一"参考点，rel 与查询无关，绝对阈值可跨查询
  移植。
- **图游走必须有界**：度 20 的 20 万边图上，无界 2 跳 214ms vs 有界（100/跳）10ms。
- **TTL 是导出量**：归档期限 = −τ_eff·ln(θ)，与连续衰减不双轨并存。

## 换真实嵌入/LLM/重排

复制 `config.example.json` 为 `config.json`：`embedding.provider = openai | local`、
`rerank.provider = local | openai`、`llm.provider = openai`，密钥走
`DEUSMITH_EMBEDDING_API_KEY` / `DEUSMITH_RERANK_API_KEY` / `DEUSMITH_LLM_API_KEY`
环境变量。本地 bge 模型：`model_name` 指向模型目录（bge 系需带 `query_prefix`），实测
bge-small-zh-v1.5 相比词法哈希 dev 查询探针 0.571 → 0.714（+14.3pp）。重排模型同样
用户自行下载（如 `BAAI/bge-reranker-base`），默认关闭不影响既有检索行为。

## 项目结构

```
Deusmith-MemorySystem/
├── src/deusmith/     核心实现：params models algos embeddings rerank llm store
│                     shallow pipeline offline evidence retryhttp console
│                     webconsole + schema.sql + web/index.html（可视化前端）
├── config.example.json
├── pyproject.toml    # deusmith / deusmith-web 入口点
└── README.md
```

> 测试套件（219 项，含真 PG 集成）、验证脚本（schema/参数一致性/探针门禁/延迟基准）、
> CI 与设计文档不随包分发；完整版本封存于 git 标签 `v0.1.0` / `v0.2.0`。

## 已知边界

- 注入 Token 用估算器（CJK 1 token/字），接真实模型前按其 tokenizer 校准预算；
- 超参保持设计默认值，未经真实交互数据校准（校准协议与探针集见标签内 scripts/）；
- Redis / Neo4j / Qdrant / MinIO 为演进适配位，本版不引入（单库起步栈）。
