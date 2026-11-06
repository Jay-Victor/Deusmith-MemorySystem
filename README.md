# Deusmith-MemorySystem

> 类人脑的智能体记忆系统 —— 三层记忆流转 · pgvector 语义检索 · 遗忘曲线 · 冲突仲裁

为 AI Agent 设计的持久记忆层：把「对话里说过什么」沉淀为可检索、可遗忘、可仲裁、可解释的结构化记忆。
后端单库起步（PostgreSQL 17 + pgvector），模型全部可插拔（embedding / rerank / LLM 均支持本地模型或 API），自带 CLI / REPL / TUI / Web 四种控制台形态。

## 核心能力

| # | 设计目标 | 实现 |
|---|---------|------|
| 1 | 控制台 | **CLI**（一次性命令）、**REPL**（交互终端）、**TUI**（textual 终端界面）、**Web 可视化**（浏览器仪表盘）四形态 |
| 2 | 向量库 + 用户自配模型 | pgvector HNSW 语义检索；embedding（本地 / OpenAI 兼容 / 词法哈希兜底）与 rerank（本地 cross-encoder / API）由用户自行下载配置 |
| 3 | 多层记忆 + 层间特殊处理 | 浅层会话 → 中层情景语义 → 深层核心；晋升有门槛、冲突不越级、深层只可废止不可篡改、压缩合并、过期归档 |
| 4 | 多参数管理 | 时间五元组（created / updated / valid_from / valid_to / last_access）+ weight / confidence / importance / emotion / sensitivity 等 20 余参数全量入库 |
| 5 | 参数互相影响 + 遗忘机制 | S ← importance·emotion·confidence·activation → τ_eff → recency → 检索分；归档期限为导出量；边衰减剪枝、权重慢衰减、基尼垄断监控 |

## 安装

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e .

# 可选 extras
.venv/Scripts/python -m pip install -e ".[local]"   # sentence-transformers：本地嵌入/重排模型
.venv/Scripts/python -m pip install -e ".[tui]"     # textual：终端 TUI
```

安装后获得三个入口点：`deusmith`（CLI/REPL）、`deusmith-web`（Web 控制台）、`deusmith-tui`（TUI）。

**后端**：生产用 PostgreSQL ≥16 + pgvector（建库后 `CREATE EXTENSION vector`，或直接
`python scripts/init_db.py` 按嵌入维度建表）。无 PG 环境时内存模式可跑通全部功能（数据不持久化）。

## 快速开始

### 30 秒体验（零配置、零依赖）

```bash
deusmith                          # 交互 REPL：write / search / inject / tiers / forget / govern …
deusmith --memory write "请记住：我对花粉过敏。"
deusmith --memory search "过敏"
```

### 作为 Python 库

```python
from deusmith.embeddings import HashEmbedding      # 或 LocalModelEmbedding / OpenAICompatibleEmbedding
from deusmith.llm import RuleBasedLLM              # 或 OpenAICompatibleLLM
from deusmith.pipeline import DeusmithMemory
from deusmith.store import InMemoryStore           # 生产换 PostgresStore(dsn=...)

mem = DeusmithMemory(store=InMemoryStore(), embedder=HashEmbedding(dim=256),
                     llm=RuleBasedLLM(), scope="tenant0/user1")

s = mem.session("chat-001")
mem.write(s, "请记住：我对花粉过敏。")
mem.write(s, "我喜欢摄影，周末常去拍夜景。")
mem.write(s, "我现在住在北京。")

out = mem.inject_context(s, "用户的居住地和爱好", token_budget=4000)
print(out["system"])   # 深层身份/策略（常驻段）
print(out["user"])     # 相关记忆 + 冲突提示 + 最近会话投影
print(out["tokens"], out["degraded"])

mem.end_session(s)     # 摘要 + 晋升评估 + 会话持久化

from deusmith.offline import OfflineGovernor
report = OfflineGovernor(mem.store, mem.embedder, mem.llm).run("tenant0/user1")
# 离线治理：边衰减/剪枝/环路消解/压缩合并/层级流转/权重防垄断(Gini)/一致性审计
```

## 控制台（四种形态）

| 形态 | 启动 | 适用 |
|------|------|------|
| CLI | `deusmith --scope t/u write "…"` / `search` / `tiers` … | 脚本化、管道、cron |
| REPL | `deusmith` | 终端交互调试 |
| TUI | `deusmith-tui` 或 `deusmith --tui` | 终端内的图形界面（需 `[tui]` extra） |
| Web | `deusmith-web` 或 `deusmith --web`，REPL 内 `web` 命令 | 浏览器可视化仪表盘 |

- **REPL 命令**：`write` → `search`（得分/rel/精排明细）→ `inject` → `tiers`（层×状态分布）
  → `node` / `slot` / `why` / `explain`（全参数与可解释性）→ `stats`（基尼与事件）
  → `forget <id>`（单条遗忘曲线与归档期限）→ `govern` → `web` → `config`。
- **TUI 五标签页**（Ctrl+Q 退出）：① 写入/检索/注入预览 ② 记忆浏览器（tier/status/文本
  过滤 + 行选全参数详情）③ 遗忘曲线（本节点/冷/热参考三条曲线 + 归档期限）④ 事件流
  ⑤ 离线治理（线程 worker，不冻结 UI）。
- **Web 控制台**（默认 `http://127.0.0.1:8765`，零第三方依赖）：写入（来源权威可调）/
  检索/注入预览三段、记忆浏览器（全参数抽屉 + explain 因子条）、遗忘曲线
  （θ 阈值线 + 当前 Δt + 归档期限标记）、记忆图谱（力导向布局，颜色=层级、大小=权重、
  边色=类型，点选高亮邻居）、槽位版本链（双时间窗口）、事件流、治理报告、scope 在线切换。

配置文件定位：`--config` > `$DEUSMITH_CONFIG` > `./config.json`；都没有则内存模式。

## 配置

复制 `config.example.json` 为 `config.json`（已被 .gitignore 防护，可安全存放密钥）：

| 节 | 键 | 说明 |
|----|----|------|
| `storage` | `backend` / `dsn` | `postgres`（默认）或 `memory`；DSN 也可走 `$DEUSMITH_PG_DSN` |
| `embedding` | `provider` | `hash`（零依赖兜底）/ `local`（sentence-transformers）/ `openai`（兼容 /embeddings） |
| `embedding` | `model_name` / `query_prefix` | local 模式指向模型目录或 HF ID；bge 系检索模型必须带查询前缀 |
| `rerank` | `provider` | `none`（默认关闭）/ `local`（cross-encoder，如 BAAI/bge-reranker-base）/ `openai`（兼容 /rerank） |
| `llm` | `provider` | `rule`（离线规则引擎兜底）/ `openai`（兼容 /chat/completions） |

环境变量：`DEUSMITH_PG_DSN`、`DEUSMITH_CONFIG`、`DEUSMITH_EMBEDDING_API_KEY`、
`DEUSMITH_RERANK_API_KEY`、`DEUSMITH_LLM_API_KEY`。

- **嵌入与重排模型均由用户自行下载**（如 `BAAI/bge-small-zh-v1.5`、`BAAI/bge-reranker-base`），
  `model_name` 支持本地目录或 HF ID；实测 bge-small-zh-v1.5 相比词法哈希，dev 查询探针
  0.571 → 0.714（+14.3pp）。
- **换嵌入模型需重建表**：`memory_node.content_vector` 的维度必须与模型输出一致
  （bge 系 512 维 vs 哈希 256 维）；控制台启动时会做维度守卫并明确报错。
- 在线组件失败自动降级：LLM 失败 → 规则引擎；rerank 失败 → RRF 排序（均留事件痕）。

## 架构

**三层记忆**：

| 层 | 内容 | 存储 | 淘汰 |
|----|------|------|------|
| 浅层（工作记忆） | 当前会话滚动窗口 + 摘要 | PG 持久化、重启恢复 | 会话结束 30 分钟后清理原文 |
| 中层（情景语义） | 事件/事实/偏好节点 + 关系边 | pgvector HNSW + 边表图 + 版本链 | 遗忘曲线归档、压缩合并 |
| 深层（核心记忆） | 身份/价值观/策略 | 版本链 + 审计 | 只可废止（supersede），不可篡改（DB 触发器强制） |

**三条路径**：

- **写入**：抽取（LLM/规则）→ 可信度评估 → 幂等/冲突仲裁（槽位+极性，supersedes 双时间窗口）→
  分层落库 → 即时建边（kNN 语义 + 同会话时序）→ 晋升评估。
- **读取**：查询 → 多路召回（HNSW 向量 / 有界 2 跳图游走 / 稀有词关键词）→ RRF 绝对刻度融合 →
  可选 rerank 精排（`rel' = (1-w)·rel_rrf + w·rerank_score`）→ MMR 多样性裁剪 → 命中强化 →
  上下文注入（常驻/相关/浅层投影三段 + 超预算降级阶梯）。
- **离线治理**（每日/每 6 小时）：边衰减与剪枝 → 环路消解（Tarjan）→ 压缩合并（聚类+摘要，
  防级联）→ 节点权重慢衰减（防富者愈富）→ 层级流转 → 一致性审计 + 基尼告警。

## 设计决策

1. **冲突判定用槽位身份 + 极性，不用相似度**：否定句（"我不再喜欢摄影了"）与原句余弦相似度
   0.56–0.74，高于多数良性改写——任何能合并良性改写的相似度阈值必然吞掉否定更新；
   槽位+极性判定零丢失。多值谓语（喜欢）换宾语是两条并存记忆，单值谓语（住在）换宾语才是冲突。
2. **绝对刻度打分**：RRF 融合除以「全通道第一」参考点，rel 与查询措辞无关，绝对阈值可跨查询移植。
3. **TTL 是导出量**：归档期限 = −τ_eff·ln(θ)，与连续衰减不双轨并存。
4. **图游走必须有界**：度 20 的 20 万边图上，无界 2 跳 214ms vs 有界（100/跳）10ms。
5. **时效因子有界弱化**：`(1-s) + s·rec^γ`——遗忘管"忘没忘"，不管"排不排序"，
   压排序的永远是语义相关性。
6. **权重防垄断**：检索命中只升不降会让 weight 饱和到 1.0 失去区分度；慢时钟衰减
   （半衰期≈347 天，floor 0.2）与 recency 快时钟分工，另以基尼系数监控告警。
7. **可解释性内建**：`explain_injection(trace_id)` / `why_selected(node_id)` /
   `explain_score(node_id)`（因子分解）/ `slot_history(subject, predicate)`（版本链）。

## 验证与质量

| 命令 | 覆盖 | 状态 |
|------|------|------|
| `python demo.py --selftest` | 算法/边界断言 | 25 项 |
| `deusmith --smoke` | CLI/REPL + rerank 混合/降级 | 18 项 |
| `deusmith-web --smoke` | Web API 全端点 | 24 项 |
| `deusmith-tui --smoke` | TUI 五标签页（无头） | 16 项 |
| `python scripts/check_params.py` | 参数表 ↔ 代码默认值一致 | 13 项 |
| `python scripts/check_sql.py` | DDL 真引擎建库 + 逐条打约束 | 22 项 |
| `python scripts/eval_probes.py` | 七类门禁 + 查询探针（dev/held-out 70/30） | 门禁 7 类 |

`scripts/` 另含 `bench_pg.py`（检索延迟基准）、`calibrate.py`（单维扫参协议）、
`init_db.py`（按嵌入维度建表）。CI（py3.11/3.14 矩阵）跑模块导入 + 全部冒烟 + 端到端 + 离线治理。

## 项目结构

```
Deusmith-MemorySystem/
├── src/deusmith/
│   ├── params.py        参数体系（单一事实源，docs/03 镜像）
│   ├── models.py        节点/边数据模型（槽位、双时间、敏感分级）
│   ├── algos.py         遗忘/强化/打分/RRF/MMR/聚类 纯函数
│   ├── embeddings.py    嵌入三 provider（hash/local/openai）
│   ├── rerank.py        精排三 provider（none/local/openai）
│   ├── llm.py           LLM 挂钩（规则引擎兜底 + OpenAI 兼容）
│   ├── store.py         InMemoryStore / PostgresStore（同契约）
│   ├── shallow.py       浅层会话（滚动窗口+摘要）
│   ├── pipeline.py      写入/检索/注入管线 + 可解释性
│   ├── offline.py       离线治理
│   ├── evidence.py      内容寻址证据存储
│   ├── retryhttp.py     在线 API 重试/退避
│   ├── console.py       CLI/REPL ·  webconsole.py + web/  Web ·  tui.py  TUI
│   └── schema.sql       PG DDL（触发器强制不变量）
├── scripts/             验证/校准/基准/建库脚本
├── docs/                schema 副本 / 参数表 / 数值演算
├── config.example.json
└── pyproject.toml
```

> 完整测试套件（219 项，含真 PG 集成）与设计过程文档封存于 git 标签 `v0.1.0` / `v0.2.0`。

## 已知边界

- 注入 Token 用估算器（CJK 1 token/字），接真实模型前按其 tokenizer 校准预算；
- 超参保持设计默认值，未经真实交互数据校准（校准协议见 `scripts/calibrate.py`）；
- Redis / Neo4j / Qdrant / MinIO 为演进适配位，本版不引入（单库起步栈）。
