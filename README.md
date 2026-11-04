# Deusmith-MemorySystem

类人脑的智能体记忆管理数据库：**浅层工作记忆 / 中层情景语义记忆 / 深层核心记忆** 三层流转，
pgvector 语义召回 + 图游走双路检索，证据驱动的可信度与冲突仲裁，写读分离 + 离线治理。

- 后端：**PostgreSQL 17 + pgvector**（单库起步栈，不用 SQLite）
- Embedding：本地模型 / OpenAI 兼容 API / 零依赖词法哈希（离线保底）三选一
- LLM 挂钩：抽取/摘要/仲裁可用规则引擎离线跑，也可换 OpenAI 兼容 API
- 验证：单测 164 · DDL 真引擎 22/22 · 参数一致性 13/13 · 探针门禁 7/7

## 交付物索引

| 文件 | 内容 |
|---|---|
| `docs/00-review.md` | 对原方案的逐条评审（判定表 + 三个最贵错误 + 只做三件事） |
| `docs/01-architecture.md` | 架构：不变量 I1–I10、分层、数据模型、参数耦合、读写离线三路径、演进触发 |
| `docs/02-schema.sql` | PostgreSQL DDL（约束/索引/触发器/函数；与 `src/deusmith/schema.sql` 逐字节一致） |
| `docs/03-params.yaml` | 参数表（7 个 tunable 维度，`check_params.py` 保证与代码零漂移） |
| `docs/04-numeric-examples.md` | 数值演算（m102 勘误 +11.8%）· 衰减曲线 · PG 延迟实测 · 边界表 |
| `docs/eval-report.md` | 七类探针门禁 7/7 + 相似度判据坏变体对照（污染声明） |
| `assets/deusmith-architecture.html` | 架构图（主题变量 HTML） |
| `src/deusmith/` | 实现：`params` `models` `algos` `embeddings` `llm` `store` `shallow` `pipeline` `offline` |
| `tests/`（164 项） | 单测 + 真 PG 集成测试（无库自动 skip） |
| `scripts/` | `check_params` `check_sql` `eval_probes` `bench_pg` |
| `demo.py` | `--selftest`（25 项边界断言）/ `--demo`（全部数值表） |

## 快速开始

```bash
# 1. 环境：Python ≥3.11 + PostgreSQL ≥16（需 pgvector 扩展）
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .          # 或 pip install "psycopg[binary]"

# 2. 数据库（示例：本机 PG）
createdb deusmith
psql -d deusmith -c "CREATE EXTENSION vector"

# 3. 验证（全部应绿）
export DEUSMITH_PG_DSN="postgresql://postgres@127.0.0.1:5432/deusmith"
PYTHONPATH=src .venv/Scripts/python.exe -m unittest discover -s tests
PYTHONPATH=src .venv/Scripts/python.exe demo.py --selftest
PYTHONPATH=src .venv/Scripts/python.exe scripts/check_params.py
PYTHONPATH=src .venv/Scripts/python.exe scripts/check_sql.py      # 需可连 PG
PYTHONPATH=src .venv/Scripts/python.exe scripts/eval_probes.py   # 50 节点七类门禁
PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_pg.py       # 延迟基准（~1 分钟）

# 4. 演示
PYTHONPATH=src .venv/Scripts/python.exe demo.py --demo
```

无 PG 环境也能跑单测/演示（InMemoryStore 自动降级，PG 集成测试自动 skip）。

### 本机即用 PostgreSQL（免安装，Windows x64）

仓库 `pg/` 目录内置便携版 **PostgreSQL 17.6 + 编译好的 pgvector 0.8.7**（EDB 发行版 +
MSVC 编译，即本交付的验证环境），数据目录 `pg/pgdata`，仅监听 `127.0.0.1:55433`，trust 认证。
业务库 `deusmith` 的 schema 已由 `scripts/init_db.py` 初始化。

```bash
# 启动（建议在独立终端/后台运行，勿与会被超时杀进程树的调用混在一起）
pg/pgsql/bin/pg_ctl.exe -D "E:/ALL_FOR_MY_PROJECTS/Deusmith/Deusmith-MemorySystem/pg/pgdata" -o "-p 55433" -w -l pg/server.log start
# 停止
pg/pgsql/bin/pg_ctl.exe -D "E:/ALL_FOR_MY_PROJECTS/Deusmith/Deusmith-MemorySystem/pg/pgdata" -m fast stop
```

默认 DSN：`postgresql://postgres@127.0.0.1:55433/deusmith`。换自己的 PG：安装 PG≥16 +
pgvector 扩展，设 `DEUSMITH_PG_DSN` 后重跑 `scripts/init_db.py`。`pg/` 已 gitignore，勿入库。

## 使用示例

```python
from deusmith.embeddings import HashEmbedding
from deusmith.llm import RuleBasedLLM
from deusmith.pipeline import DeusmithMemory
from deusmith.store import InMemoryStore   # 生产换 PostgresStore(dsn=...)

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

# 会话结束：摘要 + 晋升评估 + 30 分钟后原文清理
mem.end_session(s)

# 离线治理（建议定时）：衰减/剪枝/环路/压缩/流转/审计
from deusmith.offline import OfflineGovernor
report = OfflineGovernor(mem.store, mem.embedder, mem.llm).run("tenant0/user1")
```

换真实嵌入/LLM：复制 `config.example.json` 为 `config.json`，
`embedding.provider=openai|local`、`llm.provider=openai`，密钥走
`DEUSMITH_EMBEDDING_API_KEY` / `DEUSMITH_LLM_API_KEY` 环境变量。

## 关键设计（详见 docs/00 评审与 docs/01 架构）

- **冲突判定用槽位身份+极性，不用相似度**：实测否定句（"我不再喜欢摄影了"）与原句
  余弦相似度 0.56–0.74，高于多数良性改写——任何能合并良性改写的相似度阈值必然吞掉
  否定更新；槽位+极性判定零丢失（docs/eval-report.md）。
- **绝对刻度打分**：RRF 融合除以"全通道第一"参考点，rel 与查询无关，
  "score<0.45 视为没记住"这类阈值可跨查询移植。
- **深层只可废止**：DB 触发器拦截 DELETE/内容 UPDATE，变更走版本链+审计；
  identity_policy 入深层必须 human_confirmed。
- **图游走有界**：度 20 的 20 万边图上，无界 2 跳 214ms vs 有界（100/跳）10ms。
- **TTL 是导出量**：归档期限 = -τ_eff·ln(0.2)，与连续衰减不双轨并存。

## 已知未验证项（上线前必看）

1. **没跑过的**：Redis / Neo4j / FalkorDB / MinIO / Qdrant 适配器（v1 决策不引入，
   仅留演进位）；`sentence-transformers` 本地模型与 OpenAI 兼容 API 路径未实测
   （接口有单测，网络路径未联网验证）；多用户并发与 Windows 以外平台。
2. **预算而非实测**：注入 Token 用 `estimate_tokens` 估算（CJK 1 token/字、ASCII 4 字符
   ≈1 token），不是真实 tokenizer；接真实模型前请按其 tokenizer 校准预算。
3. **参数未校准**：全部超参保持设计文档默认值，未经真实交互数据校准；
   探针集即开发集（曾用于修 bug，已声明污染），正式调参前需 70/30 切分另建留出集。

## 项目结构

```
Deusmith-MemorySystem/
├── docs/            00 评审 · 01 架构 · 02 schema · 03 参数 · 04 数值 · eval-report
├── src/deusmith/    params models algos embeddings llm store shallow pipeline offline + schema.sql
├── tests/           164 项（含真 PG 集成，无库自动 skip）
├── scripts/         check_params check_sql eval_probes bench_pg
├── assets/          架构图 HTML
├── demo.py          --selftest / --demo
└── config.example.json
```
