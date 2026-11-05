# Deusmith-MemorySystem

类人脑的智能体记忆管理数据库：**浅层工作记忆 / 中层情景语义记忆 / 深层核心记忆** 三层流转，
pgvector 语义召回 + 图游走双路检索，证据驱动的可信度与冲突仲裁，写读分离 + 离线治理。

- 后端：**PostgreSQL 17 + pgvector**（单库起步栈，不用 SQLite；psycopg_pool 连接池默认开启）
- Embedding：本地模型（sentence-transformers/bge）/ OpenAI 兼容 API / 零依赖词法哈希 三选一
- LLM 挂钩：抽取/摘要/仲裁可用规则引擎离线跑（在线 LLM 失败自动降级），也可换 OpenAI 兼容 API
- 验证：单测 216 · DDL 真引擎 22/22 · 参数一致性 13/13 · 探针门禁 7/7 · CI（GitHub Actions + pgvector 容器）

## 交付物索引

| 文件 | 内容 |
|---|---|
| `docs/00-review.md` | 对原方案的逐条评审（判定表 + 三个最贵错误 + 只做三件事） |
| `docs/01-architecture.md` | 架构：不变量 I1–I10、分层、数据模型、参数耦合、读写离线三路径、演进触发 |
| `docs/02-schema.sql` | PostgreSQL DDL（约束/索引/触发器/函数；与 `src/deusmith/schema.sql` 逐字节一致） |
| `docs/03-params.yaml` | 参数表（7 个 tunable 维度，`check_params.py` 保证与代码零漂移） |
| `docs/04-numeric-examples.md` | 数值演算（m102 勘误 +11.8%）· 衰减曲线 · PG 延迟实测 · 边界表 · 嵌入基线 |
| `docs/eval-report.md` | 七类探针门禁 7/7 + 查询探针（dev/held-out 切分）+ 坏变体对照 |
| `assets/deusmith-architecture.html` | 架构图（主题变量 HTML） |
| `src/deusmith/` | 实现：`params` `models` `algos` `embeddings` `llm` `store` `shallow` `pipeline` `offline` `evidence` `retryhttp` + schema.sql |
| `tests/`（216 项） | 单测 + 真 PG 集成（池化/会话/证据/时序/追踪）+ API mock 服务器 |
| `scripts/` | `check_params` `check_sql` `eval_probes` `calibrate` `bench_pg` `init_db` |
| `.github/workflows/ci.yml` | CI：pgvector 服务容器跑全量验证 |
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

本地语义模型（已验证路径）：`pip install sentence-transformers` 后配置
`embedding.provider=local`、`model_name` 指向模型目录（如项目内
`bge-small-zh-v1.5/`，bge 系需带 `query_prefix`）。实测
bge-small-zh-v1.5 相比词法哈希：dev 查询探针 0.571 → 0.714（+14.3pp，见 docs/04 §I）。

## 阶段二能力（v0.2.0）

- **连接池化**：`PostgresStore` 默认 psycopg_pool（1–8 连接），线程安全；未安装自动退回单连接。
- **浅层会话持久化**：`shallow_session`/`shallow_round` 表，进程重启后 `session(id)` 自动恢复（全量轮次落库，内存窗口 10）。
- **API 硬化**：embedding/LLM 调用带指数退避重试（5xx/429/网络错误）、批量上限、成本计数（`stats`）；写入路径 LLM 失败自动降级规则引擎（`llm_fallback` 事件留痕）。
- **双时间推理**：高权威纠正写 `valid_from/valid_to` 版本窗口；`slot_history()` 查时序历史（"三月住上海、五月搬北京"）；过期记忆检索降权。
- **证据存储**：`LocalEvidenceStore` 本地内容寻址（sha256 去重、verify 防篡改），写入路径自动挂 `evidence_refs`。
- **可解释性**：`explain_injection(trace_id)`（这次注入了什么）、`why_selected(node_id)`（为什么记住）、`explain_score(node_id)`（因子分解）。
- **抽取覆盖面**：中英文偏好/否定/居住/姓名/过敏/显式记住 + 通用槽位事实；英文 don't/can't 否定。
- **权重防垄断**：离线治理对节点权重做惰性慢衰减（λ_w=0.002/天，floor 0.2）+ 基尼系数监控（>0.8 告警），防"富者愈富"。
- **校准协议**：20 条查询探针 70/30 固定种子切分；`calibrate.py` 单维扫参 + held-out 终验（跑过即作废）。
- **v0.2.1 检索通道修正（bge 实测驱动）**：时效因子有界弱化 `(1-ρ)+ρ·rec^γ`；recent 通道移出融合（连续性由浅层投影承担）；keyword 通道只收稀有 term（df<10）；常驻记忆单独成段注入。修正后语义增益显现：dev 探针 hash 0.571 → bge 0.714（+14.3pp）。
- **v0.2.2 校准闭环**：`calibrate.py --embed local` 四维扫参（mmr/gamma/beta/topk）——mmr 默认 0.7 恰在平台起点；gamma 呈单调梯度（0.3→0.786，门禁无退化）记为候选方向；按防过拟合协议默认全保持（dev=14 分辨率 ±7.1pp）。

```python
# 可解释性示例
out = mem.inject_context(s, "用户的居住地", trace_id="tr_1")
print(mem.explain_injection("tr_1"))     # 注入内容/预算/降级
print(mem.why_selected(node_id))         # 哪些检索选中过它
print(mem.explain_score(node_id))        # weight/recency/conf 因子分解
print(mem.slot_history("user", "住在"))  # 时序版本链
```

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

1. **没跑过的**：Redis / Neo4j / FalkorDB / MinIO / Qdrant 适配器（v1 决策不引入，仅留演进位）；OpenAI 兼容 API 的真实外部端点（协议路径已用本地 mock 服务器测过，真实服务商未联网验证）；多用户并发生产负载与 Windows 以外平台的端到端（CI 已覆盖 ubuntu + pgvector 容器的测试矩阵）。
2. **预算而非实测**：注入 Token 用 `estimate_tokens` 估算（CJK 1 token/字、ASCII 4 字符 ≈1 token），不是真实 tokenizer；接真实模型前请按其 tokenizer 校准预算。
3. **参数未校准**：全部超参保持设计文档默认值。查询探针量化了 HashEmbedding 的语义缺口（dev 0.214），且参数扫描在 hash 下全线无提升——瓶颈在嵌入层；换真实模型后按 `calibrate.py` 协议重扫（dev 扫参、held-out 终验一次后作废）。探针集 dev 部分已声明污染，held-out 在 v0.2.0 消费过一次。

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
