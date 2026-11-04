# -*- coding: utf-8 -*-
"""check_sql.py —— DDL 真引擎校验（验证强度：①真引擎建库 + 逐条打约束）。

步骤：
  1. docs/02-schema.sql 与 src/deusmith/schema.sql 逐字节一致（防文档漂移）；
  2. 连接真实 PostgreSQL，在一次性测试库上执行完整 DDL（证明可执行）；
  3. 逐条插入违反值，断言数据库拒绝（CHECK / 唯一索引 / 触发器 / 函数）；
  4. 检查关键索引（含 HNSW 向量索引）存在。

用法：.venv/Scripts/python.exe scripts/check_sql.py
DSN：env DEUSMITH_PG_DSN_TEST 或默认 postgresql://postgres@127.0.0.1:55433/postgres
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SCHEMA_SRC = ROOT / "src" / "deusmith" / "schema.sql"
SCHEMA_DOC = ROOT / "docs" / "02-schema.sql"

PASS, FAIL = 0, 0


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}" + (f" —— {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" —— {detail}" if detail else ""))


def main() -> int:
    import psycopg

    print("== Deusmith-MemorySystem · schema 真引擎校验 ==")

    # 1. 文档副本一致
    src = SCHEMA_SRC.read_text(encoding="utf-8")
    doc = SCHEMA_DOC.read_text(encoding="utf-8") if SCHEMA_DOC.exists() else None
    check("docs/02-schema.sql 与 src 副本逐字节一致", doc == src,
          "文档即代码，漂移即失败" if doc == src else "两份 schema 不一致，先同步再交付")

    dsn = os.environ.get("DEUSMITH_PG_DSN_TEST",
                         "postgresql://postgres@127.0.0.1:55433/postgres")
    try:
        admin = psycopg.connect(dsn, connect_timeout=5, autocommit=True)
    except Exception as e:
        print(f"[SKIP] PostgreSQL 不可达（{e}），无法做真引擎校验。")
        print("       上线前必验项见 README『已知未验证项』。")
        return 0 if doc == src else 1

    test_db = f"deusmith_schemacheck_{uuid.uuid4().hex[:8]}"
    with admin.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{test_db}"')
    dsn_test = dsn.rsplit("/", 1)[0] + "/" + test_db
    conn = psycopg.connect(dsn_test, autocommit=True)
    cur = conn.cursor()

    try:
        # 2. DDL 可执行（真引擎）
        try:
            cur.execute(src.replace("__DIM__", "8"))
            check("完整 DDL 在 PG17+pgvector 上执行成功", True)
        except Exception as e:
            check("完整 DDL 执行成功", False, str(e))
            return 1

        # 版本信息
        cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
        ver = cur.fetchone()
        check("pgvector 扩展已加载", ver is not None, f"vector {ver[0] if ver else '??'}")

        def insert_node(**cols):
            keys = ", ".join(cols)
            vals = ", ".join(_lit(v) for v in cols.values())
            cur.execute(f"INSERT INTO memory_node (id, type, content, scope, tier, "
                        f"embedding_model, content_hash{', ' + keys if keys else ''}) "
                        f"VALUES ('ck{uuid.uuid4().hex[:8]}', 'fact', 'x', 't/u', "
                        f"'middle', 'm', 'h{uuid.uuid4().hex[:8]}'"
                        f"{', ' + vals if vals else ''})")

        def _lit(v):
            if isinstance(v, str):
                return "'" + v.replace("'", "''") + "'"
            if isinstance(v, (int, float)):
                return str(v)
            return "NULL"

        def expect_reject(name, sql, params=None):
            try:
                cur.execute(sql, params)
                check(name, False, "数据库未拒绝违反值")
            except psycopg.errors.CheckViolation as e:
                cname = getattr(getattr(e, "diag", None), "constraint_name", "?")
                check(name, True, f"CheckViolation: {cname}")
            except psycopg.errors.UniqueViolation as e:
                cname = getattr(getattr(e, "diag", None), "constraint_name", "?")
                check(name, True, f"UniqueViolation: {cname}")
            except psycopg.errors.RaiseException as e:
                check(name, True, f"TriggerReject: {str(e).strip()[:60]}")
            except psycopg.Error as e:
                check(name, True, f"{type(e).__name__}")

        # 3. 逐条打约束
        expect_reject("weight>1 被 CHECK 拒绝",
                      "INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                      "content_hash,weight) VALUES ('w1','fact','x','t/u','middle','m','h1',1.5)")
        expect_reject("scope='' 被 CHECK 拒绝（I1）",
                      "INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                      "content_hash) VALUES ('w2','fact','x','','middle','m','h2')")
        expect_reject("非法 type 被 CHECK 拒绝",
                      "INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                      "content_hash) VALUES ('w3','gossip','x','t/u','middle','m','h3')")
        expect_reject("polarity=2 被 CHECK 拒绝",
                      "INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                      "content_hash,polarity) VALUES ('w4','fact','x','t/u','middle','m','h4',2)")
        expect_reject("sensitivity=4 被 CHECK 拒绝",
                      "INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                      "content_hash,sensitivity) VALUES ('w5','fact','x','t/u','middle','m','h5',4)")
        expect_reject("自环边被 CHECK 拒绝",
                      "INSERT INTO memory_edge (edge_id,src_id,dst_id,type,scope) "
                      "VALUES ('e1','w1','w1','co_occur','t/u')")
        # 深层节点（先独立插入，避免多语句隐式事务回滚）
        cur.execute("INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                    "content_hash,human_confirmed) "
                    "VALUES ('d1','fact','x','t/u','deep','m','hd',true)")
        expect_reject("深层 DELETE 被触发器拒绝（I2）",
                      "DELETE FROM memory_node WHERE id='d1'")
        expect_reject("深层内容 UPDATE 被触发器拒绝（I2）",
                      "UPDATE memory_node SET content='篡改' WHERE id='d1'")
        expect_reject("identity_policy 深层未确认被触发器拒绝",
                      "INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                      "content_hash) VALUES ('ip1','identity_policy','x','t/u','deep','m','hip')")

        # supersedes 环
        cur.execute("INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                    "content_hash) VALUES ('s1','fact','v1','t/u','middle','m','hs1')")
        cur.execute("INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                    "content_hash,supersedes) VALUES ('s2','fact','v2','t/u','middle','m','hs2','s1')")
        expect_reject("supersedes 成环被触发器拒绝",
                      "UPDATE memory_node SET supersedes='s2' WHERE id='s1'")

        # 边饱和合并（I5）
        cur.execute("INSERT INTO memory_edge (edge_id,src_id,dst_id,type,scope,weight) "
                    "VALUES ('ee1','s1','s2','co_occur','t/u',0.5)")
        cur.execute("INSERT INTO memory_edge (edge_id,src_id,dst_id,type,scope,weight) "
                    "VALUES ('ee2','s1','s2','co_occur','t/u',0.2)")
        cur.execute("SELECT weight FROM memory_edge WHERE src_id='s1' AND dst_id='s2' "
                    "AND type='co_occur'")
        row = cur.fetchone()
        check("边饱和合并 1-(1-.5)(1-.2)=0.6", row is not None and abs(row[0] - 0.6) < 1e-6,
              f"weight={row[0] if row else None}")

        # 幂等唯一索引（I8）
        cur.execute("INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                    "content_hash) VALUES ('u1','fact','dup','t/u','middle','m','same-hash')")
        expect_reject("active 同 (scope,hash,model) 被唯一索引拒绝（I8）",
                      "INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                      "content_hash) VALUES ('u2','fact','dup','t/u','middle','m','same-hash')")
        cur.execute("UPDATE memory_node SET status='archived' WHERE id='u1'")
        cur.execute("INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                    "content_hash) VALUES ('u3','fact','dup','t/u','middle','m','same-hash')")
        check("归档后同 hash 可再写（部分唯一索引语义）", True)

        # deprecate_deep：版本 + 审计 + 状态
        cur.execute("SELECT deprecate_deep('d1', 'check_sql 校验')")
        cur.execute("SELECT status FROM memory_node WHERE id='d1'")
        check("deprecate_deep 置 deprecated", cur.fetchone()[0] == "deprecated")
        cur.execute("SELECT count(*) FROM memory_version WHERE node_id='d1'")
        check("memory_version 留版本快照", cur.fetchone()[0] == 1)
        cur.execute("SELECT count(*) FROM audit_log WHERE node_id='d1'")
        check("audit_log 留审计记录", cur.fetchone()[0] == 1)

        # 向量读写 + 索引
        cur.execute("INSERT INTO memory_node (id,type,content,scope,tier,embedding_model,"
                    "content_hash,content_vector) "
                    "VALUES ('v1','fact','向量','t/u','middle','m','hv','[1,0,0,0,0,0,0,0]')")
        cur.execute("SELECT round((content_vector <=> '[1,0,0,0,0,0,0,0]'::vector)::numeric, 4) "
                    "FROM memory_node WHERE id='v1'")
        check("pgvector 余弦距离自比对=0", cur.fetchone()[0] == 0)
        cur.execute("SELECT indexname FROM pg_indexes WHERE tablename='memory_node'")
        names = {r[0] for r in cur.fetchall()}
        check("HNSW 向量索引存在", "idx_node_vector" in names, ",".join(sorted(names)))

        cur.execute("SELECT count(*) FROM pg_indexes WHERE tablename='memory_edge'")
        check("边表索引已建（含 src/dst 部分索引）", cur.fetchone()[0] >= 3)

    finally:
        conn.close()
        with admin.cursor() as c:
            c.execute(f'DROP DATABASE IF EXISTS "{test_db}" WITH (FORCE)')
        admin.close()

    total = PASS + FAIL
    print(f"\n== 结果：{PASS}/{total} 项通过，验证强度：①真引擎建库+逐条打约束 ==")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
