-- ============================================================================
-- Deusmith-MemorySystem · PostgreSQL 17 + pgvector Schema
-- 单一事实源：src/deusmith/schema.sql（docs/02-schema.sql 为其副本，
--            scripts/check_sql.py 校验两者逐字节一致，防漂移）。
-- 占位符：__DIM__ 由初始化代码按 embedding 维度渲染（默认 256）。
-- 不变量落点：I1 scope 必填 / I2 深层不可删改 / I5 边饱和合并 /
--             I8 幂等写入 / I10 敏感分级（查询侧门控）。
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- 节点表（单表+类型判别，I9）
CREATE TABLE IF NOT EXISTS memory_node (
    id                  text PRIMARY KEY,
    type                text NOT NULL CHECK (type IN (
                            'event','fact','preference','emotion','summary',
                            'task','tool_result','identity_policy','conflict')),
    content             text NOT NULL CHECK (length(content) > 0),
    scope               text NOT NULL CHECK (scope <> ''),          -- I1
    tier                text NOT NULL CHECK (tier IN ('shallow','middle','deep')),
    content_vector      vector(__DIM__),
    embedding_model     text NOT NULL DEFAULT '',
    -- 槽位（冲突判定前提，评审错误#1）
    subject             text,
    predicate           text,
    object_value        text,
    polarity            smallint NOT NULL DEFAULT 0 CHECK (polarity IN (-1,0,1)),
    -- 时间
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),
    valid_from          timestamptz,
    valid_to            timestamptz,
    last_access_at      timestamptz NOT NULL DEFAULT now(),
    -- 来源与可信
    source              text NOT NULL DEFAULT 'user',
    source_auth         real NOT NULL DEFAULT 0.5 CHECK (source_auth BETWEEN 0 AND 1),
    confidence          real NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
    consistency         real NOT NULL DEFAULT 0.5 CHECK (consistency BETWEEN 0 AND 1),
    evidence_refs       text[] NOT NULL DEFAULT '{}',
    -- 权重与使用（decay_rate 不入库：由参数导出，评审 §四）
    weight              real NOT NULL DEFAULT 0.5 CHECK (weight BETWEEN 0 AND 1),
    activation_count    integer NOT NULL DEFAULT 0 CHECK (activation_count >= 0),
    importance          real NOT NULL DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),
    emotion             real NOT NULL DEFAULT 0.0 CHECK (emotion BETWEEN 0 AND 1),
    -- 治理
    sensitivity         smallint NOT NULL DEFAULT 1 CHECK (sensitivity BETWEEN 0 AND 3),
    status              text NOT NULL DEFAULT 'active' CHECK (status IN (
                            'active','archived','contested','deprecated','deleted_logical')),
    version             integer NOT NULL DEFAULT 1 CHECK (version >= 1),
    supersedes          text REFERENCES memory_node(id),
    contested_by        text,
    content_hash        text NOT NULL,
    -- 流转统计
    session_id          text,
    cross_session_count integer NOT NULL DEFAULT 0 CHECK (cross_session_count >= 0),
    explicit_remember   boolean NOT NULL DEFAULT false,
    human_confirmed     boolean NOT NULL DEFAULT false,
    -- 摘要谱系（防级联压缩）
    generation          integer NOT NULL DEFAULT 0 CHECK (generation >= 0),
    metadata            jsonb NOT NULL DEFAULT '{}'
);

-- 幂等写入（I8）：active/contested 状态下 (scope, hash, model) 唯一；归档/废止后可再写
CREATE UNIQUE INDEX IF NOT EXISTS uq_node_active_hash
    ON memory_node (scope, content_hash, embedding_model)
    WHERE status IN ('active','contested');

CREATE INDEX IF NOT EXISTS idx_node_scope_status   ON memory_node (scope, status);
CREATE INDEX IF NOT EXISTS idx_node_slot
    ON memory_node (scope, subject, predicate)
    WHERE subject IS NOT NULL AND status IN ('active','contested');
CREATE INDEX IF NOT EXISTS idx_node_last_access    ON memory_node (scope, last_access_at DESC);
CREATE INDEX IF NOT EXISTS idx_node_supersedes     ON memory_node (supersedes) WHERE supersedes IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_node_tier_active    ON memory_node (scope, tier) WHERE status = 'active';
-- 向量索引：HNSW 余弦
CREATE INDEX IF NOT EXISTS idx_node_vector
    ON memory_node USING hnsw (content_vector vector_cosine_ops);

-- ---------------------------------------------------------------- 边表
CREATE TABLE IF NOT EXISTS memory_edge (
    edge_id             text PRIMARY KEY,
    src_id              text NOT NULL REFERENCES memory_node(id) ON DELETE CASCADE,
    dst_id              text NOT NULL REFERENCES memory_node(id) ON DELETE CASCADE,
    type                text NOT NULL CHECK (type IN (
                            'semantic_similar','temporal_next','causal','co_occur',
                            'coref','supports','contradicts','derives_from','summary_of')),
    direction           text NOT NULL DEFAULT 'directed' CHECK (direction IN ('directed','undirected')),
    weight              real NOT NULL DEFAULT 0.5 CHECK (weight BETWEEN 0 AND 1),
    confidence          real NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
    created_at          timestamptz NOT NULL DEFAULT now(),
    last_reinforced_at  timestamptz NOT NULL DEFAULT now(),
    evidence_refs       text[] NOT NULL DEFAULT '{}',
    status              text NOT NULL DEFAULT 'active' CHECK (status IN ('active','weakened','pruned')),
    scope               text NOT NULL CHECK (scope <> ''),
    CONSTRAINT chk_no_self_loop CHECK (src_id <> dst_id),
    CONSTRAINT uq_edge_triple UNIQUE (src_id, dst_id, type)
);

CREATE INDEX IF NOT EXISTS idx_edge_src ON memory_edge (src_id) WHERE status <> 'pruned';
CREATE INDEX IF NOT EXISTS idx_edge_dst ON memory_edge (dst_id) WHERE status <> 'pruned';

-- ---------------------------------------------------------------- 日志表（统一事件 + 独立审计）
CREATE TABLE IF NOT EXISTS event_log (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts          timestamptz NOT NULL DEFAULT now(),
    trace_id    text,
    event_type  text NOT NULL,
    scope       text NOT NULL,
    payload     jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_event_trace  ON event_log (trace_id);
CREATE INDEX IF NOT EXISTS idx_event_type   ON event_log (event_type, ts DESC);

-- 审计（只增不改不删：深层变更/仲裁/废止/人工确认全量留痕）
CREATE TABLE IF NOT EXISTS audit_log (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts          timestamptz NOT NULL DEFAULT now(),
    action      text NOT NULL,
    node_id     text NOT NULL,
    actor       text NOT NULL DEFAULT 'system',
    detail      jsonb NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------- 深层版本表（I2）
CREATE TABLE IF NOT EXISTS memory_version (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    node_id     text NOT NULL,
    version     integer NOT NULL CHECK (version >= 1),
    snapshot    jsonb NOT NULL,
    valid_from  timestamptz NOT NULL DEFAULT now(),
    valid_to    timestamptz,
    created_by  text NOT NULL DEFAULT 'system',
    CONSTRAINT uq_version UNIQUE (node_id, version)
);

-- ---------------------------------------------------------------- 证据表（对象存储指针）
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id  text PRIMARY KEY,
    node_id      text NOT NULL,
    source       text NOT NULL,
    content_hash text NOT NULL,
    pointer_uri  text NOT NULL,           -- 本地目录或 s3://…/minio://…
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_evidence_node ON evidence (node_id);

-- ---------------------------------------------------------------- 浅层会话持久化（进程重启可恢复）
CREATE TABLE IF NOT EXISTS shallow_session (
    scope           text NOT NULL CHECK (scope <> ''),
    session_id      text NOT NULL,
    rolling_summary text NOT NULL DEFAULT '',
    created_at      timestamptz NOT NULL DEFAULT now(),
    last_active_at  timestamptz NOT NULL DEFAULT now(),
    closed_at       timestamptz,
    PRIMARY KEY (scope, session_id)
);
CREATE INDEX IF NOT EXISTS idx_ss_scope ON shallow_session (scope, last_active_at DESC);

CREATE TABLE IF NOT EXISTS shallow_round (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    scope       text NOT NULL,
    session_id  text NOT NULL,
    role        text NOT NULL CHECK (role IN ('user','assistant','tool')),
    content     text NOT NULL,
    ts          timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (scope, session_id)
        REFERENCES shallow_session (scope, session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sr_session ON shallow_round (scope, session_id, id);

-- ============================================================================
-- 触发器与函数
-- ============================================================================

-- updated_at 自动维护
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_touch ON memory_node;
CREATE TRIGGER trg_touch BEFORE UPDATE ON memory_node
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- I2：深层只可废止，不可物理删除/内容变更
CREATE OR REPLACE FUNCTION protect_deep() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' AND OLD.tier = 'deep' THEN
        RAISE EXCEPTION 'deep node % 不可删除（I2）：走 deprecate_deep()', OLD.id;
    END IF;
    IF TG_OP = 'UPDATE' AND OLD.tier = 'deep' THEN
        IF NEW.content IS DISTINCT FROM OLD.content
           OR NEW.scope IS DISTINCT FROM OLD.scope
           OR NEW.subject IS DISTINCT FROM OLD.subject
           OR NEW.predicate IS DISTINCT FROM OLD.predicate
           OR NEW.object_value IS DISTINCT FROM OLD.object_value
           OR NEW.polarity IS DISTINCT FROM OLD.polarity
           OR NEW.sensitivity IS DISTINCT FROM OLD.sensitivity
           OR NEW.type IS DISTINCT FROM OLD.type
           OR NEW.tier IS DISTINCT FROM OLD.tier
           OR (NEW.status NOT IN ('deprecated','contested')
               AND NEW.status IS DISTINCT FROM OLD.status)
        THEN
            RAISE EXCEPTION 'deep node % 内容不可更新（I2）：用新版本 supersede', OLD.id;
        END IF;
        RETURN NEW;
    END IF;
    RETURN COALESCE(NEW, OLD);
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_protect_deep ON memory_node;
CREATE TRIGGER trg_protect_deep BEFORE DELETE OR UPDATE ON memory_node
    FOR EACH ROW EXECUTE FUNCTION protect_deep();

-- 深层身份/策略硬门：必须人工确认（§二.3）
CREATE OR REPLACE FUNCTION deep_insert_guard() RETURNS trigger AS $$
BEGIN
    IF NEW.tier = 'deep' AND NEW.type = 'identity_policy' AND NOT NEW.human_confirmed THEN
        RAISE EXCEPTION 'identity_policy 进入深层必须 human_confirmed（§二.3）';
    END IF;
    RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_deep_guard ON memory_node;
CREATE TRIGGER trg_deep_guard BEFORE INSERT ON memory_node
    FOR EACH ROW EXECUTE FUNCTION deep_insert_guard();

-- supersedes 环检测（递归 CTE）
CREATE OR REPLACE FUNCTION check_supersedes_cycle() RETURNS trigger AS $$
DECLARE cnt integer;
BEGIN
    IF NEW.supersedes IS NULL THEN RETURN NEW; END IF;
    IF NEW.supersedes = NEW.id THEN
        RAISE EXCEPTION 'supersedes 不能指向自身';
    END IF;
    WITH RECURSIVE anc AS (
        SELECT id, supersedes FROM memory_node WHERE id = NEW.supersedes
        UNION ALL
        SELECT m.id, m.supersedes FROM memory_node m JOIN anc ON m.id = anc.supersedes
    )
    SELECT count(*) INTO cnt FROM anc WHERE id = NEW.id;
    IF cnt > 0 THEN
        RAISE EXCEPTION 'supersedes 环检测失败：%', NEW.id;
    END IF;
    RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_supersedes_cycle ON memory_node;
CREATE TRIGGER trg_supersedes_cycle BEFORE INSERT OR UPDATE OF supersedes ON memory_node
    FOR EACH ROW EXECUTE FUNCTION check_supersedes_cycle();

-- I5：边饱和合并 —— 同 (src,dst,type) 的插入自动合并，客户端无法绕过
CREATE OR REPLACE FUNCTION merge_edge() RETURNS trigger AS $$
DECLARE ex record;
BEGIN
    SELECT * INTO ex FROM memory_edge
     WHERE src_id = NEW.src_id AND dst_id = NEW.dst_id AND type = NEW.type;
    IF FOUND THEN
        UPDATE memory_edge SET
            weight = 1 - (1 - ex.weight) * (1 - NEW.weight),          -- 饱和强化
            confidence = greatest(ex.confidence, NEW.confidence),
            last_reinforced_at = now(),
            status = CASE WHEN ex.status = 'pruned' THEN 'active' ELSE ex.status END,
            evidence_refs = array(SELECT DISTINCT unnest(ex.evidence_refs || NEW.evidence_refs))
        WHERE edge_id = ex.edge_id;
        RETURN NULL;   -- 跳过插入
    END IF;
    RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_merge_edge ON memory_edge;
CREATE TRIGGER trg_merge_edge BEFORE INSERT ON memory_edge
    FOR EACH ROW EXECUTE FUNCTION merge_edge();

-- 深层废止函数（唯一合法的深层退出路径）：版本快照 + 状态废止 + 审计
CREATE OR REPLACE FUNCTION deprecate_deep(p_node_id text, p_reason text,
                                          p_actor text DEFAULT 'system') RETURNS void AS $$
DECLARE n memory_node%ROWTYPE;
BEGIN
    SELECT * INTO n FROM memory_node WHERE id = p_node_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION '节点 % 不存在', p_node_id;
    END IF;
    IF n.tier <> 'deep' THEN
        RAISE EXCEPTION 'deprecate_deep 只适用于深层节点，% 是 %', p_node_id, n.tier;
    END IF;
    INSERT INTO memory_version (node_id, version, snapshot, valid_to)
    VALUES (n.id, n.version, to_jsonb(n), now());
    UPDATE memory_node SET status = 'deprecated' WHERE id = p_node_id;
    INSERT INTO audit_log (action, node_id, actor, detail)
    VALUES ('deprecate_deep', p_node_id, p_actor, jsonb_build_object('reason', p_reason));
END $$ LANGUAGE plpgsql;
