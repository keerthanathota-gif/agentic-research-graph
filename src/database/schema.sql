
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";      -- Fuzzy string matching
CREATE EXTENSION IF NOT EXISTS "btree_gin";    -- Composite indexes

CREATE TABLE nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(50) NOT NULL,
    canonical_name TEXT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_node_type CHECK (type IN (
        'paper', 'concept', 'method', 'metric', 'dataset', 'tool', 'problem'
    ))
);

CREATE TABLE edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_relationship UNIQUE(source_id, target_id, type),

    CONSTRAINT no_self_loops CHECK (source_id != target_id)
);

CREATE TABLE entity_occurrences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    paper_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    local_name TEXT,
    usage_context TEXT,
    is_definition BOOLEAN DEFAULT FALSE,
    confidence NUMERIC(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE processing_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    stage VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'failed', 'in_progress')),
    duration_ms INTEGER,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE graph_statistics (
    date DATE PRIMARY KEY,
    papers_count INTEGER DEFAULT 0,
    entities_count INTEGER DEFAULT 0,
    relationships_count INTEGER DEFAULT 0,
    new_papers_today INTEGER DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE dead_letter_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id TEXT NOT NULL,
    failure_stage VARCHAR(50),
    error_type VARCHAR(50),
    error_message TEXT,
    attempt_count INTEGER,
    context JSONB,
    priority VARCHAR(20),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    retried_at TIMESTAMP
);

CREATE INDEX idx_nodes_type ON nodes(type);
CREATE INDEX idx_nodes_canonical_name ON nodes(canonical_name);
CREATE INDEX idx_nodes_type_name ON nodes(type, canonical_name);
CREATE INDEX idx_nodes_properties ON nodes USING GIN(properties);
CREATE INDEX idx_nodes_search ON nodes USING GIN(
    to_tsvector('english', canonical_name || ' ' || COALESCE((properties->>'definition')::text, ''))
);

CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_type ON edges(type);
CREATE INDEX idx_edges_source_type ON edges(source_id, type);
CREATE INDEX idx_edges_target_type ON edges(target_id, type);
CREATE INDEX idx_edges_composite ON edges(source_id, target_id, type);
CREATE INDEX idx_edges_properties ON edges USING GIN(properties);

CREATE INDEX idx_entity_occ_entity ON entity_occurrences(entity_id);
CREATE INDEX idx_entity_occ_paper ON entity_occurrences(paper_id);

CREATE INDEX idx_processing_log_paper ON processing_log(paper_id);
CREATE INDEX idx_processing_log_status ON processing_log(status);
CREATE INDEX idx_processing_log_stage ON processing_log(stage);

CREATE INDEX idx_dlq_paper ON dead_letter_queue(paper_id);
CREATE INDEX idx_dlq_error_type ON dead_letter_queue(error_type);

CREATE MATERIALIZED VIEW mv_paper_impact AS
SELECT
    p.id,
    p.canonical_name as title,
    p.properties->>'arxiv_id' as arxiv_id,
    (p.properties->>'publication_date')::timestamp as publication_date,
    COUNT(DISTINCT e_cited.id) as citation_count,
    COUNT(DISTINCT e_improved.id) as improved_by_count,
    COUNT(DISTINCT e_extended.id) as extended_by_count,
    COUNT(DISTINCT e_used.id) as used_by_count,
    COUNT(DISTINCT e_challenged.id) as challenged_by_count,
    (
        COUNT(DISTINCT e_cited.id) * 1.0 +
        COUNT(DISTINCT e_improved.id) * 2.0 +
        COUNT(DISTINCT e_extended.id) * 1.5
    ) as impact_score
FROM nodes p
LEFT JOIN edges e_cited ON e_cited.target_id = p.id AND e_cited.type = 'cites'
LEFT JOIN edges e_improved ON e_improved.target_id = p.id AND e_improved.type = 'improves_on'
LEFT JOIN edges e_extended ON e_extended.target_id = p.id AND e_extended.type = 'extends'
LEFT JOIN edges e_used ON e_used.target_id = p.id AND e_used.type = 'uses_method'
LEFT JOIN edges e_challenged ON e_challenged.target_id = p.id AND e_challenged.type = 'challenges'
WHERE p.type = 'paper'
GROUP BY p.id, p.canonical_name, p.properties;

CREATE UNIQUE INDEX ON mv_paper_impact(id);
CREATE INDEX ON mv_paper_impact(impact_score DESC);
CREATE INDEX ON mv_paper_impact(publication_date DESC);

CREATE MATERIALIZED VIEW mv_entity_popularity AS
SELECT
    e.id,
    e.type,
    e.canonical_name,
    COUNT(DISTINCT eo.paper_id) as papers_mentioned_in,
    COUNT(DISTINCT CASE WHEN edge.type = 'uses_method' THEN edge.source_id END) as used_by_count,
    COUNT(DISTINCT CASE WHEN edge.type = 'improves_on' THEN edge.source_id END) as improved_by_count,
    (e.properties->>'occurrence_count')::integer as total_mentions
FROM nodes e
LEFT JOIN entity_occurrences eo ON eo.entity_id = e.id
LEFT JOIN edges edge ON edge.target_id = e.id
WHERE e.type IN ('concept', 'method', 'metric', 'dataset')
GROUP BY e.id, e.type, e.canonical_name, e.properties;

CREATE UNIQUE INDEX ON mv_entity_popularity(id);
CREATE INDEX ON mv_entity_popularity(type, papers_mentioned_in DESC);

CREATE OR REPLACE FUNCTION find_papers_that_improve(method_name TEXT)
RETURNS TABLE (
    paper_id UUID,
    paper_title TEXT,
    arxiv_id TEXT,
    improvement_claim TEXT,
    confidence NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id,
        p.canonical_name,
        p.properties->>'arxiv_id',
        e.properties->'quantitative'->>'improvement',
        (e.properties->>'confidence')::numeric
    FROM nodes m
    JOIN edges e ON e.target_id = m.id AND e.type = 'improves_on'
    JOIN nodes p ON p.id = e.source_id AND p.type = 'paper'
    WHERE m.type = 'method'
        AND m.canonical_name ILIKE '%' || method_name || '%'
    ORDER BY (e.properties->>'confidence')::numeric DESC;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION find_method_lineage(method_name TEXT, max_depth INT DEFAULT 5)
RETURNS TABLE (
    depth INT,
    method_id UUID,
    method_name TEXT,
    relationship_type TEXT,
    improvement_claim TEXT
) AS $$
WITH RECURSIVE lineage AS (
    SELECT
        0 as depth,
        m.id,
        m.canonical_name,
        ''::TEXT as rel_type,
        ''::TEXT as improvement
    FROM nodes m
    WHERE m.type = 'method'
        AND m.canonical_name ILIKE '%' || method_name || '%'

    UNION ALL

    SELECT
        l.depth + 1,
        m.id,
        m.canonical_name,
        e.type,
        COALESCE(e.properties->'quantitative'->>'improvement', 'N/A')
    FROM lineage l
    JOIN edges e ON e.target_id = l.id
        AND e.type IN ('improves_on', 'extends', 'builds_on')
    JOIN nodes m ON m.id = e.source_id AND m.type = 'method'
    WHERE l.depth < max_depth
)
SELECT * FROM lineage ORDER BY depth;
$$ LANGUAGE sql;

CREATE OR REPLACE FUNCTION search_entities(
    search_query TEXT,
    entity_type TEXT DEFAULT NULL,
    limit_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    type VARCHAR,
    name TEXT,
    relevance REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        n.id,
        n.type,
        n.canonical_name,
        ts_rank(
            to_tsvector('english', n.canonical_name || ' ' || COALESCE((n.properties->>'definition')::text, '')),
            plainto_tsquery('english', search_query)
        ) as relevance
    FROM nodes n
    WHERE
        (entity_type IS NULL OR n.type = entity_type)
        AND to_tsvector('english', n.canonical_name || ' ' || COALESCE((n.properties->>'definition')::text, ''))
            @@ plainto_tsquery('english', search_query)
    ORDER BY relevance DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_nodes_updated_at
    BEFORE UPDATE ON nodes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_edges_updated_at
    BEFORE UPDATE ON edges
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

INSERT INTO graph_statistics (date) VALUES (CURRENT_DATE)
ON CONFLICT (date) DO NOTHING;

COMMENT ON TABLE nodes IS 'All graph entities: papers, concepts, methods, metrics, datasets';
COMMENT ON TABLE edges IS 'All relationships between nodes with typed edges';
COMMENT ON TABLE entity_occurrences IS 'Tracks which papers mention which entities';
COMMENT ON TABLE processing_log IS 'Audit log of paper processing pipeline';
COMMENT ON TABLE graph_statistics IS 'Daily statistics for monitoring';
COMMENT ON TABLE dead_letter_queue IS 'Failed processing jobs for manual review';

COMMENT ON COLUMN nodes.properties IS 'Flexible JSONB storage for entity-specific attributes';
COMMENT ON COLUMN edges.properties IS 'Relationship metadata including evidence, confidence, quantitative data';
