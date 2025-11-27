# Graph Schema Design

## Table of Contents
1. [Overview](#overview)
2. [Node Types](#node-types)
3. [Edge Types](#edge-types)
4. [PostgreSQL Schema](#postgresql-schema)
5. [Query Patterns](#query-patterns)
6. [Semantic Encoding](#semantic-encoding)
7. [Schema Evolution Strategy](#schema-evolution-strategy)

---

## Overview

### Graph Model Philosophy

The knowledge graph uses a **property graph model** implemented in PostgreSQL. This design choice balances:
- **Expressiveness**: Rich properties on nodes and edges
- **Query power**: SQL + recursive CTEs for graph traversal
- **Operational simplicity**: Standard RDBMS operations
- **Performance**: Indexed queries, materialized views

### Core Design Decisions

**1. Hybrid Relational-Graph Model**
- Nodes and edges stored in tables with JSONB properties
- Relationships as foreign keys enable JOIN-based traversal
- Recursive CTEs for multi-hop queries

**2. Property-Rich Entities**
- Every node/edge has flexible JSONB `properties` column
- Enables schema evolution without migrations
- Stores provenance, confidence, evidence

**3. Type-Based Polymorphism**
- Single `nodes` table with `type` discriminator
- Enables cross-type queries ("find all entities mentioned in paper X")
- Type-specific properties in JSONB

---

## Node Types

### 1. Paper Node

**Represents**: Academic papers in the corpus

```typescript
interface PaperNode {
  id: UUID;
  type: 'paper';
  canonical_name: string;  // Title
  properties: {
    arxiv_id: string;
    doi?: string;
    title: string;
    authors: Author[];
    abstract: string;
    publication_date: Date;
    categories: string[];          // ['cs.CV', 'cs.LG']
    venue?: string;                // 'CVPR 2023', 'arXiv'
    version: number;               // arXiv version
    pdf_url: string;
    source_repo: 'arxiv' | 'openreview' | 'semantic_scholar';

    // Quality metrics
    quality_score: number;         // 0-1, from QC agent
    completeness: number;          // 0-1, extraction completeness
    confidence: number;            // 0-1, overall extraction confidence

    // Statistics
    citation_count: number;
    reference_count: number;
    entity_count: number;
    relationship_count: number;

    // Processing metadata
    processed_at: Date;
    processing_duration_ms: number;
    extraction_version: string;    // Version of extraction system
  };
  created_at: Date;
  updated_at: Date;
}

interface Author {
  name: string;
  affiliation?: string;
  email?: string;
  orcid?: string;
}
```

**Example**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "paper",
  "canonical_name": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
  "properties": {
    "arxiv_id": "2308.04079",
    "title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
    "authors": [
      {"name": "Bernhard Kerbl", "affiliation": "Inria"},
      {"name": "Georgios Kopanas", "affiliation": "Inria"}
    ],
    "abstract": "Radiance Field methods have recently...",
    "publication_date": "2023-08-07T00:00:00Z",
    "categories": ["cs.CV", "cs.GR"],
    "venue": "ACM TOG (SIGGRAPH 2023)",
    "quality_score": 0.92,
    "completeness": 0.89,
    "citation_count": 247,
    "entity_count": 47,
    "relationship_count": 34
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:32:04Z"
}
```

---

### 2. Concept Node

**Represents**: High-level research concepts and ideas

```typescript
interface ConceptNode {
  id: UUID;
  type: 'concept';
  canonical_name: string;          // "Neural Radiance Fields"
  properties: {
    aliases: string[];             // ["NeRF", "neural radiance field"]
    category: 'technique' | 'problem' | 'application' | 'theory';
    definition?: string;
    definition_paper_id?: UUID;    // Paper that introduced this concept
    definition_source?: string;    // Section where defined

    // Taxonomy
    parent_concepts?: string[];    // Broader concepts
    child_concepts?: string[];     // More specific concepts

    // Usage statistics
    occurrence_count: number;      // How many papers mention this
    first_seen_date: Date;         // When first appeared in corpus
    popularity_trend: number[];    // Monthly mention counts

    // Semantic
    embedding?: number[];          // Vector for similarity search
    related_concepts?: string[];   // Semantically similar
  };
  created_at: Date;
  updated_at: Date;
}
```

**Example**:
```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "type": "concept",
  "canonical_name": "Neural Radiance Fields",
  "properties": {
    "aliases": ["NeRF", "NeRFs", "neural radiance field"],
    "category": "technique",
    "definition": "Continuous volumetric scene representations encoded in neural network weights",
    "definition_paper_id": "paper_nerf_original_id",
    "parent_concepts": ["neural scene representation", "3D reconstruction"],
    "child_concepts": ["Instant-NGP", "Mip-NeRF", "Block-NeRF"],
    "occurrence_count": 1247,
    "first_seen_date": "2020-03-01T00:00:00Z",
    "related_concepts": ["Signed Distance Functions", "Volumetric Rendering"]
  }
}
```

---

### 3. Method Node

**Represents**: Specific algorithms, techniques, architectures

```typescript
interface MethodNode {
  id: UUID;
  type: 'method';
  canonical_name: string;
  properties: {
    aliases: string[];
    method_type: 'algorithm' | 'architecture' | 'training_procedure' | 'optimization' | 'representation';
    category: 'proposed' | 'adapted' | 'baseline';

    // Technical details
    description?: string;
    parameters?: Record<string, any>;
    input_output?: {
      input: string;
      output: string;
    };

    // Origin
    introduced_in_paper_id?: UUID;
    introduced_date?: Date;
    based_on?: string[];          // Methods this builds upon

    // Usage
    occurrence_count: number;
    used_as_baseline_count: number;
    improved_by_count: number;

    // Performance (if reported consistently)
    typical_metrics?: Record<string, number>;
  };
  created_at: Date;
  updated_at: Date;
}
```

**Example**:
```json
{
  "id": "770e8400-e29b-41d4-a716-446655440002",
  "type": "method",
  "canonical_name": "3D Gaussian Splatting",
  "properties": {
    "aliases": ["3D-GS", "3DGS", "Gaussian Splatting"],
    "method_type": "representation",
    "category": "proposed",
    "description": "Represents 3D scenes as explicit anisotropic 3D Gaussians with learnable parameters",
    "parameters": {
      "primitive": "anisotropic 3D Gaussians",
      "color_encoding": "spherical harmonics",
      "rendering": "tile-based rasterization"
    },
    "input_output": {
      "input": "Multi-view images",
      "output": "Real-time renderable 3D scene"
    },
    "introduced_in_paper_id": "paper_3dgs_id",
    "introduced_date": "2023-08-07T00:00:00Z",
    "based_on": ["Point-based rendering", "Splatting techniques"],
    "occurrence_count": 89,
    "improved_by_count": 23
  }
}
```

---

### 4. Metric Node

**Represents**: Evaluation metrics and measurements

```typescript
interface MetricNode {
  id: UUID;
  type: 'metric';
  canonical_name: string;
  properties: {
    aliases: string[];
    metric_type: 'quality' | 'speed' | 'memory' | 'size' | 'robustness';
    unit?: string;                    // "dB", "fps", "MB", "seconds"
    direction: 'higher_is_better' | 'lower_is_better';

    description?: string;
    formula?: string;                 // LaTeX or plain text
    typical_range?: [number, number];

    // Usage statistics
    occurrence_count: number;
    papers_using: number;
  };
  created_at: Date;
  updated_at: Date;
}
```

**Example**:
```json
{
  "id": "880e8400-e29b-41d4-a716-446655440003",
  "type": "metric",
  "canonical_name": "Peak Signal-to-Noise Ratio",
  "properties": {
    "aliases": ["PSNR", "peak signal to noise ratio"],
    "metric_type": "quality",
    "unit": "dB",
    "direction": "higher_is_better",
    "description": "Measures reconstruction quality between images",
    "formula": "10 * log10(MAX^2 / MSE)",
    "typical_range": [20, 40],
    "occurrence_count": 892,
    "papers_using": 743
  }
}
```

---

### 5. Dataset Node

**Represents**: Datasets, benchmarks, evaluation settings

```typescript
interface DatasetNode {
  id: UUID;
  type: 'dataset';
  canonical_name: string;
  properties: {
    aliases: string[];
    dataset_type: 'training' | 'evaluation' | 'benchmark' | 'synthetic' | 'real';
    domain: string;                   // "indoor scenes", "faces", "objects"

    description?: string;
    introduced_in_paper_id?: UUID;
    url?: string;
    size?: {
      num_samples: number;
      storage_size_gb?: number;
    };

    // Usage
    occurrence_count: number;
    papers_training: number;
    papers_evaluation: number;
  };
  created_at: Date;
  updated_at: Date;
}
```

**Example**:
```json
{
  "id": "990e8400-e29b-41d4-a716-446655440004",
  "type": "dataset",
  "canonical_name": "Mip-NeRF 360 Dataset",
  "properties": {
    "aliases": ["Mip-NeRF 360", "MipNeRF360"],
    "dataset_type": "evaluation",
    "domain": "360-degree unbounded scenes",
    "description": "Collection of 360-degree captures with multiple scales",
    "introduced_in_paper_id": "paper_mipnerf360_id",
    "size": {"num_samples": 9},
    "occurrence_count": 156,
    "papers_evaluation": 143
  }
}
```

---

## Edge Types

### Relationship Type System

```typescript
enum RelationType {
  // Improvement relationships
  IMPROVES_ON = 'improves_on',
  EXTENDS = 'extends',
  SIMPLIFIES = 'simplifies',
  GENERALIZES = 'generalizes',
  OPTIMIZES = 'optimizes',

  // Problem-solution
  SOLVES = 'solves',
  ADDRESSES_LIMITATION = 'addresses_limitation',
  ENABLES = 'enables',

  // Usage relationships
  USES_METHOD = 'uses_method',
  USES_DATASET = 'uses_dataset',
  USES_METRIC = 'uses_metric',
  EVALUATES_ON = 'evaluates_on',
  TRAINS_ON = 'trains_on',

  // Conceptual relationships
  INTRODUCES = 'introduces',
  RELATED_TO = 'related_to',
  INSPIRED_BY = 'inspired_by',
  BUILDS_ON = 'builds_on',
  PART_OF = 'part_of',
  INSTANCE_OF = 'instance_of',

  // Critical relationships
  CHALLENGES = 'challenges',
  CONTRADICTS = 'contradicts',
  IDENTIFIES_LIMITATION = 'identifies_limitation',
  REFUTES = 'refutes',

  // Citation relationships
  CITES = 'cites',
  CITED_BY = 'cited_by',

  // Comparison
  COMPARES_WITH = 'compares_with',
  OUTPERFORMS = 'outperforms',
  EQUIVALENT_TO = 'equivalent_to'
}
```

### Edge Schema

```typescript
interface Edge {
  id: UUID;
  source_id: UUID;
  target_id: UUID;
  type: RelationType;
  properties: {
    // Evidence and provenance
    evidence: Evidence[];
    papers: UUID[];              // Papers that assert this relationship
    confidence: number;          // 0-1

    // Qualifications
    qualifications?: string[];   // ["for outdoor scenes only"]
    scope?: string;              // Contextual limitations

    // Quantitative data
    quantitative?: {
      metric: string;
      source_value?: number;
      target_value?: number;
      improvement: string;       // "3x faster", "2.1 dB higher"
      improvement_percent?: number;
    };

    // Temporal
    first_asserted_date: Date;
    last_confirmed_date: Date;

    // Validation
    validation_status: 'confirmed' | 'disputed' | 'needs_review';
    human_verified: boolean;
    verified_by?: string;
    verified_at?: Date;
  };
  created_at: Date;
  updated_at: Date;
}

interface Evidence {
  quote: string;
  paper_id: UUID;
  section: string;
  paragraph: number;
  sentence_index?: number;
  confidence: number;
}
```

### Edge Examples

#### 1. IMPROVES_ON Edge

```json
{
  "id": "edge_001",
  "source_id": "method_3dgs",
  "target_id": "method_nerf",
  "type": "improves_on",
  "properties": {
    "evidence": [{
      "quote": "Our method achieves real-time rendering (≥30 fps) compared to NeRF's 30-second per-frame time",
      "paper_id": "paper_3dgs",
      "section": "Abstract",
      "paragraph": 1,
      "confidence": 0.96
    }],
    "papers": ["paper_3dgs"],
    "confidence": 0.96,
    "qualifications": [],
    "quantitative": {
      "metric": "rendering_speed",
      "source_value": 30.0,
      "target_value": 0.033,
      "improvement": "900x faster",
      "improvement_percent": 90000
    },
    "validation_status": "confirmed",
    "human_verified": true
  }
}
```

#### 2. USES_DATASET Edge

```json
{
  "id": "edge_002",
  "source_id": "paper_3dgs",
  "target_id": "dataset_mipnerf360",
  "type": "uses_dataset",
  "properties": {
    "evidence": [{
      "quote": "We evaluate on the Mip-NeRF 360 dataset",
      "paper_id": "paper_3dgs",
      "section": "Experiments",
      "paragraph": 3,
      "confidence": 0.99
    }],
    "papers": ["paper_3dgs"],
    "confidence": 0.99,
    "scope": "evaluation only, not training",
    "validation_status": "confirmed"
  }
}
```

#### 3. EXTENDS Edge

```json
{
  "id": "edge_003",
  "source_id": "method_mip_splatting",
  "target_id": "method_3dgs",
  "type": "extends",
  "properties": {
    "evidence": [{
      "quote": "We extend 3D Gaussian Splatting to handle anti-aliasing through 3D smoothing and 2D Mip filtering",
      "paper_id": "paper_mip_splatting",
      "section": "Introduction",
      "paragraph": 4,
      "confidence": 0.93
    }],
    "papers": ["paper_mip_splatting"],
    "confidence": 0.93,
    "qualifications": ["adds anti-aliasing capability"],
    "validation_status": "confirmed"
  }
}
```

#### 4. CHALLENGES Edge

```json
{
  "id": "edge_004",
  "source_id": "paper_analysis_3dgs",
  "target_id": "method_3dgs",
  "type": "identifies_limitation",
  "properties": {
    "evidence": [{
      "quote": "3D Gaussian Splatting suffers from aliasing artifacts at high resolutions due to lack of anti-aliasing",
      "paper_id": "paper_analysis_3dgs",
      "section": "Analysis",
      "paragraph": 7,
      "confidence": 0.88
    }],
    "papers": ["paper_analysis_3dgs"],
    "confidence": 0.88,
    "qualifications": ["at high resolutions"],
    "validation_status": "confirmed"
  }
}
```

---

## PostgreSQL Schema

### Core Tables

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";      -- Fuzzy string matching
CREATE EXTENSION IF NOT EXISTS "btree_gin";    -- Composite indexes

-- Nodes table (all entity types)
CREATE TABLE nodes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  type VARCHAR(50) NOT NULL,
  canonical_name TEXT NOT NULL,
  properties JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for nodes
CREATE INDEX idx_nodes_type ON nodes(type);
CREATE INDEX idx_nodes_canonical_name ON nodes(canonical_name);
CREATE INDEX idx_nodes_type_name ON nodes(type, canonical_name);
CREATE INDEX idx_nodes_properties ON nodes USING GIN(properties);
CREATE INDEX idx_nodes_search ON nodes USING GIN(
  to_tsvector('english', canonical_name || ' ' || COALESCE((properties->>'definition')::text, ''))
);

-- Edges table (all relationships)
CREATE TABLE edges (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  source_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  target_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  type VARCHAR(50) NOT NULL,
  properties JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

  -- Prevent duplicate relationships
  CONSTRAINT unique_relationship UNIQUE(source_id, target_id, type)
);

-- Indexes for edges
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_type ON edges(type);
CREATE INDEX idx_edges_source_type ON edges(source_id, type);
CREATE INDEX idx_edges_target_type ON edges(target_id, type);
CREATE INDEX idx_edges_composite ON edges(source_id, target_id, type);
CREATE INDEX idx_edges_properties ON edges USING GIN(properties);

-- Entity occurrences (track which papers mention which entities)
CREATE TABLE entity_occurrences (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  entity_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  paper_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  local_name TEXT,                    -- How this paper refers to the entity
  usage_context TEXT,                 -- Surrounding text
  is_definition BOOLEAN DEFAULT FALSE,
  confidence NUMERIC(3,2),
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_entity_occ_entity ON entity_occurrences(entity_id);
CREATE INDEX idx_entity_occ_paper ON entity_occurrences(paper_id);

-- Processing metadata
CREATE TABLE processing_log (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  paper_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  stage VARCHAR(50) NOT NULL,
  status VARCHAR(20) NOT NULL,        -- success, failed, in_progress
  duration_ms INTEGER,
  error_message TEXT,
  metadata JSONB,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_processing_log_paper ON processing_log(paper_id);
CREATE INDEX idx_processing_log_status ON processing_log(status);

-- Graph statistics (for monitoring)
CREATE TABLE graph_statistics (
  date DATE PRIMARY KEY,
  papers_count INTEGER DEFAULT 0,
  entities_count INTEGER DEFAULT 0,
  relationships_count INTEGER DEFAULT 0,
  new_papers_today INTEGER DEFAULT 0,
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Dead letter queue
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

CREATE INDEX idx_dlq_paper ON dead_letter_queue(paper_id);
CREATE INDEX idx_dlq_error_type ON dead_letter_queue(error_type);
```

### Materialized Views

```sql
-- Paper impact metrics
CREATE MATERIALIZED VIEW mv_paper_impact AS
SELECT
  p.id,
  p.canonical_name as title,
  p.properties->>'arxiv_id' as arxiv_id,
  (p.properties->>'publication_date')::date as publication_date,
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

-- Entity popularity
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

-- Refresh strategy: incremental refresh every hour
-- In production, trigger refresh after batch processing completes
```

### Helper Functions

```sql
-- Function: Find papers that improve on a given method
CREATE OR REPLACE FUNCTION find_papers_that_improve(method_name TEXT)
RETURNS TABLE (
  paper_id UUID,
  paper_title TEXT,
  improvement_claim TEXT,
  confidence NUMERIC
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    p.id,
    p.canonical_name,
    e.properties->>'quantitative'->>'improvement' as improvement_claim,
    (e.properties->>'confidence')::numeric as confidence
  FROM nodes m
  JOIN edges e ON e.target_id = m.id AND e.type = 'improves_on'
  JOIN nodes p ON p.id = e.source_id AND p.type = 'paper'
  WHERE m.type = 'method'
    AND m.canonical_name ILIKE '%' || method_name || '%'
  ORDER BY (e.properties->>'confidence')::numeric DESC;
END;
$$ LANGUAGE plpgsql;

-- Function: Find method lineage (chain of improvements)
CREATE OR REPLACE FUNCTION find_method_lineage(method_name TEXT, max_depth INT DEFAULT 5)
RETURNS TABLE (
  depth INT,
  method_id UUID,
  method_name TEXT,
  relationship_type TEXT,
  improvement_claim TEXT
) AS $$
WITH RECURSIVE lineage AS (
  -- Base case: starting method
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

  -- Recursive case: methods that improve/extend this one
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

-- Function: Semantic search on entities
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
```

---

## Query Patterns

### 1. Find Papers That Improve a Method

```sql
SELECT
  p.canonical_name as paper_title,
  p.properties->>'arxiv_id' as arxiv_id,
  e.properties->'quantitative'->>'improvement' as improvement,
  e.properties->'quantitative'->>'metric' as metric,
  e.properties->'evidence'->0->>'quote' as evidence_quote,
  (e.properties->>'confidence')::numeric as confidence
FROM nodes m
JOIN edges e ON e.target_id = m.id AND e.type = 'improves_on'
JOIN nodes p ON p.id = e.source_id AND p.type = 'paper'
WHERE m.canonical_name = '3D Gaussian Splatting'
ORDER BY (e.properties->>'confidence')::numeric DESC;
```

**Result**:
```
paper_title                           | arxiv_id    | improvement    | metric  | confidence
--------------------------------------|-------------|----------------|---------|----------
Mip-Splatting: Alias-free 3D GS      | 2311.16493  | 2.3 dB higher  | PSNR    | 0.94
Scaffold-GS: Structured 3D Gaussians | 2312.00109  | 40% less memory| memory  | 0.91
```

### 2. Find All Papers Using a Dataset

```sql
SELECT
  p.canonical_name as paper_title,
  p.properties->>'publication_date' as pub_date,
  e.type as usage_type,
  e.properties->>'scope' as usage_scope
FROM nodes d
JOIN edges e ON e.target_id = d.id AND e.type IN ('uses_dataset', 'evaluates_on', 'trains_on')
JOIN nodes p ON p.id = e.source_id AND p.type = 'paper'
WHERE d.canonical_name = 'Mip-NeRF 360 Dataset'
ORDER BY (p.properties->>'publication_date')::date DESC;
```

### 3. Method Lineage (Recursive CTE)

```sql
WITH RECURSIVE method_chain AS (
  -- Start with NeRF
  SELECT
    0 as depth,
    id,
    canonical_name,
    properties->>'introduced_date' as date,
    ''::text as relationship
  FROM nodes
  WHERE canonical_name = 'Neural Radiance Fields'
    AND type = 'method'

  UNION ALL

  -- Find methods that improve/extend previous ones
  SELECT
    mc.depth + 1,
    n.id,
    n.canonical_name,
    n.properties->>'introduced_date',
    e.type
  FROM method_chain mc
  JOIN edges e ON e.target_id = mc.id
    AND e.type IN ('improves_on', 'extends', 'builds_on')
  JOIN nodes n ON n.id = e.source_id
    AND n.type = 'method'
  WHERE mc.depth < 5
)
SELECT
  depth,
  canonical_name,
  date,
  relationship
FROM method_chain
ORDER BY depth, date;
```

**Result**:
```
depth | canonical_name          | date       | relationship
------|-------------------------|------------|--------------
0     | Neural Radiance Fields  | 2020-03-01 |
1     | Mip-NeRF                | 2021-07-01 | extends
1     | Instant-NGP             | 2022-01-01 | improves_on
2     | 3D Gaussian Splatting   | 2023-08-01 | improves_on
3     | Mip-Splatting          | 2023-11-01 | extends
3     | Scaffold-GS             | 2023-12-01 | extends
```

### 4. Find Papers by Semantic Query

```sql
-- Papers about "real-time neural rendering"
SELECT
  p.canonical_name,
  p.properties->>'arxiv_id',
  ts_rank(
    to_tsvector('english', p.canonical_name || ' ' || (p.properties->>'abstract')),
    plainto_tsquery('english', 'real-time neural rendering')
  ) as relevance
FROM nodes p
WHERE p.type = 'paper'
  AND to_tsvector('english', p.canonical_name || ' ' || (p.properties->>'abstract'))
      @@ plainto_tsquery('english', 'real-time neural rendering')
ORDER BY relevance DESC
LIMIT 10;
```

### 5. Find Contradictory Claims

```sql
-- Find pairs of papers with contradictory relationships
SELECT
  p1.canonical_name as paper1,
  p2.canonical_name as paper2,
  m.canonical_name as method,
  e1.type as paper1_claims,
  e2.type as paper2_claims,
  e1.properties->'evidence'->0->>'quote' as evidence1,
  e2.properties->'evidence'->0->>'quote' as evidence2
FROM nodes m
JOIN edges e1 ON e1.target_id = m.id
JOIN edges e2 ON e2.target_id = m.id AND e2.id != e1.id
JOIN nodes p1 ON p1.id = e1.source_id
JOIN nodes p2 ON p2.id = e2.source_id
WHERE m.type = 'method'
  AND (
    (e1.type = 'improves_on' AND e2.type = 'challenges') OR
    (e1.type = 'solves' AND e2.type = 'identifies_limitation')
  );
```

### 6. Entity Co-occurrence Analysis

```sql
-- Find entities frequently mentioned together
SELECT
  e1.canonical_name as entity1,
  e2.canonical_name as entity2,
  COUNT(DISTINCT eo1.paper_id) as cooccurrence_count
FROM entity_occurrences eo1
JOIN entity_occurrences eo2
  ON eo1.paper_id = eo2.paper_id
  AND eo1.entity_id < eo2.entity_id  -- Prevent duplicates
JOIN nodes e1 ON e1.id = eo1.entity_id
JOIN nodes e2 ON e2.id = eo2.entity_id
WHERE e1.type = 'method' AND e2.type = 'method'
GROUP BY e1.canonical_name, e2.canonical_name
HAVING COUNT(DISTINCT eo1.paper_id) >= 5
ORDER BY cooccurrence_count DESC
LIMIT 20;
```

---

## Semantic Encoding

### Relationship Semantics

**Directional Relationships**:
- `A improves_on B` ≠ `B improves_on A`
- Source is the improving entity, target is improved upon
- Temporal: source typically newer than target

**Transitive Relationships**:
- `extends`, `builds_on`, `part_of` are transitive
- `improves_on` is NOT transitive (A improves B, B improves C doesn't mean A improves C)

**Symmetric Relationships**:
- `related_to`, `compares_with`, `equivalent_to`
- Stored once with lower UUID as source

**Confidence Propagation**:
```typescript
function propagateConfidence(path: Edge[]): number {
  // Confidence decreases with path length
  const baseConfidence = path.map(e => e.properties.confidence).reduce((a, b) => a * b, 1);
  const lengthPenalty = Math.pow(0.9, path.length - 1);
  return baseConfidence * lengthPenalty;
}
```

### Evidence Weighting

Multiple papers asserting same relationship increases confidence:

```typescript
function aggregateEvidence(edges: Edge[]): number {
  // Bayesian aggregation
  const individual = edges.map(e => e.properties.confidence);
  const combined = 1 - individual.reduce((acc, p) => acc * (1 - p), 1);
  return Math.min(combined, 0.99);  // Cap at 0.99
}
```

---

## Schema Evolution Strategy

### Dynamic Node/Edge Type Creation

**Question**: Should agents dynamically create new node or edge types?

**Answer**: **Controlled expansion with review**

**Approach**:
1. **Predefined core types** (paper, concept, method, metric, dataset)
2. **Extensible via properties** (new subtypes in JSONB)
3. **Proposal mechanism** for new types

```typescript
interface TypeProposal {
  proposed_type: string;
  rationale: string;
  example_instances: any[];
  proposed_by: 'agent' | 'human';
  status: 'pending' | 'approved' | 'rejected';
}

// Agents can propose new types
async function proposeNewType(proposal: TypeProposal): Promise<void> {
  await db('type_proposals').insert(proposal);
  await notifyAdmins(proposal);
}

// Admin reviews and approves
async function approveType(proposalId: string): Promise<void> {
  // Update schema, migrate data, update agent prompts
  await db.transaction(async (trx) => {
    await trx('type_proposals').where({ id: proposalId }).update({ status: 'approved' });
    // Run migration...
  });
}
```

**Examples of potential new types**:
- `Tool` node (software tools, libraries)
- `Evaluation Protocol` node (specific evaluation procedures)
- `Hardware` node (GPU models, compute resources)

### Versioning

Every extraction includes version tag:
```json
{
  "extraction_version": "1.2.0",
  "schema_version": "1.0.0",
  "agent_versions": {
    "concept_extractor": "1.1.0",
    "relationship_extractor": "1.3.1"
  }
}
```

Enables:
- Reprocessing papers with newer agents
- A/B testing new extraction strategies
- Rollback if new version produces worse results

### Migration Strategy

When schema changes:
1. Add new columns/tables (non-breaking)
2. Populate via background job
3. Deprecate old columns after migration complete
4. Drop old columns in next major version

```sql
-- Example: Adding embedding column
ALTER TABLE nodes ADD COLUMN embedding vector(1536);  -- pgvector extension

-- Backfill asynchronously
UPDATE nodes SET embedding = generate_embedding(canonical_name)
WHERE embedding IS NULL AND type IN ('concept', 'method')
LIMIT 1000;  -- Batch by batch
```

---

## Summary

This graph schema provides:

**Expressiveness**: Rich properties on nodes and edges capture nuanced research relationships

**Query Power**: SQL + recursive CTEs enable complex graph traversal and analytics

**Flexibility**: JSONB properties allow schema evolution without migrations

**Performance**: Comprehensive indexing, materialized views for common queries

**Integrity**: Foreign keys, unique constraints, ACID transactions

**Scalability**: Designed to handle 100K+ papers, millions of entities/relationships

**Provenance**: Every extraction tracked to source, version, confidence

The schema strikes a balance between structure (for reliable queries) and flexibility (for research evolution).
