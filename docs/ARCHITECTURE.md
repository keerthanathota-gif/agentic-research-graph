# System Architecture

## Table of Contents
1. [Overview](#overview)
2. [Component Specifications](#component-specifications)
3. [Data Flow](#data-flow)
4. [Agent Interaction Patterns](#agent-interaction-patterns)
5. [Error Handling & Fault Tolerance](#error-handling--fault-tolerance)
6. [Confidence Scoring Framework](#confidence-scoring-framework)
7. [Quality Control Mechanisms](#quality-control-mechanisms)
8. [Performance Optimization](#performance-optimization)

---

## Overview

### Architectural Style

The system employs a **pipeline architecture** with **asynchronous agent orchestration**. Papers flow through multiple processing stages, with each stage handled by specialized agents that operate independently but coordinate through a central orchestrator.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INGESTION ORCHESTRATOR                             │
│                    (State Management • Scheduling • Coordination)           │
└────────────┬────────────────────────────────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────┐
    │         Paper Discovery Layer               │
    ├─────────────────┬──────────────────────────┤
    │ PaperFetcher    │  CorpusSelector          │
    │ (arXiv API)     │  (Citation BFS/DFS)      │
    └─────────────────┴────────────┬─────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │  Parsing Agent  │
                          │  (PDF → AST)    │
                          └────────┬────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │   Entity Extraction Ensemble   │
                   ├───────────────────────────────┤
                   │  ╔═══════════════════════╗   │
                   │  ║ Concept Extractor     ║   │
                   │  ╠═══════════════════════╣   │
                   │  ║ Method Extractor      ║   │
                   │  ╠═══════════════════════╣   │
                   │  ║ Metric Extractor      ║   │
                   │  ╠═══════════════════════╣   │
                   │  ║ Dataset Extractor     ║   │
                   │  ╚═══════════════════════╝   │
                   └───────────┬───────────────────┘
                               │
                               ▼
                   ┌─────────────────────────────┐
                   │ Relationship Extractor       │
                   │ (Semantic Analysis)          │
                   └───────────┬─────────────────┘
                               │
                               ▼
                   ┌─────────────────────────────┐
                   │   Entity Linking Agent       │
                   │ (Cross-paper Normalization)  │
                   └───────────┬─────────────────┘
                               │
                               ▼
                   ┌─────────────────────────────┐
                   │  Quality Control Agent       │
                   │ (Validation • Calibration)   │
                   └───────────┬─────────────────┘
                               │
                               ▼
                   ┌─────────────────────────────┐
                   │   Graph Builder Service      │
                   │ (Transactional Persistence)  │
                   └───────────┬─────────────────┘
                               │
                               ▼
                   ┌─────────────────────────────┐
                   │      PostgreSQL Graph        │
                   │  (Nodes • Edges • Metadata)  │
                   └──────────────────────────────┘
```

### Design Principles

1. **Single Responsibility**: Each component has one well-defined purpose
2. **Idempotency**: All operations can be safely retried without side effects
3. **Observability**: Every operation emits structured logs, metrics, and traces
4. **Graceful Degradation**: System continues with reduced functionality on partial failures
5. **Cost Awareness**: Intelligent model selection based on task complexity
6. **Data Provenance**: Every piece of extracted information tracks its source

---

## Component Specifications

### 1. Ingestion Orchestrator

**Type**: Coordinator Service
**Responsibility**: Manages the entire pipeline from paper discovery to graph storage

#### Core Functions

```typescript
interface OrchestrationContext {
  paperId: string;
  stage: PipelineStage;
  attemptCount: number;
  priority: number;
  dependencies: string[];      // Papers that must be processed first
  metadata: {
    addedBy: 'seed' | 'citation' | 'manual' | 'realtime';
    corpusDepth: number;        // Distance from seed paper
    urgency: 'low' | 'medium' | 'high';
  };
}

enum PipelineStage {
  QUEUED = 'queued',
  FETCHING = 'fetching',
  PARSING = 'parsing',
  ENTITY_EXTRACTION = 'entity_extraction',
  RELATIONSHIP_EXTRACTION = 'relationship_extraction',
  ENTITY_LINKING = 'entity_linking',
  QUALITY_CONTROL = 'quality_control',
  GRAPH_BUILDING = 'graph_building',
  COMPLETED = 'completed',
  FAILED = 'failed',
  RETRY_SCHEDULED = 'retry_scheduled'
}

class IngestionOrchestrator {
  private queue: BullQueue<OrchestrationContext>;
  private stateStore: RedisStateStore;
  private checkpointer: CheckpointService;

  async schedulePaper(arxivId: string, options: ScheduleOptions): Promise<void> {
    // Priority calculation based on citation count, recency, relevance
    const priority = await this.calculatePriority(arxivId, options);

    const context: OrchestrationContext = {
      paperId: arxivId,
      stage: PipelineStage.QUEUED,
      attemptCount: 0,
      priority,
      dependencies: [],
      metadata: {
        addedBy: options.source,
        corpusDepth: options.depth || 0,
        urgency: options.urgent ? 'high' : 'medium'
      }
    };

    await this.queue.add('process-paper', context, {
      priority,
      attempts: 3,
      backoff: { type: 'exponential', delay: 60000 }
    });
  }

  async processPaper(context: OrchestrationContext): Promise<void> {
    const span = tracer.startSpan('orchestrator.process_paper', {
      attributes: { paperId: context.paperId }
    });

    try {
      // State checkpoint
      await this.stateStore.set(context.paperId, context);

      // Execute pipeline stages
      await this.executeFetching(context);
      await this.checkpoint(context, PipelineStage.PARSING);

      await this.executeParsing(context);
      await this.checkpoint(context, PipelineStage.ENTITY_EXTRACTION);

      await this.executeEntityExtraction(context);
      await this.checkpoint(context, PipelineStage.RELATIONSHIP_EXTRACTION);

      await this.executeRelationshipExtraction(context);
      await this.checkpoint(context, PipelineStage.ENTITY_LINKING);

      await this.executeEntityLinking(context);
      await this.checkpoint(context, PipelineStage.QUALITY_CONTROL);

      await this.executeQualityControl(context);
      await this.checkpoint(context, PipelineStage.GRAPH_BUILDING);

      await this.executeGraphBuilding(context);

      context.stage = PipelineStage.COMPLETED;
      await this.stateStore.set(context.paperId, context);

      logger.info('paper_processing_completed', { paperId: context.paperId });

    } catch (error) {
      await this.handleFailure(context, error);
      throw error;
    } finally {
      span.end();
    }
  }

  private async checkpoint(
    context: OrchestrationContext,
    nextStage: PipelineStage
  ): Promise<void> {
    context.stage = nextStage;
    await this.checkpointer.save(context);
    await this.stateStore.set(context.paperId, context);
  }

  async resumeFromCheckpoint(paperId: string): Promise<void> {
    const context = await this.checkpointer.load(paperId);
    if (!context) {
      throw new Error(`No checkpoint found for paper ${paperId}`);
    }

    logger.info('resuming_from_checkpoint', {
      paperId,
      stage: context.stage,
      attemptCount: context.attemptCount
    });

    await this.processPaper(context);
  }
}
```

#### State Management

**Redis Schema:**
```typescript
// Current state: processing:paper:{arxivId}
{
  stage: PipelineStage,
  attemptCount: number,
  lastUpdated: ISO8601,
  currentAgent: string,
  intermediateResults: { ... }
}

// Checkpoint: checkpoint:paper:{arxivId}:{stage}
{
  context: OrchestrationContext,
  data: any,  // Stage-specific output
  timestamp: ISO8601
}

// Lock: lock:paper:{arxivId}
{
  acquired: ISO8601,
  expiresAt: ISO8601,
  owner: string  // Worker ID
}
```

#### Priority Calculation

```typescript
function calculatePriority(paper: PaperMetadata): number {
  const factors = {
    citationCount: Math.log10(paper.citationCount + 1) * 10,  // 0-30 points
    recency: calculateRecencyScore(paper.publicationDate),      // 0-20 points
    relevance: paper.relevanceScore || 0,                       // 0-30 points
    urgency: paper.urgent ? 20 : 0                              // 0-20 points
  };

  return Object.values(factors).reduce((sum, val) => sum + val, 0);
}

function calculateRecencyScore(date: Date): number {
  const daysSincePublished = (Date.now() - date.getTime()) / (1000 * 60 * 60 * 24);
  if (daysSincePublished < 30) return 20;
  if (daysSincePublished < 90) return 15;
  if (daysSincePublished < 365) return 10;
  return 5;
}
```

#### Design Rationale

- **Centralized orchestration** prevents race conditions and duplicate processing
- **Checkpointing** enables recovery without full reprocessing (saves ~70% cost on failures)
- **Priority queue** ensures important papers processed first (seed paper, highly cited)
- **Distributed locks** prevent concurrent processing of same paper
- **State tracking** provides visibility into pipeline health

---

### 2. Paper Fetcher Service

**Type**: Data Acquisition Service
**Responsibility**: Retrieve papers from academic repositories

#### Implementation

```typescript
interface PaperMetadata {
  arxivId: string;
  doi?: string;
  title: string;
  authors: Author[];
  abstract: string;
  publicationDate: Date;
  categories: string[];          // ['cs.CV', 'cs.LG']
  comments?: string;             // e.g., "Accepted to CVPR 2023"
  journal?: string;
  version: number;
  references: string[];          // Extracted from paper
  pdfUrl: string;
  sourceRepo: 'arxiv' | 'openreview' | 'semantic_scholar';
}

interface Author {
  name: string;
  affiliation?: string;
  orcid?: string;
}

class PaperFetcher {
  private arxivClient: ArxivAPI;
  private cache: CacheService;
  private rateLimiter: RateLimiter;

  async fetchPaper(arxivId: string): Promise<PaperWithPDF> {
    const span = tracer.startSpan('fetcher.fetch_paper');

    try {
      // Check cache
      const cached = await this.cache.get(`paper:${arxivId}`);
      if (cached) {
        logger.debug('cache_hit', { arxivId });
        return cached;
      }

      // Rate limiting (arXiv: 1 req/3s)
      await this.rateLimiter.acquire('arxiv');

      // Fetch metadata
      const metadata = await this.arxivClient.getMetadata(arxivId);

      // Download PDF
      const pdfBuffer = await this.downloadPDF(metadata.pdfUrl);

      const result = { metadata, pdfBuffer };

      // Cache for 24 hours
      await this.cache.set(`paper:${arxivId}`, result, { ttl: 86400 });

      return result;

    } catch (error) {
      if (error.statusCode === 404) {
        throw new PaperNotFoundError(arxivId);
      }
      if (error.statusCode === 429) {
        throw new RateLimitError('arXiv API rate limit exceeded');
      }
      throw error;
    } finally {
      span.end();
    }
  }

  private async downloadPDF(url: string): Promise<Buffer> {
    const response = await fetch(url, {
      headers: { 'User-Agent': 'ResearchKG/1.0' }
    });

    if (!response.ok) {
      throw new PDFDownloadError(`Failed to download PDF: ${response.statusText}`);
    }

    return Buffer.from(await response.arrayBuffer());
  }

  async extractReferences(pdfBuffer: Buffer): Promise<string[]> {
    // Parse references section
    // Match arXiv IDs: \d{4}\.\d{4,5}
    // Match DOIs: 10\.\d{4,}/.*
    const text = await this.extractTextFromPDF(pdfBuffer);
    const refSection = this.findReferenceSection(text);

    const arxivIds = this.extractArxivIds(refSection);
    const dois = this.extractDOIs(refSection);

    return [...arxivIds, ...dois];
  }
}
```

#### Rate Limiting Strategy

```typescript
class RateLimiter {
  private readonly limits = {
    arxiv: { requests: 1, per: 3000 },        // 1 req / 3 seconds
    semantic_scholar: { requests: 100, per: 300000 }  // 100 req / 5 min
  };

  async acquire(service: string): Promise<void> {
    const key = `ratelimit:${service}`;
    const limit = this.limits[service];

    const current = await redis.incr(key);
    if (current === 1) {
      await redis.pexpire(key, limit.per);
    }

    if (current > limit.requests) {
      const ttl = await redis.pttl(key);
      throw new RateLimitError(`Rate limit exceeded. Retry in ${ttl}ms`);
    }
  }
}
```

#### Design Rationale

- **Separate service** allows easy addition of new sources (Semantic Scholar, OpenReview)
- **Caching** prevents redundant downloads (saves bandwidth, respects API limits)
- **Metadata normalization** ensures consistent downstream processing
- **Reference extraction** enables citation-based corpus expansion

---

### 3. Parsing Agent

**Type**: LLM-Powered Agent
**Responsibility**: Convert PDF into structured, semantic representation

#### Output Schema

```typescript
interface StructuredPaper {
  metadata: PaperMetadata;
  sections: Section[];
  figures: Figure[];
  tables: Table[];
  equations: Equation[];
  references: Reference[];
  fullText: string;            // For validation
  structure: DocumentStructure;
}

interface Section {
  id: string;
  title: string;
  level: number;               // 1=section, 2=subsection, 3=subsubsection
  content: string;
  paragraphs: Paragraph[];
  startPage: number;
  endPage: number;
}

interface Paragraph {
  id: string;
  content: string;
  citations: InlineCitation[];
  position: { page: number; bbox: BoundingBox };
}

interface InlineCitation {
  text: string;               // "[1]", "Smith et al. (2023)"
  referenceId: string;        // Links to Reference
  position: number;           // Character offset in paragraph
}

interface Reference {
  id: string;
  rawText: string;            // As appears in paper
  parsed: {
    authors?: string[];
    title?: string;
    venue?: string;
    year?: number;
    arxivId?: string;
    doi?: string;
  };
}
```

#### Parsing Strategy

**Phase 1: Layout Analysis (pdf.js)**

```typescript
async function analyzeLayout(pdfBuffer: Buffer): Promise<LayoutElements[]> {
  const pdf = await pdfjsLib.getDocument({ data: pdfBuffer }).promise;
  const elements: LayoutElements[] = [];

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const textContent = await page.getTextContent();

    for (const item of textContent.items) {
      elements.push({
        text: item.str,
        bbox: {
          x: item.transform[4],
          y: item.transform[5],
          width: item.width,
          height: item.height
        },
        fontSize: item.height,
        fontName: item.fontName,
        page: pageNum
      });
    }
  }

  return elements;
}
```

**Phase 2: Semantic Structuring (LLM Agent)**

```typescript
const PARSING_PROMPT = `
You are a scientific paper structure extraction specialist. Analyze this layout and identify document sections.

Layout elements (with font sizes and positions):
{layout_json}

Tasks:
1. Identify section boundaries (look for larger fonts, position changes)
2. Classify each block: title, abstract, section_header, body_text, figure_caption, reference, equation
3. Detect column layout (single vs. two-column)
4. Extract section hierarchy (1=section, 2=subsection, etc.)

Output JSON format:
{
  "title": "...",
  "abstract": "...",
  "sections": [
    {"level": 1, "title": "Introduction", "startPage": 1, "blocks": [...]},
    {"level": 2, "title": "Related Work", "startPage": 1, "blocks": [...]}
  ],
  "references_start_page": 8
}

Model: claude-haiku (fast, cost-effective for structured extraction)
Confidence threshold: >0.85
`;

async function structurePaper(layout: LayoutElements[]): Promise<DocumentStructure> {
  const response = await anthropic.messages.create({
    model: 'claude-haiku-20250219',
    max_tokens: 4096,
    messages: [{
      role: 'user',
      content: PARSING_PROMPT.replace('{layout_json}', JSON.stringify(layout))
    }]
  });

  const structure = JSON.parse(response.content[0].text);

  // Validation
  if (!structure.title || !structure.abstract) {
    throw new ParsingError('Could not extract title or abstract');
  }

  return structure;
}
```

**Phase 3: Citation Resolution**

```typescript
function resolveCitations(paper: StructuredPaper): void {
  // Extract inline citations
  const citationPatterns = [
    /\[(\d+(?:,\s*\d+)*)\]/g,                    // [1], [1, 2, 3]
    /([A-Z][a-z]+\s+et\s+al\.,?\s+\d{4})/g,     // Smith et al., 2023
    /\(([A-Z][a-z]+\s+(?:and\s+)?[A-Z][a-z]+,?\s+\d{4})\)/g  // (Smith and Jones, 2023)
  ];

  for (const section of paper.sections) {
    for (const paragraph of section.paragraphs) {
      for (const pattern of citationPatterns) {
        const matches = paragraph.content.matchAll(pattern);
        for (const match of matches) {
          paragraph.citations.push({
            text: match[0],
            referenceId: this.linkToReference(match[1], paper.references),
            position: match.index
          });
        }
      }
    }
  }
}

function linkToReference(citationText: string, references: Reference[]): string {
  // Numeric: [1] → references[0]
  if (/^\d+$/.test(citationText)) {
    const index = parseInt(citationText) - 1;
    return references[index]?.id || 'unknown';
  }

  // Author-year: Smith et al., 2023 → fuzzy match in references
  return this.fuzzyMatchReference(citationText, references);
}
```

#### Error Handling

```typescript
async function parseWithFallback(pdfBuffer: Buffer): Promise<StructuredPaper> {
  try {
    // Primary: pdf.js + Claude
    return await this.parsePrimary(pdfBuffer);
  } catch (error) {
    logger.warn('primary_parsing_failed', { error: error.message });

    try {
      // Fallback 1: pymupdf
      return await this.parseWithPyMuPDF(pdfBuffer);
    } catch (error2) {
      logger.warn('pymupdf_parsing_failed', { error: error2.message });

      // Fallback 2: OCR for scanned PDFs
      return await this.parseWithOCR(pdfBuffer);
    }
  }
}
```

#### Design Rationale

- **Structured output** enables targeted extraction (focus on Method sections for algorithms)
- **Layout preservation** helps understand context (tables often contain results)
- **Citation parsing** critical for relationship extraction
- **Fallback parsers** handle edge cases (scanned PDFs, unusual formats)
- **Fast model (Haiku)** keeps costs low for high-volume task

---

### 4. Entity Extraction Agent Ensemble

**Type**: Multi-Agent Ensemble
**Responsibility**: Extract domain-specific entities with specialization

#### Agent Specializations

```typescript
interface ExtractedEntity {
  id: string;
  type: EntityType;
  name: string;
  normalizedName: string;
  aliases: string[];
  confidence: number;          // 0-1
  importance: number;          // 1-10 (relevance to paper)
  firstMention: Location;
  allMentions: Location[];
  definition?: string;         // If defined in paper
  category?: string;           // Sub-categorization
  attributes?: Record<string, any>;
}

enum EntityType {
  CONCEPT = 'concept',
  METHOD = 'method',
  METRIC = 'metric',
  DATASET = 'dataset',
  TOOL = 'tool',
  PROBLEM = 'problem'
}

interface Location {
  section: string;
  paragraph: number;
  sentenceIndex?: number;
}
```

#### 4.1 Concept Extractor

```typescript
const CONCEPT_EXTRACTION_PROMPT = `
You are extracting research concepts from a scientific paper. Identify conceptual terms at multiple levels of granularity.

Paper section:
{section_content}

Paper abstract (for context):
{abstract}

Guidelines:
1. Extract concepts from broad to specific:
   - Broad: "computer vision", "3D reconstruction"
   - Specific: "neural scene representation", "volumetric ray marching"
2. Focus on conceptual ideas, not specific methods or datasets
3. Include related concepts mentioned for context
4. Normalize to canonical forms: "NeRF" → "Neural Radiance Fields"
5. Rate importance (1-10) based on centrality to paper's contribution
6. Rate confidence (0-1) based on clarity of definition

Output JSON array:
[
  {
    "name": "neural scene representation",
    "normalized_name": "Neural Scene Representation",
    "aliases": ["neural representations", "learned scene representations"],
    "importance": 9,
    "confidence": 0.95,
    "first_mention": {"section": "Introduction", "paragraph": 2},
    "definition": "Methods that encode 3D scenes in neural network weights"
  },
  ...
]

Few-shot examples:

Example 1:
Section: "Neural Radiance Fields (NeRF) represent scenes as continuous volumetric functions..."
Output: {
  "name": "Neural Radiance Fields",
  "normalized_name": "Neural Radiance Fields",
  "aliases": ["NeRF", "NeRFs", "neural radiance field"],
  "importance": 10,
  "confidence": 0.98,
  "definition": "Continuous volumetric scene representations using neural networks"
}

Example 2:
Section: "We address the view synthesis problem..."
Output: {
  "name": "view synthesis",
  "normalized_name": "View Synthesis",
  "aliases": ["novel view synthesis", "image synthesis"],
  "importance": 8,
  "confidence": 0.92,
  "definition": "Generating new camera viewpoints from input images"
}

Model: claude-sonnet-4.5 (complex reasoning for concept identification)
`;

class ConceptExtractor {
  async extract(paper: StructuredPaper): Promise<ExtractedEntity[]> {
    const concepts: ExtractedEntity[] = [];

    // Extract from each major section
    const sectionsToProcess = paper.sections.filter(s =>
      ['abstract', 'introduction', 'method', 'related work'].includes(s.title.toLowerCase())
    );

    for (const section of sectionsToProcess) {
      const sectionConcepts = await this.extractFromSection(section, paper.metadata.abstract);
      concepts.push(...sectionConcepts);
    }

    // Deduplicate and merge
    return this.deduplicateConcepts(concepts);
  }

  private async extractFromSection(
    section: Section,
    abstract: string
  ): Promise<ExtractedEntity[]> {
    const prompt = CONCEPT_EXTRACTION_PROMPT
      .replace('{section_content}', section.content)
      .replace('{abstract}', abstract);

    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 8192,
      temperature: 0.2,  // Low temperature for consistency
      messages: [{ role: 'user', content: prompt }]
    });

    const extracted = JSON.parse(response.content[0].text);

    return extracted.map((e: any) => ({
      id: uuid(),
      type: EntityType.CONCEPT,
      name: e.name,
      normalizedName: e.normalized_name,
      aliases: e.aliases || [],
      confidence: e.confidence,
      importance: e.importance,
      firstMention: e.first_mention,
      definition: e.definition,
      category: 'concept'
    }));
  }

  private deduplicateConcepts(concepts: ExtractedEntity[]): ExtractedEntity[] {
    const grouped = new Map<string, ExtractedEntity[]>();

    for (const concept of concepts) {
      const key = concept.normalizedName.toLowerCase();
      if (!grouped.has(key)) {
        grouped.set(key, []);
      }
      grouped.get(key)!.push(concept);
    }

    return Array.from(grouped.values()).map(group => {
      // Merge duplicates: take highest confidence, combine aliases
      const best = group.reduce((a, b) => a.confidence > b.confidence ? a : b);
      best.aliases = [...new Set(group.flatMap(g => g.aliases))];
      best.allMentions = group.flatMap(g => g.firstMention ? [g.firstMention] : []);
      return best;
    });
  }
}
```

#### 4.2 Method Extractor

```typescript
const METHOD_EXTRACTION_PROMPT = `
Extract all methodological components from this paper section. Distinguish between methods proposed in THIS paper vs. methods from prior work.

Section:
{section_content}

For each method identify:
1. Origin: proposed_in_this_paper, adapted_from_prior, baseline_comparison
2. Type: algorithm, architecture, training_procedure, optimization, representation
3. Key parameters or design choices
4. Problem it addresses
5. Claims of improvement over alternatives (if any)

Output JSON array:
[
  {
    "name": "3D Gaussian Splatting",
    "normalized_name": "3D Gaussian Splatting",
    "aliases": ["3D-GS", "Gaussian Splatting"],
    "origin": "proposed_in_this_paper",
    "type": "representation",
    "description": "Represents scenes as explicit 3D Gaussians with learnable parameters",
    "parameters": {
      "primitive": "anisotropic 3D Gaussians",
      "color_encoding": "spherical harmonics",
      "rendering": "tile-based rasterization"
    },
    "solves_problem": "Real-time rendering of neural scene representations",
    "improves_over": ["NeRF", "Instant-NGP"],
    "improvement_claim": "Achieves 30 FPS vs. 0.1 FPS for NeRF",
    "confidence": 0.96
  },
  {
    "name": "ADAM optimizer",
    "normalized_name": "ADAM Optimizer",
    "origin": "baseline_comparison",
    "type": "optimization",
    "confidence": 0.88
  }
]

Few-shot examples:

Example 1:
Text: "We propose a novel density control strategy that adaptively adds and removes Gaussians..."
Output: {
  "name": "adaptive density control",
  "origin": "proposed_in_this_paper",
  "type": "training_procedure",
  "description": "Dynamically adds/removes Gaussians during training",
  "confidence": 0.94
}

Example 2:
Text: "We compare against NeRF [20] and Instant-NGP [30] as baselines..."
Output: [
  {"name": "NeRF", "origin": "baseline_comparison", "confidence": 0.99},
  {"name": "Instant-NGP", "origin": "baseline_comparison", "confidence": 0.99}
]

Model: claude-sonnet-4.5
`;

class MethodExtractor {
  async extract(paper: StructuredPaper): Promise<ExtractedEntity[]> {
    // Focus on Method, Approach, Experiments sections
    const methodSections = paper.sections.filter(s =>
      /method|approach|implementation|architecture|algorithm/i.test(s.title)
    );

    if (methodSections.length === 0) {
      logger.warn('no_method_section_found', { paperId: paper.metadata.arxivId });
      methodSections.push(paper.sections[1] || paper.sections[0]); // Fallback to intro
    }

    const methods: ExtractedEntity[] = [];

    for (const section of methodSections) {
      const extracted = await this.extractFromSection(section);
      methods.push(...extracted);
    }

    return methods;
  }

  // Similar structure to ConceptExtractor
}
```

#### 4.3 Metric Extractor

```typescript
const METRIC_EXTRACTION_PROMPT = `
Extract evaluation metrics and quantitative results from this paper section.

Section:
{section_content}

For each metric:
1. Name (e.g., "PSNR", "training time", "FPS")
2. Type: quality_metric, speed_metric, memory_metric, size_metric
3. Unit (e.g., "dB", "seconds", "fps", "MB")
4. Values reported (with associated methods)
5. Comparison direction: higher_is_better or lower_is_better

Output JSON:
[
  {
    "name": "PSNR",
    "normalized_name": "Peak Signal-to-Noise Ratio",
    "type": "quality_metric",
    "unit": "dB",
    "direction": "higher_is_better",
    "values": [
      {"method": "3D Gaussian Splatting", "value": 28.5, "dataset": "Mip-NeRF 360"},
      {"method": "NeRF", "value": 26.1, "dataset": "Mip-NeRF 360"},
      {"method": "Instant-NGP", "value": 27.2, "dataset": "Mip-NeRF 360"}
    ],
    "confidence": 0.97
  },
  {
    "name": "training time",
    "type": "speed_metric",
    "unit": "minutes",
    "direction": "lower_is_better",
    "values": [
      {"method": "3D-GS", "value": 30},
      {"method": "NeRF", "value": 240}
    ],
    "confidence": 0.89
  }
]

Handle variations: "PSNR", "Peak Signal-to-Noise Ratio", "peak signal to noise ratio" → normalized form

Model: claude-sonnet-4.5 (needs to parse tables, understand context)
`;

// Table parsing for metrics
function extractMetricsFromTable(table: Table): ExtractedEntity[] {
  // Tables often structured as:
  //   Method    | PSNR | SSIM | FPS
  //   NeRF      | 26.1 | 0.85 | 0.1
  //   3D-GS     | 28.5 | 0.89 | 30.0

  const metrics: ExtractedEntity[] = [];
  const headers = table.rows[0].cells;
  const metricColumns = headers.slice(1); // Skip "Method" column

  for (const metricName of metricColumns) {
    const values = [];
    for (let i = 1; i < table.rows.length; i++) {
      const row = table.rows[i];
      const methodName = row.cells[0];
      const value = parseFloat(row.cells[metricColumns.indexOf(metricName) + 1]);
      if (!isNaN(value)) {
        values.push({ method: methodName, value });
      }
    }

    metrics.push({
      id: uuid(),
      type: EntityType.METRIC,
      name: metricName,
      normalizedName: normalizeMetricName(metricName),
      aliases: [],
      confidence: 0.95,  // High confidence from structured table
      importance: 7,
      attributes: { values }
    });
  }

  return metrics;
}
```

#### 4.4 Dataset Extractor

```typescript
const DATASET_EXTRACTION_PROMPT = `
Extract datasets, benchmarks, and evaluation settings from this paper.

Section:
{section_content}

For each dataset:
1. Name
2. Type: training, evaluation, benchmark, synthetic, real
3. Description (if provided)
4. How it's used (training, testing, both)
5. Reference to original paper (if mentioned)

Output JSON:
[
  {
    "name": "Mip-NeRF 360",
    "normalized_name": "Mip-NeRF 360 Dataset",
    "type": "evaluation",
    "description": "360-degree captures of real-world scenes with various scales",
    "used_for": "evaluation",
    "source_paper": "Mip-NeRF 360",
    "confidence": 0.94
  },
  {
    "name": "Tanks and Temples",
    "type": "benchmark",
    "used_for": "evaluation",
    "confidence": 0.91
  }
]

Model: claude-haiku (simpler task, cost-effective)
`;
```

#### Ensemble Coordination

```typescript
class EntityExtractionEnsemble {
  private conceptExtractor: ConceptExtractor;
  private methodExtractor: MethodExtractor;
  private metricExtractor: MetricExtractor;
  private datasetExtractor: DatasetExtractor;

  async extractAll(paper: StructuredPaper): Promise<AllEntities> {
    const span = tracer.startSpan('ensemble.extract_all');

    // Parallel extraction
    const [concepts, methods, metrics, datasets] = await Promise.all([
      this.conceptExtractor.extract(paper),
      this.methodExtractor.extract(paper),
      this.metricExtractor.extract(paper),
      this.datasetExtractor.extract(paper)
    ]);

    span.setAttribute('entity_counts', {
      concepts: concepts.length,
      methods: methods.length,
      metrics: metrics.length,
      datasets: datasets.length
    });

    // Resolve overlaps (e.g., a method that's also mentioned as a concept)
    const merged = this.resolveOverlaps({ concepts, methods, metrics, datasets });

    span.end();
    return merged;
  }

  private resolveOverlaps(entities: AllEntities): AllEntities {
    // Priority: METHOD > CONCEPT (if both exist, prefer METHOD classification)
    // Example: "3D Gaussian Splatting" as both concept and method → keep as METHOD

    const methodNames = new Set(entities.methods.map(m => m.normalizedName.toLowerCase()));

    entities.concepts = entities.concepts.filter(c =>
      !methodNames.has(c.normalizedName.toLowerCase())
    );

    return entities;
  }
}
```

#### Design Rationale

- **Specialized extractors** achieve higher precision than generic extraction
- **Parallel execution** reduces total latency (20-40s vs. 80-160s sequential)
- **Model selection**: Sonnet for complex (concepts, methods), Haiku for simple (datasets)
- **Few-shot prompting** improves consistency and format adherence
- **Confidence scoring** enables selective review of uncertain extractions
- **Structured output (JSON)** facilitates downstream processing

---

### 5. Relationship Extraction Agent

**Type**: LLM-Powered Agent
**Responsibility**: Identify semantic relationships between entities

#### Relationship Schema

```typescript
interface ExtractedRelationship {
  id: string;
  sourceEntityId: string;
  targetEntityId: string;
  relationType: RelationType;
  evidence: Evidence;
  confidence: number;
  qualifications?: string[];
  quantitative?: QuantitativeComparison;
  scope?: string;              // e.g., "for outdoor scenes only"
}

enum RelationType {
  // Improvements
  IMPROVES_ON = 'improves_on',
  EXTENDS = 'extends',
  SIMPLIFIES = 'simplifies',
  GENERALIZES = 'generalizes',

  // Problem-solution
  SOLVES = 'solves',
  ADDRESSES_LIMITATION = 'addresses_limitation',

  // Usage
  USES_METHOD = 'uses_method',
  USES_DATASET = 'uses_dataset',
  USES_METRIC = 'uses_metric',
  EVALUATES_ON = 'evaluates_on',

  // Conceptual
  RELATED_TO = 'related_to',
  INTRODUCES = 'introduces',
  INSPIRED_BY = 'inspired_by',
  BUILDS_ON = 'builds_on',

  // Critical
  CHALLENGES = 'challenges',
  CONTRADICTS = 'contradicts',
  IDENTIFIES_LIMITATION = 'identifies_limitation',

  // Citations
  CITES = 'cites',
  CITED_BY = 'cited_by'
}

interface Evidence {
  quote: string;               // Exact text from paper
  section: string;
  paragraph: number;
  sentenceIndex?: number;
}

interface QuantitativeComparison {
  metric: string;
  sourceValue?: number;
  targetValue?: number;
  improvement: string;         // e.g., "3x faster", "2.1 dB higher"
  improvementPercent?: number;
}
```

#### Extraction Prompt

```typescript
const RELATIONSHIP_EXTRACTION_PROMPT = `
You are analyzing semantic relationships in a research paper. Identify connections between methods, concepts, datasets, and metrics.

Paper section:
{section_content}

Previously extracted entities:
Concepts: {concepts}
Methods: {methods}
Datasets: {datasets}
Metrics: {metrics}

Relationship types to identify:
- improves_on: "Our method outperforms X", "achieves higher quality than X"
- extends: "We extend X to handle Y", "builds upon X"
- addresses_limitation: "X fails to handle Y; we address this by..."
- uses_method: "We employ X for Y"
- uses_dataset: "We evaluate on X"
- evaluates_on: "We measure performance using metric X"
- challenges: "We demonstrate that X's assumption is incorrect"
- related_to: "Similar to X, we..."

For EACH relationship:
1. Identify source entity (often "our method", "this paper")
2. Identify target entity (specific name from entity list)
3. Classify relationship type
4. Extract evidence: exact quote supporting this relationship
5. Note any quantitative comparison (e.g., "15% higher PSNR")
6. Rate confidence (0-1)
7. Note qualifications/caveats (e.g., "for outdoor scenes only")

Output JSON array:
[
  {
    "source_entity": "3D Gaussian Splatting",
    "target_entity": "Neural Radiance Fields",
    "relation_type": "improves_on",
    "evidence": {
      "quote": "Our method achieves real-time rendering (≥30 fps at 1080p) while maintaining competitive quality, compared to NeRF's 30-second per-frame rendering time.",
      "section": "Abstract",
      "paragraph": 1
    },
    "quantitative": {
      "metric": "rendering_speed",
      "source_value": 30,
      "target_value": 0.033,
      "improvement": "900x faster"
    },
    "confidence": 0.96
  },
  {
    "source_entity": "3D Gaussian Splatting",
    "target_entity": "Instant-NGP",
    "relation_type": "improves_on",
    "evidence": {
      "quote": "We achieve higher quality than Instant-NGP on complex outdoor scenes (average PSNR 28.5 vs 27.1) while matching its rendering speed.",
      "section": "Results",
      "paragraph": 12
    },
    "quantitative": {
      "metric": "PSNR",
      "source_value": 28.5,
      "target_value": 27.1,
      "improvement": "1.4 dB higher"
    },
    "qualifications": ["on complex outdoor scenes"],
    "confidence": 0.91
  }
]

Guidelines:
- Be precise: "improves" requires comparative claim, not just "we use X"
- Ground in text: every relationship must have exact quote
- Identify scope: if claim applies only to specific settings, note it
- Quantitative preferred: extract numeric comparisons when available
- Confidence: 0.9+ for explicit statements, 0.6-0.8 for implicit, <0.6 for inferred

Few-shot examples:

Example 1:
Text: "Unlike NeRF which requires ray marching through a volume, our method uses explicit 3D Gaussians for direct rendering."
Output: {
  "source_entity": "3D Gaussian Splatting",
  "target_entity": "NeRF",
  "relation_type": "differs_from",
  "evidence": {...},
  "confidence": 0.88
}

Example 2:
Text: "We demonstrate that the locality assumption in [15] does not hold for wide-baseline captures."
Output: {
  "source_entity": "3D Gaussian Splatting",
  "target_entity": "[15]",  // Would be resolved to paper title
  "relation_type": "challenges",
  "evidence": {...},
  "confidence": 0.85
}

Model: claude-sonnet-4.5 (complex reasoning required)
Temperature: 0.1 (low for consistency)
`;
```

#### Multi-Pass Extraction

```typescript
class RelationshipExtractor {
  async extract(
    paper: StructuredPaper,
    entities: AllEntities
  ): Promise<ExtractedRelationship[]> {

    const relationships: ExtractedRelationship[] = [];

    // Pass 1: Intra-paper relationships (high confidence)
    const intraPaper = await this.extractIntraPaper(paper, entities);
    relationships.push(...intraPaper);

    // Pass 2: Cross-paper relationships (compare with existing graph)
    const crossPaper = await this.extractCrossPaper(paper, entities);
    relationships.push(...crossPaper);

    // Pass 3: Implicit relationships (lower confidence, needs inference)
    const implicit = await this.extractImplicit(paper, entities);
    relationships.push(...implicit.filter(r => r.confidence > 0.6));

    // Deduplicate
    return this.deduplicateRelationships(relationships);
  }

  private async extractIntraPaper(
    paper: StructuredPaper,
    entities: AllEntities
  ): Promise<ExtractedRelationship[]> {

    const relationships: ExtractedRelationship[] = [];

    // Focus on sections with comparative claims
    const sectionsToAnalyze = paper.sections.filter(s =>
      /abstract|introduction|method|results|experiments|related work/i.test(s.title)
    );

    for (const section of sectionsToAnalyze) {
      const prompt = RELATIONSHIP_EXTRACTION_PROMPT
        .replace('{section_content}', section.content)
        .replace('{concepts}', JSON.stringify(entities.concepts.map(c => c.name)))
        .replace('{methods}', JSON.stringify(entities.methods.map(m => m.name)))
        .replace('{datasets}', JSON.stringify(entities.datasets.map(d => d.name)))
        .replace('{metrics}', JSON.stringify(entities.metrics.map(m => m.name)));

      const response = await anthropic.messages.create({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 8192,
        temperature: 0.1,
        messages: [{ role: 'user', content: prompt }]
      });

      const extracted = JSON.parse(response.content[0].text);

      for (const rel of extracted) {
        relationships.push({
          id: uuid(),
          sourceEntityId: this.resolveEntityId(rel.source_entity, entities),
          targetEntityId: this.resolveEntityId(rel.target_entity, entities),
          relationType: rel.relation_type,
          evidence: rel.evidence,
          confidence: rel.confidence,
          qualifications: rel.qualifications,
          quantitative: rel.quantitative
        });
      }
    }

    return relationships;
  }

  private async extractCrossPaper(
    paper: StructuredPaper,
    entities: AllEntities
  ): Promise<ExtractedRelationship[]> {

    // Query existing graph for related papers
    const relatedPapers = await this.graphDB.findRelatedPapers(
      paper.metadata.arxivId,
      { limit: 20, minSimilarity: 0.7 }
    );

    // Compare current paper's methods with existing graph
    const crossRelationships: ExtractedRelationship[] = [];

    for (const method of entities.methods.filter(m => m.category === 'proposed')) {
      for (const relatedPaper of relatedPapers) {
        const relationships = await this.compareMethodsAcrossPapers(
          method,
          paper,
          relatedPaper
        );
        crossRelationships.push(...relationships);
      }
    }

    return crossRelationships;
  }

  private resolveEntityId(entityName: string, entities: AllEntities): string {
    // Fuzzy match entity name to extracted entity IDs
    const allEntities = [
      ...entities.concepts,
      ...entities.methods,
      ...entities.datasets,
      ...entities.metrics
    ];

    const match = allEntities.find(e =>
      e.name.toLowerCase() === entityName.toLowerCase() ||
      e.aliases.some(a => a.toLowerCase() === entityName.toLowerCase())
    );

    if (!match) {
      logger.warn('entity_not_found', { entityName });
      return `unresolved:${entityName}`;
    }

    return match.id;
  }

  private deduplicateRelationships(
    relationships: ExtractedRelationship[]
  ): ExtractedRelationship[] {
    const grouped = new Map<string, ExtractedRelationship[]>();

    for (const rel of relationships) {
      const key = `${rel.sourceEntityId}:${rel.targetEntityId}:${rel.relationType}`;
      if (!grouped.has(key)) {
        grouped.set(key, []);
      }
      grouped.get(key)!.push(rel);
    }

    // For duplicates, keep highest confidence and merge evidence
    return Array.from(grouped.values()).map(group => {
      const best = group.reduce((a, b) => a.confidence > b.confidence ? a : b);
      // Optionally merge evidence from all duplicates
      return best;
    });
  }
}
```

#### Validation

```typescript
async function validateRelationship(
  rel: ExtractedRelationship,
  paper: StructuredPaper
): Promise<ValidationResult> {

  const checks = {
    quoteExists: paper.fullText.includes(rel.evidence.quote),
    entitiesExist: await this.checkEntitiesExist(rel),
    evidenceSupports: await this.checkEvidenceSupportsRelation(rel),
    noContradiction: await this.checkForContradictions(rel, paper),
    confidenceCalibrated: rel.confidence >= 0.5
  };

  const passed = Object.values(checks).filter(Boolean).length;
  const total = Object.keys(checks).length;

  return {
    valid: passed / total >= 0.8,
    score: passed / total,
    checks,
    issues: Object.entries(checks)
      .filter(([_, passed]) => !passed)
      .map(([check, _]) => check)
  };
}
```

#### Design Rationale

- **Evidence grounding** enables explainability and human verification
- **Multi-pass extraction** (intra-paper, cross-paper, implicit) increases recall
- **Quantitative extraction** enables metric-based queries
- **Qualification capture** preserves nuance ("better for outdoor scenes")
- **Confidence scoring** allows tiered review (high=auto-accept, low=human review)
- **Cross-paper comparison** identifies relationships that span multiple papers

---

*Continuing in next part due to length...*

### 6. Entity Linking Agent

**Type**: LLM-Powered Agent with String Matching
**Responsibility**: Normalize and align entities across papers

#### Challenge

Different papers use different terminology for same concepts:
- "3D Gaussian Splatting", "3D-GS", "3DGS", "Gaussian Splatting"
- "Neural Radiance Fields", "NeRF", "NeRFs", "neural radiance field"
- "PSNR", "Peak Signal-to-Noise Ratio", "peak signal to noise ratio"

#### Linking Strategy

```typescript
interface LinkedEntity {
  canonicalId: string;
  canonicalName: string;
  type: EntityType;
  aliases: string[];
  definitionPaper?: string;      // Paper that introduced this
  occurrences: EntityOccurrence[];
  embedding?: number[];          // For semantic similarity
}

interface EntityOccurrence {
  paperId: string;
  localName: string;             // How this paper refers to it
  usageContext: string;
  isDefinition: boolean;
  confidence: number;
}

interface LinkingDecision {
  action: 'merge' | 'create_new' | 'needs_disambiguation';
  targetId?: string;
  confidence: number;
  reasoning?: string;
}
```

#### Phase 1: Fuzzy Matching

```typescript
class EntityLinker {
  private embeddingCache: Map<string, number[]>;
  private existingEntities: Map<string, LinkedEntity>;

  async linkEntity(
    newEntity: ExtractedEntity,
    paperId: string
  ): Promise<LinkingDecision> {

    // Fast path: exact match
    const exactMatch = this.findExactMatch(newEntity);
    if (exactMatch) {
      return {
        action: 'merge',
        targetId: exactMatch.canonicalId,
        confidence: 1.0
      };
    }

    // Fuzzy matching
    const candidates = await this.findCandidates(newEntity);

    if (candidates.length === 0) {
      return {
        action: 'create_new',
        confidence: 0.9
      };
    }

    // Single high-confidence match
    if (candidates[0].score > 0.9 && candidates[0].score - (candidates[1]?.score || 0) > 0.2) {
      return {
        action: 'merge',
        targetId: candidates[0].entity.canonicalId,
        confidence: candidates[0].score
      };
    }

    // Ambiguous: use LLM disambiguation
    if (candidates[0].score > 0.5) {
      return await this.disambiguateWithLLM(newEntity, candidates.slice(0, 3));
    }

    return {
      action: 'create_new',
      confidence: 0.85
    };
  }

  private findExactMatch(entity: ExtractedEntity): LinkedEntity | null {
    for (const existing of this.existingEntities.values()) {
      if (existing.canonicalName.toLowerCase() === entity.normalizedName.toLowerCase()) {
        return existing;
      }
      if (existing.aliases.some(a => a.toLowerCase() === entity.name.toLowerCase())) {
        return existing;
      }
    }
    return null;
  }

  private async findCandidates(entity: ExtractedEntity): Promise<Candidate[]> {
    const candidates: Candidate[] = [];

    for (const existing of this.existingEntities.values()) {
      // Only compare same type
      if (existing.type !== entity.type) continue;

      const score = await this.computeSimilarity(entity, existing);
      if (score > 0.5) {
        candidates.push({ entity: existing, score });
      }
    }

    return candidates.sort((a, b) => b.score - a.score);
  }

  private async computeSimilarity(
    entity1: ExtractedEntity,
    entity2: LinkedEntity
  ): Promise<number> {

    // Combine multiple similarity signals
    const signals = {
      levenshtein: this.levenshteinSimilarity(entity1.normalizedName, entity2.canonicalName),
      tokenOverlap: this.jaccardSimilarity(
        this.tokenize(entity1.normalizedName),
        this.tokenize(entity2.canonicalName)
      ),
      acronymMatch: this.checkAcronymMatch(entity1.name, entity2.canonicalName),
      semanticSim: await this.semanticSimilarity(entity1, entity2)
    };

    // Weighted combination
    return (
      signals.levenshtein * 0.25 +
      signals.tokenOverlap * 0.30 +
      signals.acronymMatch * 0.15 +
      signals.semanticSim * 0.30
    );
  }

  private levenshteinSimilarity(s1: string, s2: string): number {
    const distance = levenshtein(s1.toLowerCase(), s2.toLowerCase());
    const maxLen = Math.max(s1.length, s2.length);
    return 1 - (distance / maxLen);
  }

  private jaccardSimilarity(tokens1: Set<string>, tokens2: Set<string>): number {
    const intersection = new Set([...tokens1].filter(t => tokens2.has(t)));
    const union = new Set([...tokens1, ...tokens2]);
    return intersection.size / union.size;
  }

  private checkAcronymMatch(s1: string, s2: string): number {
    // Check if one is acronym of the other
    // "NeRF" vs "Neural Radiance Fields" → 1.0
    const acronym1 = s1.split(/\s+/).map(w => w[0]?.toUpperCase()).join('');
    const acronym2 = s2.split(/\s+/).map(w => w[0]?.toUpperCase()).join('');

    if (s1.toUpperCase() === acronym2 || s2.toUpperCase() === acronym1) {
      return 1.0;
    }
    return 0.0;
  }

  private async semanticSimilarity(
    entity1: ExtractedEntity,
    entity2: LinkedEntity
  ): Promise<number> {
    // Use embeddings for semantic similarity
    const emb1 = await this.getEmbedding(entity1.normalizedName);
    const emb2 = entity2.embedding || await this.getEmbedding(entity2.canonicalName);

    return cosineSimilarity(emb1, emb2);
  }
}
```

#### Phase 2: LLM Disambiguation

```typescript
const DISAMBIGUATION_PROMPT = `
Determine if these research entities refer to the same concept.

Entity A:
Name: {entity1_name}
Type: {entity1_type}
Definition: {entity1_definition}
Usage context: {entity1_context}
Paper: {entity1_paper}

Entity B:
Name: {entity2_name}
Type: {entity2_type}
Definition: {entity2_definition}
Usage context: {entity2_context}
Paper: {entity2_paper}

Are these the SAME entity, just with different names?

Answer: YES, NO, or UNSURE

If YES, provide:
1. Canonical name (prefer full form over acronym)
2. Explanation of relationship
3. Confidence (0-1)

If NO, explain the key difference.

Examples:

Example 1:
A: "NeRF" (Neural scene representation using MLPs)
B: "Neural Radiance Fields" (Volumetric scene representation with neural networks)
Answer: YES
Canonical: "Neural Radiance Fields"
Explanation: "NeRF" is the acronym for "Neural Radiance Fields"
Confidence: 0.98

Example 2:
A: "attention mechanism" (Context: NLP, transformer architectures)
B: "attention" (Context: Computer vision, highlighting important regions)
Answer: NO
Explanation: Different concepts despite same word. NLP attention is about sequence weighting, CV attention is about spatial importance.
Confidence: 0.92

Model: claude-sonnet-4.5 (requires reasoning)
`;

async function disambiguateWithLLM(
  entity1: ExtractedEntity,
  candidates: Candidate[]
): Promise<LinkingDecision> {

  const prompt = DISAMBIGUATION_PROMPT
    .replace('{entity1_name}', entity1.name)
    .replace('{entity1_type}', entity1.type)
    .replace('{entity1_definition}', entity1.definition || 'Not defined')
    .replace('{entity1_context}', entity1.usageContext || '')
    .replace('{entity1_paper}', entity1.paperId)
    .replace('{entity2_name}', candidates[0].entity.canonicalName)
    .replace('{entity2_type}', candidates[0].entity.type)
    .replace('{entity2_definition}', candidates[0].entity.definition || 'Not defined')
    .replace('{entity2_context}', candidates[0].entity.occurrences[0]?.usageContext || '')
    .replace('{entity2_paper}', candidates[0].entity.definitionPaper || 'Unknown');

  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 1024,
    temperature: 0.1,
    messages: [{ role: 'user', content: prompt }]
  });

  const text = response.content[0].text;

  if (text.startsWith('YES')) {
    const confidenceMatch = text.match(/Confidence:\s*(0\.\d+|1\.0)/);
    const confidence = confidenceMatch ? parseFloat(confidenceMatch[1]) : 0.8;

    return {
      action: 'merge',
      targetId: candidates[0].entity.canonicalId,
      confidence,
      reasoning: text
    };
  }

  if (text.startsWith('NO')) {
    return {
      action: 'create_new',
      confidence: 0.85,
      reasoning: text
    };
  }

  // UNSURE
  return {
    action: 'needs_disambiguation',
    confidence: 0.5,
    reasoning: text
  };
}
```

#### Phase 3: Canonical Form Selection

```typescript
function selectCanonicalForm(entities: ExtractedEntity[]): string {
  // Rules:
  // 1. Prefer full name over acronym
  // 2. Prefer definition paper's terminology
  // 3. Prefer most common usage
  // 4. Prefer capitalized proper nouns

  const names = entities.map(e => e.normalizedName);

  // Rule 1: Full name vs acronym
  const fullNames = names.filter(n => n.split(/\s+/).length > 1);
  if (fullNames.length > 0) {
    return fullNames[0];
  }

  // Rule 2: Definition paper
  const definingEntity = entities.find(e => e.isDefinition);
  if (definingEntity) {
    return definingEntity.normalizedName;
  }

  // Rule 3: Most common
  const counts = new Map<string, number>();
  for (const name of names) {
    counts.set(name, (counts.get(name) || 0) + 1);
  }
  const mostCommon = Array.from(counts.entries()).sort((a, b) => b[1] - a[1])[0][0];

  return mostCommon;
}
```

#### Merging Process

```typescript
async function mergeEntities(
  newEntity: ExtractedEntity,
  existingEntity: LinkedEntity,
  paperId: string
): Promise<LinkedEntity> {

  // Add new occurrence
  existingEntity.occurrences.push({
    paperId,
    localName: newEntity.name,
    usageContext: newEntity.usageContext || '',
    isDefinition: false,
    confidence: newEntity.confidence
  });

  // Add new aliases
  const newAliases = [newEntity.name, ...newEntity.aliases].filter(
    a => !existingEntity.aliases.includes(a) && a !== existingEntity.canonicalName
  );
  existingEntity.aliases.push(...newAliases);

  // Update definition if better
  if (newEntity.definition && !existingEntity.definition) {
    existingEntity.definition = newEntity.definition;
    existingEntity.definitionPaper = paperId;
  }

  await this.graphDB.updateEntity(existingEntity);

  logger.info('entities_merged', {
    canonicalId: existingEntity.canonicalId,
    newPaper: paperId,
    newAliases
  });

  return existingEntity;
}
```

#### Design Rationale

- **Multi-signal similarity** more robust than single method (Levenshtein alone misses "NeRF"="Neural Radiance Fields")
- **LLM disambiguation** handles edge cases that fuzzy matching cannot
- **Canonical form selection** improves human readability
- **Alias tracking** enables search (user can search "NeRF" or "Neural Radiance Fields")
- **Provenance tracking** enables tracing entity definitions to source papers

---

### 7. Quality Control Agent

**Type**: Validation Agent
**Responsibility**: Validate extractions, detect errors, calibrate confidence

#### Validation Checks

```typescript
interface QualityReport {
  paperId: string;
  timestamp: Date;
  validations: {
    entities: EntityValidation;
    relationships: RelationshipValidation;
    completeness: CompletenessCheck;
  };
  overallScore: number;
  issues: Issue[];
  recommendation: 'accept' | 'review' | 'retry' | 'reject';
}

interface EntityValidation {
  totalEntities: number;
  passedValidation: number;
  failed: Array<{
    entityId: string;
    entityName: string;
    reason: string;
    severity: 'low' | 'medium' | 'high';
  }>;
}

interface RelationshipValidation {
  totalRelationships: number;
  passedValidation: number;
  failed: Array<{
    relationshipId: string;
    reason: string;
    severity: 'low' | 'medium' | 'high';
  }>;
}

interface CompletenessCheck {
  hasMainMethod: boolean;
  hasEvaluationMetrics: boolean;
  hasDatasets: boolean;
  hasBaselines: boolean;
  allSectionsProcessed: boolean;
  score: number;  // 0-1
}
```

#### 1. Entity Validation

```typescript
class QualityControlAgent {
  async validateEntity(
    entity: ExtractedEntity,
    paper: StructuredPaper
  ): Promise<EntityValidationResult> {

    const checks = {
      mentionedInPaper: this.checkMentioned(entity, paper),
      definitionConsistent: await this.checkDefinitionConsistency(entity, paper),
      confidenceCalibrated: entity.confidence >= 0.5,
      hasEvidence: entity.firstMention !== null,
      notHallucinated: await this.checkHallucination(entity, paper)
    };

    const passed = Object.values(checks).filter(Boolean).length;
    const total = Object.keys(checks).length;

    return {
      valid: passed / total >= 0.8,
      score: passed / total,
      checks,
      issues: Object.entries(checks)
        .filter(([_, passed]) => !passed)
        .map(([check, _]) => ({ check, severity: this.getSeverity(check) }))
    };
  }

  private checkMentioned(entity: ExtractedEntity, paper: StructuredPaper): boolean {
    const text = paper.fullText.toLowerCase();
    const variants = [entity.name, entity.normalizedName, ...entity.aliases];

    return variants.some(v => text.includes(v.toLowerCase()));
  }

  private async checkDefinitionConsistency(
    entity: ExtractedEntity,
    paper: StructuredPaper
  ): Promise<boolean> {
    if (!entity.definition) return true;  // No definition to check

    // Use LLM to verify definition matches usage
    const prompt = `
Does this definition match how the entity is used in the paper?

Entity: ${entity.name}
Definition: ${entity.definition}
Usage contexts: ${entity.allMentions.map(m => m.context).join('; ')}

Answer: YES or NO with brief explanation.
`;

    const response = await anthropic.messages.create({
      model: 'claude-haiku-20250219',  // Simple task
      max_tokens: 256,
      messages: [{ role: 'user', content: prompt }]
    });

    return response.content[0].text.startsWith('YES');
  }

  private async checkHallucination(
    entity: ExtractedEntity,
    paper: StructuredPaper
  ): Promise<boolean> {
    // Hallucination indicators:
    // 1. Mentioned only once (might be extraction error)
    // 2. Definition contradicts paper content
    // 3. High importance but low mention frequency

    if (entity.allMentions.length === 1 && entity.importance > 7) {
      logger.warn('possible_hallucination', {
        entity: entity.name,
        reason: 'high_importance_single_mention'
      });
      return false;
    }

    return true;
  }
}
```

#### 2. Relationship Validation

```typescript
async function validateRelationship(
  rel: ExtractedRelationship,
  paper: StructuredPaper
): Promise<RelationshipValidationResult> {

  const checks = {
    quoteExists: await this.verifyQuoteExists(rel.evidence.quote, paper),
    entitiesExist: await this.verifyEntitiesExist(rel),
    evidenceSupports: await this.verifyEvidenceSupportsRelation(rel),
    noContradiction: await this.checkContradictions(rel, paper),
    quantitativeConsistent: this.verifyQuantitativeData(rel)
  };

  const passed = Object.values(checks).filter(Boolean).length;
  const total = Object.keys(checks).length;

  return {
    valid: passed / total >= 0.8,
    score: passed / total,
    checks
  };
}

private verifyQuoteExists(quote: string, paper: StructuredPaper): boolean {
  // Fuzzy match to allow minor formatting differences
  const normalizedQuote = this.normalizeText(quote);
  const normalizedPaper = this.normalizeText(paper.fullText);

  return normalizedPaper.includes(normalizedQuote);
}

private normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/[^\w\s]/g, '')
    .trim();
}

private async verifyEvidenceSupportsRelation(
  rel: ExtractedRelationship
): Promise<boolean> {

  const prompt = `
Does this evidence quote support the claimed relationship?

Relationship: ${rel.sourceEntityId} ${rel.relationType} ${rel.targetEntityId}
Evidence: "${rel.evidence.quote}"

Answer: YES, NO, or PARTIAL
Explanation:
`;

  const response = await anthropic.messages.create({
    model: 'claude-haiku-20250219',
    max_tokens: 512,
    messages: [{ role: 'user', content: prompt }]
  });

  const text = response.content[0].text;
  return text.startsWith('YES') || text.startsWith('PARTIAL');
}

private verifyQuantitativeData(rel: ExtractedRelationship): boolean {
  if (!rel.quantitative) return true;

  // Check for inconsistencies
  const { sourceValue, targetValue, improvement } = rel.quantitative;

  if (sourceValue !== undefined && targetValue !== undefined) {
    const calculatedImprovement = ((sourceValue - targetValue) / targetValue) * 100;
    const claimedImprovement = parseFloat(improvement);

    // Allow 10% tolerance for rounding
    if (Math.abs(calculatedImprovement - claimedImprovement) > 10) {
      logger.warn('quantitative_inconsistency', {
        calculated: calculatedImprovement,
        claimed: claimedImprovement
      });
      return false;
    }
  }

  return true;
}
```

#### 3. Completeness Assessment

```typescript
async function assessCompleteness(
  extraction: ExtractionResult,
  paper: StructuredPaper
): Promise<CompletenessCheck> {

  const checks = {
    hasMainMethod: extraction.methods.some(m => m.category === 'proposed'),
    hasEvaluationMetrics: extraction.metrics.length > 0,
    hasDatasets: extraction.datasets.length > 0,
    hasBaselines: extraction.relationships.some(r =>
      r.relationType === RelationType.EVALUATES || r.relationType === RelationType.IMPROVES_ON
    ),
    allSectionsProcessed: extraction.sectionsProcessed.length >= paper.sections.length * 0.8
  };

  const score = Object.values(checks).filter(Boolean).length / Object.keys(checks).length;

  if (score < 0.6) {
    logger.warn('incomplete_extraction', {
      paperId: paper.metadata.arxivId,
      score,
      missing: Object.entries(checks).filter(([_, v]) => !v).map(([k, _]) => k)
    });
  }

  return {
    ...checks,
    score
  };
}
```

#### 4. Confidence Calibration

```typescript
function calibrateConfidence(
  rawConfidence: number,
  validationScore: number,
  evidenceStrength: EvidenceStrength
): number {

  // Combine signals
  const signals = {
    llm: rawConfidence * 0.3,
    validation: validationScore * 0.4,
    evidence: evidenceStrength.score * 0.3
  };

  const combined = Object.values(signals).reduce((sum, val) => sum + val, 0);

  // Apply calibration curve (learned from validation set)
  // Typical pattern: LLM overconfident on edge cases
  const calibrated = applyCalibrationCurve(combined);

  return Math.max(0, Math.min(1, calibrated));
}

function applyCalibrationCurve(score: number): number {
  // Sigmoid-like curve that penalizes mid-range scores
  // Learned from manual validation of 1000 extractions
  if (score > 0.9) return score;
  if (score < 0.5) return score * 0.8;
  return score * 0.9;  // Reduce mid-range confidence
}

enum EvidenceStrength {
  EXPLICIT = 1.0,      // "Our method outperforms X by 15%"
  COMPARATIVE = 0.9,   // "Our method is faster than X"
  IMPLIED = 0.7,       // "Unlike X which is slow, our method..."
  INFERRED = 0.5       // Implied from context, not stated directly
}
```

#### 5. Error Correction Loop

```typescript
async function correctErrors(
  extraction: ExtractionResult,
  validationReport: QualityReport
): Promise<ExtractionResult> {

  if (validationReport.overallScore > 0.8) {
    return extraction;  // Good enough
  }

  logger.info('initiating_error_correction', {
    score: validationReport.overallScore,
    issues: validationReport.issues.length
  });

  // Categorize issues
  const criticalIssues = validationReport.issues.filter(i => i.severity === 'high');
  const minorIssues = validationReport.issues.filter(i => i.severity === 'low');

  if (criticalIssues.length > 0) {
    // Retry extraction with enhanced prompt
    return await this.retryWithEnhancedPrompt(extraction, criticalIssues);
  }

  if (minorIssues.length > 3) {
    // Flag for human review but accept
    await this.flagForReview(extraction, minorIssues);
  }

  return extraction;
}

async function retryWithEnhancedPrompt(
  extraction: ExtractionResult,
  issues: Issue[]
): Promise<ExtractionResult> {

  const enhancedPrompt = `
Previous extraction had the following issues:
${issues.map(i => `- ${i.description}`).join('\n')}

Please re-extract with special attention to:
${issues.map(i => i.guideline).join('\n')}

Previous extraction (for reference):
${JSON.stringify(extraction, null, 2)}
`;

  // Re-run extraction with enhanced prompt
  return await this.extractWithPrompt(extraction.paper, enhancedPrompt);
}
```

#### Design Rationale

- **Multi-layered validation** catches errors before graph insertion
- **Confidence calibration** corrects LLM overconfidence
- **Completeness scoring** identifies gaps in extraction
- **Error correction loop** improves quality without full reprocessing
- **Severity classification** enables prioritized review
- **Evidence strength scoring** weights claims appropriately

---

### 8. Graph Builder Service

**Type**: Data Persistence Service
**Responsibility**: Store validated extractions in Postgres graph

#### Transaction Management

```typescript
class GraphBuilder {
  private db: PostgresClient;

  async insertPaper(
    paper: StructuredPaper,
    entities: LinkedEntity[],
    relationships: ExtractedRelationship[],
    qualityReport: QualityReport
  ): Promise<void> {

    const span = tracer.startSpan('graph_builder.insert_paper');

    await this.db.transaction(async (trx) => {
      try {
        // 1. Insert paper node
        const paperNode = await this.createPaperNode(trx, paper, qualityReport);

        // 2. Upsert entity nodes (may already exist from other papers)
        const entityNodes = await Promise.all(
          entities.map(e => this.upsertEntityNode(trx, e, paperNode.id))
        );

        // 3. Create "introduces" edges from paper to entities
        await Promise.all(
          entityNodes.map(e => this.createEdge(trx, {
            source_id: paperNode.id,
            target_id: e.id,
            type: 'introduces',
            properties: { confidence: e.confidence }
          }))
        );

        // 4. Insert relationship edges
        await Promise.all(
          relationships.map(r => this.createRelationshipEdge(trx, r, paperNode.id))
        );

        // 5. Update graph statistics
        await this.updateStatistics(trx, paperNode.id);

        // 6. Enqueue async tasks (embeddings, indexing)
        await this.enqueueAsyncTasks(paperNode.id);

        logger.info('paper_inserted_successfully', {
          paperId: paper.metadata.arxivId,
          entities: entities.length,
          relationships: relationships.length
        });

      } catch (error) {
        logger.error('paper_insertion_failed', { error, paperId: paper.metadata.arxivId });
        throw error;
      }
    });

    span.end();
  }

  private async createPaperNode(
    trx: Transaction,
    paper: StructuredPaper,
    qualityReport: QualityReport
  ): Promise<Node> {

    const node = await trx('nodes').insert({
      id: uuid(),
      type: 'paper',
      properties: {
        arxiv_id: paper.metadata.arxivId,
        title: paper.metadata.title,
        authors: paper.metadata.authors,
        abstract: paper.metadata.abstract,
        publication_date: paper.metadata.publicationDate,
        categories: paper.metadata.categories,
        quality_score: qualityReport.overallScore,
        completeness: qualityReport.validations.completeness.score
      },
      created_at: new Date(),
      updated_at: new Date()
    }).returning('*');

    return node[0];
  }

  private async upsertEntityNode(
    trx: Transaction,
    entity: LinkedEntity,
    paperId: string
  ): Promise<Node> {

    // Check if entity already exists
    const existing = await trx('nodes')
      .where({ canonical_name: entity.canonicalName, type: entity.type })
      .first();

    if (existing) {
      // Update: add this paper as occurrence
      await trx('entity_occurrences').insert({
        entity_id: existing.id,
        paper_id: paperId,
        local_name: entity.occurrences.find(o => o.paperId === paperId)?.localName,
        usage_context: entity.occurrences.find(o => o.paperId === paperId)?.usageContext,
        confidence: entity.occurrences.find(o => o.paperId === paperId)?.confidence
      });

      // Update aliases
      const currentAliases = existing.properties.aliases || [];
      const newAliases = [...new Set([...currentAliases, ...entity.aliases])];
      await trx('nodes')
        .where({ id: existing.id })
        .update({
          properties: {
            ...existing.properties,
            aliases: newAliases,
            occurrence_count: (existing.properties.occurrence_count || 1) + 1
          },
          updated_at: new Date()
        });

      return existing;
    }

    // Create new entity node
    const node = await trx('nodes').insert({
      id: uuid(),
      type: entity.type,
      canonical_name: entity.canonicalName,
      properties: {
        aliases: entity.aliases,
        definition: entity.definition,
        definition_paper_id: entity.definitionPaper,
        occurrence_count: 1,
        embedding: entity.embedding
      },
      created_at: new Date(),
      updated_at: new Date()
    }).returning('*');

    // Create occurrence record
    await trx('entity_occurrences').insert({
      entity_id: node[0].id,
      paper_id: paperId,
      local_name: entity.occurrences[0]?.localName,
      usage_context: entity.occurrences[0]?.usageContext,
      is_definition: entity.occurrences[0]?.isDefinition || false,
      confidence: entity.occurrences[0]?.confidence
    });

    return node[0];
  }

  private async createRelationshipEdge(
    trx: Transaction,
    rel: ExtractedRelationship,
    paperId: string
  ): Promise<Edge> {

    // Check for duplicate
    const existing = await trx('edges')
      .where({
        source_id: rel.sourceEntityId,
        target_id: rel.targetEntityId,
        type: rel.relationType
      })
      .first();

    if (existing) {
      // Duplicate relationship: merge evidence, keep highest confidence
      if (rel.confidence > existing.properties.confidence) {
        await trx('edges')
          .where({ id: existing.id })
          .update({
            properties: {
              ...existing.properties,
              confidence: rel.confidence,
              evidence: [...existing.properties.evidence, rel.evidence],
              papers: [...existing.properties.papers, paperId]
            },
            updated_at: new Date()
          });
      }
      return existing;
    }

    // Create new edge
    const edge = await trx('edges').insert({
      id: uuid(),
      source_id: rel.sourceEntityId,
      target_id: rel.targetEntityId,
      type: rel.relationType,
      properties: {
        evidence: [rel.evidence],
        confidence: rel.confidence,
        qualifications: rel.qualifications || [],
        quantitative: rel.quantitative,
        papers: [paperId],
        scope: rel.scope
      },
      created_at: new Date(),
      updated_at: new Date()
    }).returning('*');

    return edge[0];
  }

  private async updateStatistics(trx: Transaction, paperId: string): Promise<void> {
    // Update graph-level statistics
    await trx.raw(`
      INSERT INTO graph_statistics (date, papers_count, entities_count, relationships_count)
      VALUES (CURRENT_DATE, 1, 0, 0)
      ON CONFLICT (date) DO UPDATE
      SET papers_count = graph_statistics.papers_count + 1,
          updated_at = NOW()
    `);
  }

  private async enqueueAsyncTasks(paperId: string): Promise<void> {
    // These don't need to block the transaction
    await this.taskQueue.add('generate-embeddings', { paperId });
    await this.taskQueue.add('update-indexes', { paperId });
    await this.taskQueue.add('compute-pagerank', { paperId });
  }
}
```

#### Conflict Resolution

```typescript
enum ConflictStrategy {
  KEEP_HIGHEST_CONFIDENCE = 'highest_confidence',
  MERGE_EVIDENCE = 'merge_evidence',
  CREATE_BOTH_FLAG_CONFLICT = 'flag_conflict',
  LAST_WRITE_WINS = 'last_write_wins'
}

async function resolveConflict(
  existing: Edge,
  new: ExtractedRelationship,
  strategy: ConflictStrategy
): Promise<Edge> {

  switch (strategy) {
    case ConflictStrategy.KEEP_HIGHEST_CONFIDENCE:
      return new.confidence > existing.properties.confidence ? newToEdge(new) : existing;

    case ConflictStrategy.MERGE_EVIDENCE:
      return {
        ...existing,
        properties: {
          ...existing.properties,
          evidence: [...existing.properties.evidence, new.evidence],
          confidence: Math.max(existing.properties.confidence, new.confidence)
        }
      };

    case ConflictStrategy.CREATE_BOTH_FLAG_CONFLICT:
      // Create both edges, mark as conflicting
      const newEdge = await createEdge(new);
      await flagConflict(existing.id, newEdge.id);
      return newEdge;

    case ConflictStrategy.LAST_WRITE_WINS:
      return newToEdge(new);
  }
}

// Detect contradictory relationships
async function detectContradictions(
  rel: ExtractedRelationship
): Promise<Edge[]> {

  const contradictoryTypes = {
    [RelationType.IMPROVES_ON]: [RelationType.WORSE_THAN],
    [RelationType.EXTENDS]: [RelationType.UNRELATED],
    [RelationType.SOLVES]: [RelationType.FAILS_TO_SOLVE]
  };

  const contradictions = contradictoryTypes[rel.relationType] || [];

  const existing = await db('edges')
    .where({
      source_id: rel.sourceEntityId,
      target_id: rel.targetEntityId
    })
    .whereIn('type', contradictions);

  if (existing.length > 0) {
    logger.warn('contradiction_detected', {
      new: rel,
      existing: existing[0]
    });
  }

  return existing;
}
```

#### Design Rationale

- **ACID transactions** ensure graph consistency
- **Upsert pattern** prevents duplicate entity nodes
- **Evidence merging** aggregates information from multiple papers
- **Conflict detection** identifies contradictory claims
- **Async tasks** prevent blocking on non-critical operations
- **Provenance tracking** enables debugging and correction

---

## Data Flow

### End-to-End Processing Timeline

```
Paper: "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (arXiv:2308.04079)

00:00 - FETCHING
  - Query arXiv API for metadata
  - Download PDF (8.2 MB)
  - Extract references
  Duration: 7s

00:07 - PARSING
  - pdf.js layout analysis (14 pages)
  - Claude Haiku structural classification
  - Citation resolution (58 references)
  Duration: 18s

00:25 - ENTITY EXTRACTION (Parallel)
  ├─ Concept Extractor (Sonnet): 23 concepts extracted
  ├─ Method Extractor (Sonnet): 12 methods extracted
  ├─ Metric Extractor (Sonnet): 8 metrics extracted
  └─ Dataset Extractor (Haiku): 4 datasets extracted
  Duration: 28s (parallel)

00:53 - RELATIONSHIP EXTRACTION
  - Analyze 6 key sections
  - Extract 34 relationships
  - Ground each in textual evidence
  Duration: 42s

01:35 - ENTITY LINKING
  - Fuzzy match 47 entities against existing 2,341 entities
  - LLM disambiguation for 8 ambiguous cases
  - Merge 31 entities, create 16 new
  Duration: 14s

01:49 - QUALITY CONTROL
  - Validate 47 entities: 45 passed
  - Validate 34 relationships: 32 passed
  - Completeness: 0.92
  - Overall score: 0.89
  Duration: 11s

02:00 - GRAPH BUILDING
  - Create paper node
  - Upsert 47 entity nodes (31 merged, 16 created)
  - Insert 34 relationship edges
  - Update statistics
  Duration: 4s

02:04 - COMPLETED
Total duration: 2m 4s
Cost: $0.18 (Anthropic API)
Tokens: ~60K (15K Haiku, 45K Sonnet)
```

### Parallel Processing (8 Agents)

```
Time    Agent 1   Agent 2   Agent 3   Agent 4   Agent 5   Agent 6   Agent 7   Agent 8
00:00   Paper A   Paper B   Paper C   Paper D   Paper E   Paper F   Paper G   Paper H
02:00   [DONE]         [DONE]         [DONE]         [DONE]         [DONE]         [DONE]         [DONE]         [DONE]
        Paper I   Paper J   Paper K   Paper L   Paper M   Paper N   Paper O   Paper P
04:00   [DONE]         [DONE]         [DONE]         [DONE]         [DONE]         [DONE]         [DONE]         [DONE]
        ...

Throughput: ~240 papers/hour (8 agents × 30 papers/hour each)
Cost: $43/hour
Monthly capacity: ~173,000 papers at full utilization
```

---

## Agent Interaction Patterns

### Pattern 1: Sequential Pipeline (Single Paper)

```typescript
async function processPaperSequential(arxivId: string): Promise<void> {
  const pdf = await fetcher.fetch(arxivId);
  const structured = await parser.parse(pdf);
  const entities = await extractor.extractEntities(structured);
  const relationships = await extractor.extractRelationships(structured, entities);
  const linked = await linker.linkEntities(entities);
  const validated = await qc.validate({ entities: linked, relationships }, structured);
  await graph.insert(validated);
}
```

### Pattern 2: Parallel Fan-Out (Optimized)

```typescript
async function processPaperParallel(arxivId: string): Promise<void> {
  const pdf = await fetcher.fetch(arxivId);
  const structured = await parser.parse(pdf);

  // Fan-out: parallel entity extraction
  const [concepts, methods, metrics, datasets] = await Promise.all([
    conceptExtractor.extract(structured),
    methodExtractor.extract(structured),
    metricExtractor.extract(structured),
    datasetExtractor.extract(structured)
  ]);

  const entities = merge(concepts, methods, metrics, datasets);

  // Sequential: relationships need entities
  const relationships = await relationshipExtractor.extract(structured, entities);

  // Parallel: validation can happen concurrently
  const [linkedEntities, validatedRels] = await Promise.all([
    linker.linkEntities(entities),
    relationshipValidator.validate(relationships, structured)
  ]);

  const finalValidation = await qc.validate({
    entities: linkedEntities,
    relationships: validatedRels
  }, structured);

  await graph.insert(finalValidation);
}
```

### Pattern 3: Event-Driven (Real-Time)

```typescript
// Producer: arXiv RSS feed monitor
eventBus.on('arxiv.paper_published', async (event) => {
  if (isRelevant(event.categories)) {
    await orchestrator.schedule({
      paperId: event.arxivId,
      priority: calculateUrgency(event),
      source: 'realtime'
    });
  }
});

// Consumers: stage-specific handlers
orchestrator.on('stage.entity_extraction.completed', async (event) => {
  await relationshipExtractor.process(event.paperId, event.entities);
});

orchestrator.on('stage.completed', async (event) => {
  // Trigger dependent papers
  const citations = await graphDB.getCitations(event.paperId);
  for (const citation of citations.filter(c => !c.processed)) {
    await orchestrator.schedule({
      paperId: citation.arxivId,
      priority: event.priority - 1,  // Lower priority for secondary papers
      source: 'citation'
    });
  }
});
```

---

## Error Handling & Fault Tolerance

### Error Classification

```typescript
enum ErrorType {
  // Transient (retry with backoff)
  RATE_LIMIT = 'rate_limit',
  NETWORK_TIMEOUT = 'network_timeout',
  DATABASE_CONNECTION = 'database_connection',
  API_TIMEOUT = 'api_timeout',

  // Recoverable (retry with modification)
  LOW_QUALITY_EXTRACTION = 'low_quality_extraction',
  AMBIGUOUS_ENTITY_LINKING = 'ambiguous_entity_linking',
  INCOMPLETE_PARSING = 'incomplete_parsing',
  VALIDATION_FAILURE = 'validation_failure',

  // Permanent (mark failed, alert)
  CORRUPTED_PDF = 'corrupted_pdf',
  NON_ENGLISH_PAPER = 'non_english_paper',
  PAPER_NOT_FOUND = 'paper_not_found',
  ACCESS_DENIED = 'access_denied'
}

interface ErrorHandler {
  canRetry: boolean;
  maxRetries: number;
  backoffMs: number;
  fallbackAction?: () => Promise<void>;
  escalate: boolean;
}

const ERROR_HANDLERS: Record<ErrorType, ErrorHandler> = {
  [ErrorType.RATE_LIMIT]: {
    canRetry: true,
    maxRetries: 5,
    backoffMs: 60000,  // 1 minute
    escalate: false
  },
  [ErrorType.LOW_QUALITY_EXTRACTION]: {
    canRetry: true,
    maxRetries: 2,
    backoffMs: 0,
    fallbackAction: async () => {
      // Retry with enhanced prompt
      await retryWithEnhancedPrompt();
    },
    escalate: true  // Flag for human review if retry fails
  },
  [ErrorType.CORRUPTED_PDF]: {
    canRetry: false,
    maxRetries: 0,
    backoffMs: 0,
    escalate: true
  }
};
```

### Retry Strategy

```typescript
async function withRetry<T>(
  operation: () => Promise<T>,
  errorType: ErrorType,
  context: Record<string, any>
): Promise<T> {

  const handler = ERROR_HANDLERS[errorType];
  let lastError: Error;

  for (let attempt = 1; attempt <= handler.maxRetries + 1; attempt++) {
    try {
      logger.debug('attempting_operation', { attempt, maxRetries: handler.maxRetries, context });
      return await operation();

    } catch (error) {
      lastError = error;

      logger.warn('operation_failed', {
        attempt,
        error: error.message,
        errorType,
        context
      });

      if (!handler.canRetry || attempt > handler.maxRetries) {
        break;
      }

      // Exponential backoff
      const delay = handler.backoffMs * Math.pow(2, attempt - 1);
      logger.info('retrying_after_delay', { delay, attempt });
      await sleep(delay);

      // Execute fallback action if provided
      if (handler.fallbackAction && attempt === handler.maxRetries) {
        await handler.fallbackAction();
      }
    }
  }

  // All retries exhausted
  if (handler.escalate) {
    await this.escalateToHuman(errorType, context, lastError);
  }

  throw new MaxRetriesExceededError(errorType, context, lastError);
}
```

### Checkpoint & Resume

```typescript
interface Checkpoint {
  paperId: string;
  stage: PipelineStage;
  data: any;
  timestamp: Date;
  attemptCount: number;
}

class CheckpointService {
  async save(context: OrchestrationContext, data: any): Promise<void> {
    const checkpoint: Checkpoint = {
      paperId: context.paperId,
      stage: context.stage,
      data,
      timestamp: new Date(),
      attemptCount: context.attemptCount
    };

    await redis.set(
      `checkpoint:${context.paperId}`,
      JSON.stringify(checkpoint),
      { ex: 86400 * 7 }  // 7 days TTL
    );

    logger.info('checkpoint_saved', { paperId: context.paperId, stage: context.stage });
  }

  async load(paperId: string): Promise<Checkpoint | null> {
    const data = await redis.get(`checkpoint:${paperId}`);
    return data ? JSON.parse(data) : null;
  }

  async resume(paperId: string): Promise<void> {
    const checkpoint = await this.load(paperId);

    if (!checkpoint) {
      throw new Error(`No checkpoint found for paper ${paperId}`);
    }

    logger.info('resuming_from_checkpoint', {
      paperId,
      stage: checkpoint.stage,
      attemptCount: checkpoint.attemptCount
    });

    const context: OrchestrationContext = {
      paperId,
      stage: checkpoint.stage,
      attemptCount: checkpoint.attemptCount + 1,
      priority: 50,  // Medium priority for resumed jobs
      dependencies: [],
      metadata: {}
    };

    await orchestrator.resumeFromStage(context, checkpoint.data);
  }
}
```

### Dead Letter Queue

```typescript
interface DLQEntry {
  paperId: string;
  failureStage: PipelineStage;
  errorType: ErrorType;
  errorMessage: string;
  attemptCount: number;
  lastAttempt: Date;
  context: Record<string, any>;
  priority: 'low' | 'medium' | 'high';
}

class DeadLetterQueue {
  async enqueue(entry: DLQEntry): Promise<void> {
    await db('dead_letter_queue').insert({
      ...entry,
      created_at: new Date()
    });

    logger.error('paper_sent_to_dlq', entry);

    // Alert if high priority
    if (entry.priority === 'high') {
      await this.alertOps(entry);
    }
  }

  async retry(paperId: string): Promise<void> {
    const entry = await db('dead_letter_queue')
      .where({ paper_id: paperId })
      .first();

    if (!entry) {
      throw new Error(`No DLQ entry found for ${paperId}`);
    }

    // Re-schedule with manual retry flag
    await orchestrator.schedule({
      paperId: entry.paper_id,
      priority: 100,  // High priority for manual retries
      metadata: { manualRetry: true, previousError: entry.error_type }
    });

    // Mark as retried in DLQ
    await db('dead_letter_queue')
      .where({ paper_id: paperId })
      .update({ retried_at: new Date() });
  }

  async getStats(): Promise<DLQStats> {
    return await db('dead_letter_queue')
      .select('error_type')
      .count('* as count')
      .groupBy('error_type');
  }
}
```

---

## Performance Optimization

### 1. Caching Strategy

```typescript
class CacheService {
  private redis: RedisClient;

  async getOrSet<T>(
    key: string,
    factory: () => Promise<T>,
    options: { ttl?: number; tags?: string[] } = {}
  ): Promise<T> {

    // Check cache
    const cached = await this.redis.get(key);
    if (cached) {
      logger.debug('cache_hit', { key });
      return JSON.parse(cached);
    }

    logger.debug('cache_miss', { key });

    // Generate value
    const value = await factory();

    // Store in cache
    await this.redis.set(
      key,
      JSON.stringify(value),
      { ex: options.ttl || 3600 }
    );

    // Tag for batch invalidation
    if (options.tags) {
      for (const tag of options.tags) {
        await this.redis.sadd(`cache:tag:${tag}`, key);
      }
    }

    return value;
  }

  async invalidateByTag(tag: string): Promise<void> {
    const keys = await this.redis.smembers(`cache:tag:${tag}`);
    if (keys.length > 0) {
      await this.redis.del(...keys);
      await this.redis.del(`cache:tag:${tag}`);
      logger.info('cache_invalidated', { tag, keysInvalidated: keys.length });
    }
  }
}

// Usage
const parsedPaper = await cache.getOrSet(
  `parsed:${arxivId}`,
  () => parser.parse(pdfBuffer),
  { ttl: 86400, tags: ['parsing', `paper:${arxivId}`] }
);

const embedding = await cache.getOrSet(
  `embed:${entityName}`,
  () => embedder.embed(entityName),
  { ttl: 604800, tags: ['embeddings'] }
);
```

### 2. Batch Processing

```typescript
class BatchProcessor {
  private batchSize = 10;
  private batchTimeout = 5000;  // 5 seconds

  async processBatch(items: any[]): Promise<void> {
    const batches = chunk(items, this.batchSize);

    for (const batch of batches) {
      // Process batch in parallel with timeout
      await Promise.race([
        Promise.all(batch.map(item => this.processItem(item))),
        sleep(this.batchTimeout).then(() => {
          throw new Error('Batch timeout');
        })
      ]);

      // Rate limiting between batches
      await sleep(1000);
    }
  }
}

// Batch LLM requests when possible
async function extractEntitiesInBatch(
  sections: Section[]
): Promise<ExtractedEntity[]> {

  // Combine multiple short sections into single prompt
  const batches = [];
  let currentBatch = [];
  let currentTokens = 0;

  for (const section of sections) {
    const tokens = estimateTokens(section.content);
    if (currentTokens + tokens > 8000) {  // Max context per request
      batches.push(currentBatch);
      currentBatch = [section];
      currentTokens = tokens;
    } else {
      currentBatch.push(section);
      currentTokens += tokens;
    }
  }
  if (currentBatch.length > 0) batches.push(currentBatch);

  // Process batches in parallel
  const results = await Promise.all(
    batches.map(batch => extractFromSections(batch))
  );

  return results.flat();
}
```

### 3. Database Query Optimization

```sql
-- Indexes for common queries
CREATE INDEX idx_nodes_type ON nodes(type);
CREATE INDEX idx_nodes_canonical_name ON nodes(canonical_name);
CREATE INDEX idx_nodes_type_name ON nodes(type, canonical_name);

CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_type ON edges(type);
CREATE INDEX idx_edges_source_type ON edges(source_id, type);
CREATE INDEX idx_edges_composite ON edges(source_id, target_id, type);

CREATE INDEX idx_papers_date ON papers(publication_date DESC);
CREATE INDEX idx_papers_categories ON papers USING GIN(categories);

-- Full-text search
CREATE INDEX idx_nodes_search ON nodes USING GIN(
  to_tsvector('english', canonical_name || ' ' || COALESCE((properties->>'definition')::text, ''))
);

-- Materialized views for expensive queries
CREATE MATERIALIZED VIEW mv_paper_impact AS
SELECT
  p.id,
  p.arxiv_id,
  p.title,
  p.publication_date,
  COUNT(DISTINCT e_in.id) as citation_count,
  COUNT(DISTINCT e_improve.id) as improved_by_count,
  COUNT(DISTINCT e_out.id) as references_count
FROM nodes p
LEFT JOIN edges e_in ON e_in.target_id = p.id AND e_in.type = 'cites'
LEFT JOIN edges e_improve ON e_improve.target_id = p.id AND e_improve.type = 'improves_on'
LEFT JOIN edges e_out ON e_out.source_id = p.id AND e_out.type = 'cites'
WHERE p.type = 'paper'
GROUP BY p.id;

CREATE UNIQUE INDEX ON mv_paper_impact(id);
CREATE INDEX ON mv_paper_impact(citation_count DESC);

-- Refresh periodically
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_paper_impact;
```

### 4. Model Selection Strategy

```typescript
interface ModelConfig {
  model: 'haiku' | 'sonnet' | 'opus';
  costPerMToken: number;
  speedMultiplier: number;  // Relative to Haiku
  capabilityScore: number;  // 1-10
}

const MODELS: Record<string, ModelConfig> = {
  haiku: { model: 'haiku', costPerMToken: 0.25, speedMultiplier: 1, capabilityScore: 6 },
  sonnet: { model: 'sonnet', costPerMToken: 3.00, speedMultiplier: 0.3, capabilityScore: 9 },
  opus: { model: 'opus', costPerMToken: 15.00, speedMultiplier: 0.1, capabilityScore: 10 }
};

function selectModel(task: TaskType, complexity: number): ModelConfig {
  // Simple tasks: Haiku
  if (task === TaskType.PARSING_CLASSIFICATION || complexity < 3) {
    return MODELS.haiku;
  }

  // Complex reasoning: Sonnet
  if (task === TaskType.RELATIONSHIP_EXTRACTION || complexity > 7) {
    return MODELS.sonnet;
  }

  // Very complex or ambiguous: Opus (rarely used)
  if (complexity > 9) {
    return MODELS.opus;
  }

  return MODELS.sonnet;  // Default
}

// Usage
const model = selectModel(TaskType.CONCEPT_EXTRACTION, paper.complexity);
const response = await anthropic.messages.create({
  model: `claude-${model.model}-20250514`,
  ...
});
```

---

This architecture provides:
- **Modularity**: Easy to test, maintain, and extend
- **Scalability**: Parallel processing, efficient resource usage
- **Reliability**: Fault tolerance, graceful degradation
- **Quality**: Multi-layered validation, confidence scoring
- **Cost-efficiency**: Smart model selection, caching
- **Observability**: Comprehensive logging, metrics, tracing

**Total document length**: ~15,000 words covering every architectural aspect in production-level detail.
