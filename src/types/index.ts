/**
 * Core Type Definitions for Research Knowledge Graph System
 *
 * This file contains all TypeScript interfaces and enums used throughout
 * the agentic knowledge graph system.
 */

// ============================================================================
// Entity Types
// ============================================================================

export enum EntityType {
  CONCEPT = 'concept',
  METHOD = 'method',
  METRIC = 'metric',
  DATASET = 'dataset',
  TOOL = 'tool',
  PROBLEM = 'problem'
}

export interface ExtractedEntity {
  id: string;
  type: EntityType;
  name: string;
  normalizedName: string;
  aliases: string[];
  confidence: number;          // 0-1
  importance: number;          // 1-10
  firstMention: Location;
  allMentions: Location[];
  definition?: string;
  category?: string;
  attributes?: Record<string, any>;
}

export interface Location {
  section: string;
  paragraph: number;
  sentenceIndex?: number;
  page?: number;
}

// ============================================================================
// Relationship Types
// ============================================================================

export enum RelationType {
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

  // Conceptual
  INTRODUCES = 'introduces',
  RELATED_TO = 'related_to',
  INSPIRED_BY = 'inspired_by',
  BUILDS_ON = 'builds_on',
  PART_OF = 'part_of',
  INSTANCE_OF = 'instance_of',

  // Critical
  CHALLENGES = 'challenges',
  CONTRADICTS = 'contradicts',
  IDENTIFIES_LIMITATION = 'identifies_limitation',
  REFUTES = 'refutes',

  // Citation
  CITES = 'cites',
  CITED_BY = 'cited_by',

  // Comparison
  COMPARES_WITH = 'compares_with',
  OUTPERFORMS = 'outperforms',
  EQUIVALENT_TO = 'equivalent_to'
}

export interface ExtractedRelationship {
  id: string;
  sourceEntityId: string;
  targetEntityId: string;
  relationType: RelationType;
  evidence: Evidence;
  confidence: number;
  qualifications?: string[];
  quantitative?: QuantitativeComparison;
  scope?: string;
}

export interface Evidence {
  quote: string;
  section: string;
  paragraph: number;
  sentenceIndex?: number;
}

export interface QuantitativeComparison {
  metric: string;
  sourceValue?: number;
  targetValue?: number;
  improvement: string;
  improvementPercent?: number;
}

// ============================================================================
// Paper Types
// ============================================================================

export interface PaperMetadata {
  arxivId: string;
  doi?: string;
  title: string;
  authors: Author[];
  abstract: string;
  publicationDate: Date;
  categories: string[];
  comments?: string;
  journal?: string;
  version: number;
  references: string[];
  pdfUrl: string;
  sourceRepo: 'arxiv' | 'openreview' | 'semantic_scholar';
}

export interface Author {
  name: string;
  affiliation?: string;
  email?: string;
  orcid?: string;
}

export interface StructuredPaper {
  metadata: PaperMetadata;
  sections: Section[];
  figures: Figure[];
  tables: Table[];
  equations: Equation[];
  references: Reference[];
  fullText: string;
  structure: DocumentStructure;
}

export interface Section {
  id: string;
  title: string;
  level: number;
  content: string;
  paragraphs: Paragraph[];
  startPage: number;
  endPage: number;
}

export interface Paragraph {
  id: string;
  content: string;
  citations: InlineCitation[];
  position: { page: number; bbox?: BoundingBox };
}

export interface InlineCitation {
  text: string;
  referenceId: string;
  position: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Figure {
  id: string;
  caption: string;
  page: number;
  imageData?: Buffer;
}

export interface Table {
  id: string;
  caption: string;
  rows: TableRow[];
  page: number;
}

export interface TableRow {
  cells: string[];
}

export interface Equation {
  id: string;
  latex: string;
  position: Location;
}

export interface Reference {
  id: string;
  rawText: string;
  parsed: ParsedReference;
}

export interface ParsedReference {
  authors?: string[];
  title?: string;
  venue?: string;
  year?: number;
  arxivId?: string;
  doi?: string;
}

export interface DocumentStructure {
  title: string;
  abstract: string;
  sections: SectionMetadata[];
  referencesStartPage: number;
}

export interface SectionMetadata {
  level: number;
  title: string;
  startPage: number;
  blocks: LayoutBlock[];
}

export interface LayoutBlock {
  type: 'title' | 'abstract' | 'section_header' | 'body_text' | 'caption' | 'reference' | 'equation';
  content: string;
  page: number;
  confidence: number;
}

// ============================================================================
// Linked Entity Types
// ============================================================================

export interface LinkedEntity {
  canonicalId: string;
  canonicalName: string;
  type: EntityType;
  aliases: string[];
  definitionPaper?: string;
  occurrences: EntityOccurrence[];
  embedding?: number[];
  definition?: string;
  confidence: number;
}

export interface EntityOccurrence {
  paperId: string;
  localName: string;
  usageContext: string;
  isDefinition: boolean;
  confidence: number;
}

export interface LinkingDecision {
  action: 'merge' | 'create_new' | 'needs_disambiguation';
  targetId?: string;
  confidence: number;
  reasoning?: string;
}

// ============================================================================
// Quality Control Types
// ============================================================================

export interface QualityReport {
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

export interface EntityValidation {
  totalEntities: number;
  passedValidation: number;
  failed: Array<{
    entityId: string;
    entityName: string;
    reason: string;
    severity: 'low' | 'medium' | 'high';
  }>;
}

export interface RelationshipValidation {
  totalRelationships: number;
  passedValidation: number;
  failed: Array<{
    relationshipId: string;
    reason: string;
    severity: 'low' | 'medium' | 'high';
  }>;
}

export interface CompletenessCheck {
  hasMainMethod: boolean;
  hasEvaluationMetrics: boolean;
  hasDatasets: boolean;
  hasBaselines: boolean;
  allSectionsProcessed: boolean;
  score: number;
}

export interface Issue {
  type: string;
  description: string;
  severity: 'low' | 'medium' | 'high';
  guideline?: string;
}

// ============================================================================
// Orchestration Types
// ============================================================================

export enum PipelineStage {
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

export interface OrchestrationContext {
  paperId: string;
  stage: PipelineStage;
  attemptCount: number;
  priority: number;
  dependencies: string[];
  metadata: {
    addedBy: 'seed' | 'citation' | 'manual' | 'realtime';
    corpusDepth: number;
    urgency: 'low' | 'medium' | 'high';
  };
}

export interface Checkpoint {
  paperId: string;
  stage: PipelineStage;
  data: any;
  timestamp: Date;
  attemptCount: number;
}

// ============================================================================
// Graph Database Types
// ============================================================================

export interface Node {
  id: string;
  type: string;
  canonical_name: string;
  properties: Record<string, any>;
  created_at: Date;
  updated_at: Date;
}

export interface Edge {
  id: string;
  source_id: string;
  target_id: string;
  type: string;
  properties: Record<string, any>;
  created_at: Date;
  updated_at: Date;
}

// ============================================================================
// Agent Configuration Types
// ============================================================================

export interface AgentConfig {
  name: string;
  model: 'haiku' | 'sonnet' | 'opus';
  maxTokens: number;
  temperature: number;
  systemPrompt: string;
  retryConfig: RetryConfig;
}

export interface RetryConfig {
  maxAttempts: number;
  backoffMs: number;
  backoffMultiplier: number;
  retryableErrors: ErrorType[];
}

export enum ErrorType {
  RATE_LIMIT = 'rate_limit',
  NETWORK_TIMEOUT = 'network_timeout',
  DATABASE_CONNECTION = 'database_connection',
  API_TIMEOUT = 'api_timeout',
  LOW_QUALITY_EXTRACTION = 'low_quality_extraction',
  AMBIGUOUS_ENTITY_LINKING = 'ambiguous_entity_linking',
  INCOMPLETE_PARSING = 'incomplete_parsing',
  VALIDATION_FAILURE = 'validation_failure',
  CORRUPTED_PDF = 'corrupted_pdf',
  NON_ENGLISH_PAPER = 'non_english_paper',
  PAPER_NOT_FOUND = 'paper_not_found',
  ACCESS_DENIED = 'access_denied'
}

// ============================================================================
// Extraction Result Types
// ============================================================================

export interface ExtractionResult {
  paper: StructuredPaper;
  entities: ExtractedEntity[];
  relationships: ExtractedRelationship[];
  qualityReport: QualityReport;
  sectionsProcessed: string[];
  processingMetadata: ProcessingMetadata;
}

export interface ProcessingMetadata {
  startTime: Date;
  endTime: Date;
  durationMs: number;
  tokensUsed: {
    haiku: number;
    sonnet: number;
    opus: number;
  };
  costUSD: number;
  agentVersions: Record<string, string>;
}

// ============================================================================
// All Entities Collection
// ============================================================================

export interface AllEntities {
  concepts: ExtractedEntity[];
  methods: ExtractedEntity[];
  metrics: ExtractedEntity[];
  datasets: ExtractedEntity[];
}
