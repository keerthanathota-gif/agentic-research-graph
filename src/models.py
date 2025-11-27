"""
Core data models for the Research Knowledge Graph system.

This module defines all Pydantic models used throughout the system for
type safety, validation, and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Enums
# ============================================================================

class EntityType(str, Enum):
    """Types of entities that can be extracted from papers."""
    CONCEPT = "concept"
    METHOD = "method"
    METRIC = "metric"
    DATASET = "dataset"
    TOOL = "tool"
    PROBLEM = "problem"


class RelationType(str, Enum):
    """Types of relationships between entities."""
    # Improvement relationships
    IMPROVES_ON = "improves_on"
    EXTENDS = "extends"
    SIMPLIFIES = "simplifies"
    GENERALIZES = "generalizes"
    OPTIMIZES = "optimizes"

    # Problem-solution
    SOLVES = "solves"
    ADDRESSES_LIMITATION = "addresses_limitation"
    ENABLES = "enables"

    # Usage relationships
    USES_METHOD = "uses_method"
    USES_DATASET = "uses_dataset"
    USES_METRIC = "uses_metric"
    EVALUATES_ON = "evaluates_on"
    TRAINS_ON = "trains_on"

    # Conceptual
    INTRODUCES = "introduces"
    RELATED_TO = "related_to"
    INSPIRED_BY = "inspired_by"
    BUILDS_ON = "builds_on"
    PART_OF = "part_of"
    INSTANCE_OF = "instance_of"

    # Critical
    CHALLENGES = "challenges"
    CONTRADICTS = "contradicts"
    IDENTIFIES_LIMITATION = "identifies_limitation"
    REFUTES = "refutes"

    # Citation
    CITES = "cites"
    CITED_BY = "cited_by"

    # Comparison
    COMPARES_WITH = "compares_with"
    OUTPERFORMS = "outperforms"
    EQUIVALENT_TO = "equivalent_to"


class PipelineStage(str, Enum):
    """Stages in the paper processing pipeline."""
    QUEUED = "queued"
    FETCHING = "fetching"
    PARSING = "parsing"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATIONSHIP_EXTRACTION = "relationship_extraction"
    ENTITY_LINKING = "entity_linking"
    QUALITY_CONTROL = "quality_control"
    GRAPH_BUILDING = "graph_building"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY_SCHEDULED = "retry_scheduled"


# ============================================================================
# Location & Evidence Models
# ============================================================================

class Location(BaseModel):
    """Location within a paper."""
    section: str
    paragraph: int
    sentence_index: Optional[int] = None
    page: Optional[int] = None


class BoundingBox(BaseModel):
    """Bounding box coordinates for layout elements."""
    x: float
    y: float
    width: float
    height: float


class Evidence(BaseModel):
    """Evidence supporting a relationship."""
    quote: str
    section: str
    paragraph: int
    sentence_index: Optional[int] = None


class QuantitativeComparison(BaseModel):
    """Quantitative comparison data."""
    metric: str
    source_value: Optional[float] = None
    target_value: Optional[float] = None
    improvement: str
    improvement_percent: Optional[float] = None


# ============================================================================
# Author & Paper Metadata
# ============================================================================

class Author(BaseModel):
    """Author information."""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None


class PaperMetadata(BaseModel):
    """Metadata for an academic paper."""
    arxiv_id: str
    doi: Optional[str] = None
    title: str
    authors: List[Author]
    abstract: str
    publication_date: datetime
    categories: List[str]
    comments: Optional[str] = None
    journal: Optional[str] = None
    version: int = 1
    references: List[str] = Field(default_factory=list)
    pdf_url: str
    source_repo: str = "arxiv"


# ============================================================================
# Paper Structure Models
# ============================================================================

class InlineCitation(BaseModel):
    """Inline citation within text."""
    text: str
    reference_id: str
    position: int


class Paragraph(BaseModel):
    """A paragraph within a section."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    citations: List[InlineCitation] = Field(default_factory=list)
    position: Dict[str, Any] = Field(default_factory=dict)


class Section(BaseModel):
    """A section of a paper."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    level: int
    content: str
    paragraphs: List[Paragraph] = Field(default_factory=list)
    start_page: int
    end_page: int


class Figure(BaseModel):
    """A figure in the paper."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    caption: str
    page: int
    image_data: Optional[bytes] = None


class TableRow(BaseModel):
    """A row in a table."""
    cells: List[str]


class Table(BaseModel):
    """A table in the paper."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    caption: str
    rows: List[TableRow]
    page: int


class Equation(BaseModel):
    """A mathematical equation."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    latex: str
    position: Location


class ParsedReference(BaseModel):
    """Parsed reference information."""
    authors: Optional[List[str]] = None
    title: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None


class Reference(BaseModel):
    """A reference/citation."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    raw_text: str
    parsed: ParsedReference


class DocumentStructure(BaseModel):
    """Overall structure of a document."""
    title: str
    abstract: str
    sections: List[Dict[str, Any]]
    references_start_page: int


class StructuredPaper(BaseModel):
    """Complete structured representation of a paper."""
    metadata: PaperMetadata
    sections: List[Section]
    figures: List[Figure] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    equations: List[Equation] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)
    full_text: str
    structure: DocumentStructure


# ============================================================================
# Entity Models
# ============================================================================

class ExtractedEntity(BaseModel):
    """An entity extracted from a paper."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: EntityType
    name: str
    normalized_name: str
    aliases: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    importance: int = Field(ge=1, le=10)
    first_mention: Location
    all_mentions: List[Location] = Field(default_factory=list)
    definition: Optional[str] = None
    category: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        return max(0.0, min(1.0, v))


class EntityOccurrence(BaseModel):
    """An occurrence of an entity in a paper."""
    paper_id: str
    local_name: str
    usage_context: str
    is_definition: bool = False
    confidence: float


class LinkedEntity(BaseModel):
    """An entity linked across multiple papers."""
    canonical_id: str = Field(default_factory=lambda: str(uuid4()))
    canonical_name: str
    type: EntityType
    aliases: List[str] = Field(default_factory=list)
    definition_paper: Optional[str] = None
    occurrences: List[EntityOccurrence] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    definition: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


# ============================================================================
# Relationship Models
# ============================================================================

class ExtractedRelationship(BaseModel):
    """A relationship extracted from a paper."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    evidence: Evidence
    confidence: float = Field(ge=0.0, le=1.0)
    qualifications: List[str] = Field(default_factory=list)
    quantitative: Optional[QuantitativeComparison] = None
    scope: Optional[str] = None


# ============================================================================
# Quality Control Models
# ============================================================================

class Issue(BaseModel):
    """An issue found during quality control."""
    type: str
    description: str
    severity: str = Field(pattern="^(low|medium|high)$")
    guideline: Optional[str] = None


class EntityValidation(BaseModel):
    """Validation results for entities."""
    total_entities: int
    passed_validation: int
    failed: List[Dict[str, Any]] = Field(default_factory=list)


class RelationshipValidation(BaseModel):
    """Validation results for relationships."""
    total_relationships: int
    passed_validation: int
    failed: List[Dict[str, Any]] = Field(default_factory=list)


class CompletenessCheck(BaseModel):
    """Completeness assessment of extraction."""
    has_main_method: bool
    has_evaluation_metrics: bool
    has_datasets: bool
    has_baselines: bool
    all_sections_processed: bool
    score: float = Field(ge=0.0, le=1.0)


class QualityReport(BaseModel):
    """Quality control report for a paper extraction."""
    paper_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    validations: Dict[str, Any]
    overall_score: float = Field(ge=0.0, le=1.0)
    issues: List[Issue] = Field(default_factory=list)
    recommendation: str = Field(pattern="^(accept|review|retry|reject)$")


# ============================================================================
# Orchestration Models
# ============================================================================

class OrchestrationContext(BaseModel):
    """Context for orchestrating paper processing."""
    paper_id: str
    stage: PipelineStage
    attempt_count: int = 0
    priority: int = 50
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Checkpoint(BaseModel):
    """Checkpoint for resuming processing."""
    paper_id: str
    stage: PipelineStage
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    attempt_count: int


# ============================================================================
# Extraction Results
# ============================================================================

class ProcessingMetadata(BaseModel):
    """Metadata about processing a paper."""
    start_time: datetime
    end_time: datetime
    duration_ms: int
    tokens_used: Dict[str, int] = Field(default_factory=dict)
    cost_usd: float
    agent_versions: Dict[str, str] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Complete results from extracting a paper."""
    paper: StructuredPaper
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    quality_report: QualityReport
    sections_processed: List[str]
    processing_metadata: ProcessingMetadata


# ============================================================================
# Entity Linking Models
# ============================================================================

class LinkingDecision(BaseModel):
    """Decision about linking an entity."""
    action: str = Field(pattern="^(merge|create_new|needs_disambiguation)$")
    target_id: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None


# ============================================================================
# Configuration Models
# ============================================================================

class AgentConfig(BaseModel):
    """Configuration for an LLM agent."""
    name: str
    model: str = Field(pattern="^(haiku|sonnet|opus)$")
    max_tokens: int = 4096
    temperature: float = Field(ge=0.0, le=2.0)
    system_prompt: str = ""
    retry_max_attempts: int = 3
    retry_backoff_ms: int = 1000


class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "research_kg"
    user: str = "postgres"
    password: str
    pool_size: int = 10
    max_overflow: int = 20


# ============================================================================
# Graph Database Models
# ============================================================================

class Node(BaseModel):
    """A node in the graph database."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    canonical_name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Edge(BaseModel):
    """An edge in the graph database."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
