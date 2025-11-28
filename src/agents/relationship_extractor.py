
import json
import logging
import re
from typing import List, Dict, Any, Optional
from uuid import uuid4

from anthropic import Anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from ..models import (
    ExtractedRelationship,
    ExtractedEntity,
    RelationType,
    StructuredPaper,
    Section,
    Evidence,
    QuantitativeComparison
)

logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """
    Agent for extracting semantic relationships between entities.

    This agent analyzes paper text to find relationships like:
    - Method A improves on Method B
    - Method X solves Problem Y
    - Paper evaluates on Dataset Z
    - Method uses Metric M for evaluation
    """

    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 8192
    TEMPERATURE = 0.1  # Very low temperature for consistency

    EXTRACTION_PROMPT = """You are analyzing semantic relationships in a research paper. Identify connections between methods, concepts, datasets, and metrics.

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
- uses_dataset: "We evaluate on X", "trained on dataset X"
- uses_metric: "We measure performance using metric X"
- evaluates_on: "We benchmark against X"
- challenges: "We demonstrate that X's assumption is incorrect"
- related_to: "Similar to X, we..."
- solves: "Our approach solves problem X"

For EACH relationship:
1. Identify source entity (often "our method", "this paper's approach")
2. Identify target entity (specific name from entity list above)
3. Classify relationship type
4. Extract evidence: exact quote supporting this relationship
5. Note any quantitative comparison (e.g., "15% higher PSNR", "2x faster")
6. Rate confidence (0-1)
7. Note qualifications/caveats (e.g., "for outdoor scenes only")

Output JSON array:
[
  {{
    "source_entity": "3D Gaussian Splatting",
    "target_entity": "Neural Radiance Fields",
    "relation_type": "improves_on",
    "evidence": {{
      "quote": "Our method achieves real-time rendering (≥30 fps) while maintaining competitive quality, compared to NeRF's 30-second per-frame rendering time.",
      "section": "Abstract",
      "paragraph": 1
    }},
    "quantitative": {{
      "metric": "rendering_speed",
      "source_value": 30.0,
      "target_value": 0.033,
      "improvement": "900x faster"
    }},
    "confidence": 0.96
  }}
]

Guidelines:
- Be precise: "improves" requires comparative claim, not just "we use X"
- Ground in text: every relationship must have exact quote
- Identify scope: if claim applies only to specific settings, note it
- Quantitative preferred: extract numeric comparisons when available
- Confidence: 0.9+ for explicit statements, 0.6-0.8 for implicit, <0.6 for inferred

Few-shot examples:

Example 1:
Text: "Unlike NeRF which requires ray marching through a volume, our method uses explicit 3D Gaussians for direct rendering, achieving 900× speedup."
Output: {{
  "source_entity": "3D Gaussian Splatting",
  "target_entity": "NeRF",
  "relation_type": "improves_on",
  "evidence": {{
    "quote": "Unlike NeRF which requires ray marching through a volume, our method uses explicit 3D Gaussians for direct rendering, achieving 900× speedup.",
    "section": "Introduction",
    "paragraph": 3
  }},
  "quantitative": {{
    "metric": "rendering_speed",
    "improvement": "900x faster"
  }},
  "confidence": 0.95
}}

Example 2:
Text: "We evaluate on the Mip-NeRF 360 dataset, achieving an average PSNR of 28.5 dB."
Output: {{
  "source_entity": "3D Gaussian Splatting",
  "target_entity": "Mip-NeRF 360",
  "relation_type": "evaluates_on",
  "evidence": {{
    "quote": "We evaluate on the Mip-NeRF 360 dataset",
    "section": "Experiments",
    "paragraph": 2
  }},
  "confidence": 0.99
}}

Example 3:
Text: "We report PSNR, SSIM, and LPIPS for quantitative evaluation."
Output: {{
  "source_entity": "3D Gaussian Splatting",
  "target_entity": "PSNR",
  "relation_type": "uses_metric",
  "evidence": {{
    "quote": "We report PSNR, SSIM, and LPIPS for quantitative evaluation",
    "section": "Experiments",
    "paragraph": 1
  }},
  "confidence": 0.98
}}

Now extract relationships from the provided section. Return ONLY a JSON array.
"""

    def __init__(self, api_key: str):
        """
        Initialize the Relationship Extractor.

        Args:
            api_key: Anthropic API key
        """
        self.client = Anthropic(api_key=api_key)

    async def extract(
        self,
        paper: StructuredPaper,
        entities: Dict[str, List[ExtractedEntity]]
    ) -> List[ExtractedRelationship]:
        """
        Extract relationships from a paper given extracted entities.

        Args:
            paper: Structured paper to extract from
            entities: Previously extracted entities (concepts, methods, etc.)

        Returns:
            List of extracted relationships

        Raises:
            Exception: If extraction fails after retries
        """
        logger.info(
            f"Starting relationship extraction for paper {paper.metadata.arxiv_id}"
        )

        relationships: List[ExtractedRelationship] = []

        # Focus on sections with comparative claims and evaluations
        target_sections = [
            s for s in paper.sections
            if re.search(
                r"abstract|introduction|method|results|experiments|evaluation|discussion",
                s.title,
                re.IGNORECASE
            )
        ]

        if not target_sections:
            logger.warning(f"No target sections found, using all sections")
            target_sections = paper.sections

        # Build entity name lists for prompt
        entity_context = {
            "concepts": [e.name for e in entities.get("concepts", [])],
            "methods": [e.name for e in entities.get("methods", [])],
            "datasets": [e.name for e in entities.get("datasets", [])],
            "metrics": [e.name for e in entities.get("metrics", [])]
        }

        # Extract from each section
        for section in target_sections:
            try:
                section_rels = await self._extract_from_section(
                    section,
                    entity_context,
                    entities
                )
                relationships.extend(section_rels)
                logger.debug(
                    f"Extracted {len(section_rels)} relationships from "
                    f"section '{section.title}'"
                )
            except Exception as e:
                logger.error(
                    f"Failed to extract from section '{section.title}': {e}",
                    exc_info=True
                )

        # Deduplicate relationships
        deduplicated = self._deduplicate_relationships(relationships)

        logger.info(
            f"Relationship extraction completed for {paper.metadata.arxiv_id}: "
            f"{len(relationships)} raw -> {len(deduplicated)} after deduplication"
        )

        return deduplicated

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    async def _extract_from_section(
        self,
        section: Section,
        entity_context: Dict[str, List[str]],
        all_entities: Dict[str, List[ExtractedEntity]]
    ) -> List[ExtractedRelationship]:
        """Extract relationships from a single section."""

        prompt = self._build_prompt(section.content, entity_context)

        # Call Claude API
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            temperature=self.TEMPERATURE,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Parse response
        text_content = response.content[0]
        if text_content.type != "text":
            raise ValueError("Unexpected response type from Claude")

        json_text = self._extract_json(text_content.text)

        try:
            extracted = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON response: {e}\n"
                f"Response preview: {text_content.text[:500]}"
            )
            return []

        # Convert to ExtractedRelationship objects
        return [
            self._to_extracted_relationship(rel, section, all_entities)
            for rel in extracted
            if self._validate_relationship(rel, all_entities)
        ]

    def _build_prompt(
        self,
        section_content: str,
        entity_context: Dict[str, List[str]]
    ) -> str:
        """Build the extraction prompt."""
        max_section_length = 4000

        if len(section_content) > max_section_length:
            section_content = section_content[:max_section_length] + "..."

        # Format entity lists
        def format_list(items: List[str]) -> str:
            if not items:
                return "None"
            return ", ".join(items[:15])  # Limit to first 15

        return self.EXTRACTION_PROMPT.format(
            section_content=section_content,
            concepts=format_list(entity_context.get("concepts", [])),
            methods=format_list(entity_context.get("methods", [])),
            datasets=format_list(entity_context.get("datasets", [])),
            metrics=format_list(entity_context.get("metrics", []))
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from potentially markdown-wrapped text."""
        json_match = re.search(r'```json\n([\s\S]*?)\n```', text)
        if json_match:
            return json_match.group(1)

        code_match = re.search(r'```\n([\s\S]*?)\n```', text)
        if code_match:
            return code_match.group(1)

        return text.strip()

    def _validate_relationship(
        self,
        rel: Dict[str, Any],
        all_entities: Dict[str, List[ExtractedEntity]]
    ) -> bool:
        """Validate that relationship entities exist."""
        # Check required fields
        required = ["source_entity", "target_entity", "relation_type", "evidence"]
        if not all(field in rel for field in required):
            logger.warning(f"Relationship missing required fields: {rel}")
            return False

        # Check that entities exist
        all_entity_names = []
        for entities in all_entities.values():
            all_entity_names.extend([e.name for e in entities])
            all_entity_names.extend([
                alias for e in entities for alias in e.aliases
            ])

        source_exists = rel["source_entity"] in all_entity_names
        target_exists = rel["target_entity"] in all_entity_names

        if not (source_exists and target_exists):
            logger.debug(
                f"Relationship references unknown entities: "
                f"{rel['source_entity']} -> {rel['target_entity']}"
            )
            # Allow "this paper" or "our method" as source
            if rel["source_entity"].lower() in ["this paper", "our method", "we"]:
                source_exists = True

        return source_exists and target_exists

    def _to_extracted_relationship(
        self,
        rel: Dict[str, Any],
        section: Section,
        all_entities: Dict[str, List[ExtractedEntity]]
    ) -> ExtractedRelationship:
        """Convert API output to ExtractedRelationship model."""

        # Resolve entity IDs
        source_id = self._resolve_entity_id(
            rel["source_entity"],
            all_entities
        )
        target_id = self._resolve_entity_id(
            rel["target_entity"],
            all_entities
        )

        # Parse evidence
        evidence_dict = rel["evidence"]
        evidence = Evidence(
            quote=evidence_dict["quote"],
            section=evidence_dict.get("section", section.title),
            paragraph=evidence_dict.get("paragraph", 0)
        )

        # Parse quantitative data if present
        quantitative = None
        if "quantitative" in rel:
            quant = rel["quantitative"]
            quantitative = QuantitativeComparison(
                metric=quant["metric"],
                source_value=quant.get("source_value"),
                target_value=quant.get("target_value"),
                improvement=quant["improvement"],
                improvement_percent=quant.get("improvement_percent")
            )

        return ExtractedRelationship(
            id=str(uuid4()),
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type=RelationType(rel["relation_type"]),
            evidence=evidence,
            confidence=float(rel.get("confidence", 0.8)),
            qualifications=rel.get("qualifications", []),
            quantitative=quantitative,
            scope=rel.get("scope")
        )

    def _resolve_entity_id(
        self,
        entity_name: str,
        all_entities: Dict[str, List[ExtractedEntity]]
    ) -> str:
        """Resolve entity name to ID."""
        # Search through all entities
        for entities in all_entities.values():
            for entity in entities:
                if (entity.name.lower() == entity_name.lower() or
                    entity.normalized_name.lower() == entity_name.lower() or
                    any(alias.lower() == entity_name.lower()
                        for alias in entity.aliases)):
                    return entity.id

        # Not found - return placeholder
        logger.warning(f"Could not resolve entity '{entity_name}' to ID")
        return f"unresolved:{entity_name}"

    def _deduplicate_relationships(
        self,
        relationships: List[ExtractedRelationship]
    ) -> List[ExtractedRelationship]:
        """Deduplicate relationships keeping highest confidence."""
        # Group by source, target, type
        grouped: Dict[str, List[ExtractedRelationship]] = {}

        for rel in relationships:
            key = f"{rel.source_entity_id}:{rel.target_entity_id}:{rel.relation_type.value}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(rel)

        # Keep best from each group
        deduplicated: List[ExtractedRelationship] = []

        for key, group in grouped.items():
            if len(group) == 1:
                deduplicated.append(group[0])
                continue

            # Take highest confidence
            best = max(group, key=lambda r: r.confidence)

            # Could optionally merge evidence from all
            logger.debug(
                f"Merged {len(group)} duplicate relationships: "
                f"{best.relation_type.value}"
            )

            deduplicated.append(best)

        return deduplicated

    def validate_relationship(
        self,
        rel: ExtractedRelationship,
        paper: StructuredPaper
    ) -> Dict[str, Any]:
        """
        Validate a relationship against the paper text.

        Args:
            rel: Relationship to validate
            paper: Full paper for validation

        Returns:
            Dictionary with 'valid' (bool) and 'issues' (list) keys
        """
        issues = []

        # Check if quote exists in paper
        normalized_quote = self._normalize_text(rel.evidence.quote)
        normalized_paper = self._normalize_text(paper.full_text)

        if normalized_quote not in normalized_paper:
            issues.append("Evidence quote not found in paper text")

        # Check confidence threshold
        if rel.confidence < 0.5:
            issues.append("Confidence below acceptable threshold (0.5)")

        # Check entity IDs are not unresolved
        if rel.source_entity_id.startswith("unresolved:"):
            issues.append(f"Source entity unresolved: {rel.source_entity_id}")

        if rel.target_entity_id.startswith("unresolved:"):
            issues.append(f"Target entity unresolved: {rel.target_entity_id}")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return re.sub(r'\s+', ' ', text.lower()).strip()


def create_relationship_extractor(api_key: str) -> RelationshipExtractor:
    """
    Factory function to create a RelationshipExtractor.

    Args:
        api_key: Anthropic API key

    Returns:
        Configured RelationshipExtractor instance
    """
    return RelationshipExtractor(api_key)
"""
Relationship Extractor Agent

Specialized LLM agent for identifying semantic relationships between entities
in academic papers. Goes beyond simple citations to extract deep semantic
connections like "improves_on", "addresses_limitation", "extends", etc.

This agent:
- Identifies explicit relationships stated in text
- Infers implicit relationships from context
- Distinguishes relationship types with high precision
- Grounds every relationship in textual evidence
- Extracts quantitative comparisons when available
"""
