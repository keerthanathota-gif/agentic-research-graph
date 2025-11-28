
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
    ExtractedEntity,
    EntityType,
    StructuredPaper,
    Section,
    Location
)

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """
    Agent for extracting research concepts from academic papers.

    Attributes:
        client: Anthropic API client
        model: Claude model to use (sonnet for complex reasoning)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (low for consistency)
    """

    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 8192
    TEMPERATURE = 0.2  # Low temperature for consistency

    # Prompt template with few-shot examples
    EXTRACTION_PROMPT = """You are a research concept extraction specialist. Analyze this paper section and extract key conceptual terms.

Guidelines:
1. Extract concepts at multiple granularities:
   - Broad: "computer vision", "3D reconstruction"
   - Specific: "neural scene representation", "volumetric ray marching"
2. Focus on concepts central to the paper's contribution
3. Exclude generic terms like "experiment", "result", "paper", "approach", "method" (unless part of a specific named concept)
4. For each concept, provide:
   - Normalized form (e.g., "NeRF" → "Neural Radiance Fields")
   - Aliases (common variations and abbreviations)
   - First mention location (section, paragraph)
   - Importance score (1-10, where 10 is central to paper's contribution)
   - Confidence score (0-1, based on clarity of definition/usage)
   - Definition (if explicitly defined in the text)
   - Category: technique, problem, application, theory

Paper abstract (for context):
{abstract}

Section content:
{section_content}

Output format: JSON array of concepts

Few-shot examples:

Example 1:
Text: "Neural Radiance Fields (NeRF) represent scenes as continuous volumetric functions encoded in neural network weights..."
Output: {{
  "name": "Neural Radiance Fields",
  "normalized_name": "Neural Radiance Fields",
  "aliases": ["NeRF", "NeRFs", "neural radiance field"],
  "importance": 10,
  "confidence": 0.98,
  "first_mention": {{"section": "Introduction", "paragraph": 2}},
  "definition": "Continuous volumetric scene representations encoded in neural network weights",
  "category": "technique"
}}

Example 2:
Text: "We address the view synthesis problem, generating novel viewpoints from a set of input images..."
Output: {{
  "name": "view synthesis",
  "normalized_name": "View Synthesis",
  "aliases": ["novel view synthesis", "image synthesis"],
  "importance": 8,
  "confidence": 0.92,
  "first_mention": {{"section": "Introduction", "paragraph": 1}},
  "definition": "Generating new camera viewpoints from input images",
  "category": "problem"
}}

Example 3:
Text: "Real-time rendering remains a challenge for neural scene representations..."
Output: {{
  "name": "real-time rendering",
  "normalized_name": "Real-Time Rendering",
  "aliases": ["realtime rendering", "interactive rendering"],
  "importance": 9,
  "confidence": 0.95,
  "first_mention": {{"section": "Introduction", "paragraph": 3}},
  "category": "problem"
}}

Now extract concepts from the provided section. Return ONLY a JSON array, no additional text.
"""

    def __init__(self, api_key: str):
        """
        Initialize the Concept Extractor.

        Args:
            api_key: Anthropic API key
        """
        self.client = Anthropic(api_key=api_key)

    async def extract(self, paper: StructuredPaper) -> List[ExtractedEntity]:
        """
        Extract concepts from a structured paper.

        Args:
            paper: Structured paper to extract from

        Returns:
            List of extracted concept entities

        Raises:
            Exception: If extraction fails after retries
        """
        logger.info(f"Starting concept extraction for paper {paper.metadata.arxiv_id}")

        concepts: List[ExtractedEntity] = []

        # Target sections most likely to contain conceptual information
        target_sections = [
            s for s in paper.sections
            if re.search(r"abstract|introduction|related work|background|method",
                        s.title, re.IGNORECASE)
        ]

        if not target_sections:
            logger.warning(
                f"No target sections found for {paper.metadata.arxiv_id}, "
                f"using first 3 sections as fallback"
            )
            target_sections = paper.sections[:3]

        # Extract from each section
        for section in target_sections:
            try:
                section_concepts = await self._extract_from_section(
                    section,
                    paper.metadata.abstract
                )
                concepts.extend(section_concepts)
                logger.debug(
                    f"Extracted {len(section_concepts)} concepts from "
                    f"section '{section.title}'"
                )
            except Exception as e:
                logger.error(
                    f"Failed to extract from section '{section.title}': {e}",
                    exc_info=True
                )
                # Continue with other sections

        # Deduplicate concepts
        deduplicated = self._deduplicate_concepts(concepts)

        logger.info(
            f"Concept extraction completed for {paper.metadata.arxiv_id}: "
            f"{len(concepts)} raw -> {len(deduplicated)} after deduplication"
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
        abstract: str
    ) -> List[ExtractedEntity]:
        """
        Extract concepts from a single section with retry logic.

        Args:
            section: Section to extract from
            abstract: Paper abstract for context

        Returns:
            List of extracted entities
        """
        prompt = self._build_prompt(section.content, abstract)

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

        # Extract JSON (may be wrapped in markdown code blocks)
        json_text = self._extract_json(text_content.text)

        try:
            extracted = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON response: {e}\n"
                f"Response preview: {text_content.text[:500]}"
            )
            return []

        # Convert to ExtractedEntity objects
        return [
            self._to_extracted_entity(concept, section)
            for concept in extracted
        ]

    def _build_prompt(self, section_content: str, abstract: str) -> str:
        """Build the extraction prompt."""
        # Truncate if too long
        max_section_length = 4000
        max_abstract_length = 500

        if len(section_content) > max_section_length:
            section_content = section_content[:max_section_length] + "..."

        if len(abstract) > max_abstract_length:
            abstract = abstract[:max_abstract_length] + "..."

        return self.EXTRACTION_PROMPT.format(
            abstract=abstract,
            section_content=section_content
        )

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text that may be wrapped in markdown code blocks.

        Args:
            text: Text potentially containing JSON

        Returns:
            Extracted JSON string
        """
        # Try to find JSON in markdown code block
        json_match = re.search(r'```json\n([\s\S]*?)\n```', text)
        if json_match:
            return json_match.group(1)

        # Try to find plain code block
        code_match = re.search(r'```\n([\s\S]*?)\n```', text)
        if code_match:
            return code_match.group(1)

        # Assume entire text is JSON
        return text.strip()

    def _to_extracted_entity(
        self,
        concept: Dict[str, Any],
        section: Section
    ) -> ExtractedEntity:
        """
        Convert API output to ExtractedEntity model.

        Args:
            concept: Raw concept dict from API
            section: Section it was extracted from

        Returns:
            ExtractedEntity instance
        """
        first_mention = concept.get("first_mention", {})

        return ExtractedEntity(
            id=str(uuid4()),
            type=EntityType.CONCEPT,
            name=concept["name"],
            normalized_name=concept["normalized_name"],
            aliases=concept.get("aliases", []),
            confidence=float(concept["confidence"]),
            importance=int(concept["importance"]),
            first_mention=Location(
                section=first_mention.get("section", section.title),
                paragraph=first_mention.get("paragraph", 0)
            ),
            all_mentions=[Location(
                section=first_mention.get("section", section.title),
                paragraph=first_mention.get("paragraph", 0)
            )],
            definition=concept.get("definition"),
            category=concept.get("category", "technique")
        )

    def _deduplicate_concepts(
        self,
        concepts: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """
        Deduplicate concepts by normalizing names and merging similar ones.

        Args:
            concepts: List of extracted concepts

        Returns:
            Deduplicated list
        """
        # Group by normalized name (case-insensitive)
        grouped: Dict[str, List[ExtractedEntity]] = {}

        for concept in concepts:
            key = concept.normalized_name.lower()
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(concept)

        # Merge duplicates
        deduplicated: List[ExtractedEntity] = []

        for key, group in grouped.items():
            if len(group) == 1:
                deduplicated.append(group[0])
                continue

            # Merge: take highest confidence, combine aliases and mentions
            best = max(group, key=lambda c: c.confidence)

            # Combine all aliases
            all_aliases = set()
            for concept in group:
                all_aliases.add(concept.name)
                all_aliases.update(concept.aliases)

            # Remove canonical name from aliases
            all_aliases.discard(best.normalized_name)
            best.aliases = sorted(all_aliases)

            # Combine all mentions
            best.all_mentions = [
                mention
                for concept in group
                for mention in concept.all_mentions
            ]

            # Take first available definition
            if not best.definition:
                for concept in group:
                    if concept.definition:
                        best.definition = concept.definition
                        break

            deduplicated.append(best)

            logger.debug(
                f"Merged {len(group)} duplicate concepts into "
                f"'{best.normalized_name}' with {len(best.aliases)} aliases"
            )

        return deduplicated

    def validate_concept(
        self,
        concept: ExtractedEntity,
        paper_text: str
    ) -> Dict[str, Any]:
        """
        Validate an extracted concept against the paper text.

        Args:
            concept: Concept to validate
            paper_text: Full text of the paper

        Returns:
            Dictionary with 'valid' (bool) and 'issues' (list) keys
        """
        issues = []
        paper_lower = paper_text.lower()

        # Check if concept appears in paper
        variants = [concept.name, concept.normalized_name] + concept.aliases
        mentioned = any(
            variant.lower() in paper_lower
            for variant in variants
        )

        if not mentioned:
            issues.append("Concept not found in paper text")

        # Check confidence threshold
        if concept.confidence < 0.5:
            issues.append("Confidence below acceptable threshold (0.5)")

        # Check for potential hallucination
        if len(concept.all_mentions) == 1 and concept.importance > 7:
            issues.append(
                "High importance (>7) but mentioned only once "
                "(possible hallucination)"
            )

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }


def create_concept_extractor(api_key: str) -> ConceptExtractor:
    """
    Factory function to create a ConceptExtractor.

    Args:
        api_key: Anthropic API key

    Returns:
        Configured ConceptExtractor instance
    """
    return ConceptExtractor(api_key)


"""
Concept Extractor Agent

Specialized LLM agent for extracting research concepts from academic papers.
Uses Claude Sonnet 4.5 for complex reasoning about conceptual relationships.

This agent:
- Identifies high-level research concepts at multiple granularities
- Normalizes concept names and tracks aliases
- Assesses the importance and confidence for each extraction
- Validates extractions against paper text
"""

