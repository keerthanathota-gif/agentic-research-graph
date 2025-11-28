
import Anthropic from '@anthropic-ai/sdk';
import { v4 as uuid } from 'uuid';
import {
  ExtractedEntity,
  EntityType,
  StructuredPaper,
  Section,
  Location
} from '../types';
import { logger } from '../utils/logger';
import { withRetry } from '../utils/retry';
import { ErrorType } from '../types';

interface ConceptExtractionOutput {
  name: string;
  normalized_name: string;
  aliases: string[];
  importance: number;
  confidence: number;
  first_mention: {
    section: string;
    paragraph: number;
  };
  definition?: string;
  category?: 'technique' | 'problem' | 'metric' | 'dataset' | 'application' | 'theory';
}

export class ConceptExtractor {
  private client: Anthropic;
  private readonly model = 'claude-sonnet-4-20250514';
  private readonly maxTokens = 8192;
  private readonly temperature = 0.2; // Low temperature for consistency

  constructor(apiKey: string) {
    this.client = new Anthropic({ apiKey });
  }

  /**
   * Extract concepts from a structured paper
   */
  async extract(paper: StructuredPaper): Promise<ExtractedEntity[]> {
    const span = logger.startSpan('concept_extractor.extract', {
      paperId: paper.metadata.arxivId
    });

    try {
      const concepts: ExtractedEntity[] = [];

      // Focus on sections most likely to contain conceptual information
      const targetSections = paper.sections.filter(s =>
        /abstract|introduction|related work|background|method/i.test(s.title)
      );

      if (targetSections.length === 0) {
        logger.warn('No target sections found for concept extraction', {
          paperId: paper.metadata.arxivId,
          availableSections: paper.sections.map(s => s.title)
        });
        // Fallback: use first 3 sections
        targetSections.push(...paper.sections.slice(0, 3));
      }

      // Extract from each section
      for (const section of targetSections) {
        try {
          const sectionConcepts = await this.extractFromSection(
            section,
            paper.metadata.abstract
          );
          concepts.push(...sectionConcepts);
        } catch (error) {
          logger.error('Failed to extract concepts from section', {
            paperId: paper.metadata.arxivId,
            section: section.title,
            error: error.message
          });
          // Continue with other sections
        }
      }

      // Deduplicate concepts
      const deduplicated = this.deduplicateConcepts(concepts);

      logger.info('Concept extraction completed', {
        paperId: paper.metadata.arxivId,
        sectionsProcessed: targetSections.length,
        conceptsExtracted: concepts.length,
        afterDeduplication: deduplicated.length
      });

      span.setAttribute('concepts_extracted', deduplicated.length);
      return deduplicated;

    } catch (error) {
      logger.error('Concept extraction failed', {
        paperId: paper.metadata.arxivId,
        error: error.message
      });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Extract concepts from a single section
   */
  private async extractFromSection(
    section: Section,
    abstract: string
  ): Promise<ExtractedEntity[]> {

    const prompt = this.buildPrompt(section.content, abstract);

    const response = await withRetry(
      async () => {
        return await this.client.messages.create({
          model: this.model,
          max_tokens: this.maxTokens,
          temperature: this.temperature,
          messages: [{
            role: 'user',
            content: prompt
          }]
        });
      },
      ErrorType.API_TIMEOUT,
      { section: section.title }
    );

    // Parse response
    const textContent = response.content[0];
    if (textContent.type !== 'text') {
      throw new Error('Unexpected response type from Claude');
    }

    let extracted: ConceptExtractionOutput[];
    try {
      // Extract JSON from response (may be wrapped in markdown code blocks)
      const jsonMatch = textContent.text.match(/```json\n([\s\S]*?)\n```/);
      const jsonText = jsonMatch ? jsonMatch[1] : textContent.text;
      extracted = JSON.parse(jsonText);
    } catch (error) {
      logger.error('Failed to parse concept extraction response', {
        section: section.title,
        response: textContent.text.substring(0, 500),
        error: error.message
      });
      return []; // Return empty array rather than failing entire extraction
    }

    // Convert to ExtractedEntity format
    return extracted.map(concept => this.toExtractedEntity(concept, section));
  }

  /**
   * Build the extraction prompt with few-shot examples
   */
  private buildPrompt(sectionContent: string, abstract: string): string {
    return `You are a research concept extraction specialist. Analyze this paper section and extract key conceptual terms.

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
${abstract.substring(0, 500)}...

Section content:
${sectionContent}

Output format: JSON array of concepts

Few-shot examples:

Example 1:
Text: "Neural Radiance Fields (NeRF) represent scenes as continuous volumetric functions encoded in neural network weights..."
Output: {
  "name": "Neural Radiance Fields",
  "normalized_name": "Neural Radiance Fields",
  "aliases": ["NeRF", "NeRFs", "neural radiance field"],
  "importance": 10,
  "confidence": 0.98,
  "first_mention": {"section": "Introduction", "paragraph": 2},
  "definition": "Continuous volumetric scene representations encoded in neural network weights",
  "category": "technique"
}

Example 2:
Text: "We address the view synthesis problem, generating novel viewpoints from a set of input images..."
Output: {
  "name": "view synthesis",
  "normalized_name": "View Synthesis",
  "aliases": ["novel view synthesis", "image synthesis"],
  "importance": 8,
  "confidence": 0.92,
  "first_mention": {"section": "Introduction", "paragraph": 1},
  "definition": "Generating new camera viewpoints from input images",
  "category": "problem"
}

Example 3:
Text: "Real-time rendering remains a challenge for neural scene representations..."
Output: {
  "name": "real-time rendering",
  "normalized_name": "Real-Time Rendering",
  "aliases": ["realtime rendering", "interactive rendering"],
  "importance": 9,
  "confidence": 0.95,
  "first_mention": {"section": "Introduction", "paragraph": 3},
  "category": "problem"
}

Now extract concepts from the provided section. Return ONLY the JSON array, no additional text.`;
  }

  /**
   * Convert API output to ExtractedEntity
   */
  private toExtractedEntity(
    concept: ConceptExtractionOutput,
    section: Section
  ): ExtractedEntity {
    return {
      id: uuid(),
      type: EntityType.CONCEPT,
      name: concept.name,
      normalizedName: concept.normalized_name,
      aliases: concept.aliases || [],
      confidence: concept.confidence,
      importance: concept.importance,
      firstMention: {
        section: concept.first_mention.section,
        paragraph: concept.first_mention.paragraph
      },
      allMentions: [{
        section: concept.first_mention.section,
        paragraph: concept.first_mention.paragraph
      }],
      definition: concept.definition,
      category: concept.category || 'technique'
    };
  }

  /**
   * Deduplicate concepts by normalizing names and merging similar ones
   */
  private deduplicateConcepts(concepts: ExtractedEntity[]): ExtractedEntity[] {
    const grouped = new Map<string, ExtractedEntity[]>();

    // Group by normalized name (case-insensitive)
    for (const concept of concepts) {
      const key = concept.normalizedName.toLowerCase();
      if (!grouped.has(key)) {
        grouped.set(key, []);
      }
      grouped.get(key)!.push(concept);
    }

    // Merge duplicates
    const deduplicated: ExtractedEntity[] = [];
    for (const [key, group] of grouped.entries()) {
      if (group.length === 1) {
        deduplicated.push(group[0]);
        continue;
      }

      // Merge: take highest confidence, combine aliases and mentions
      const merged = group.reduce((best, current) => {
        if (current.confidence > best.confidence) {
          return current;
        }
        return best;
      });

      // Combine aliases from all duplicates
      const allAliases = new Set<string>();
      for (const concept of group) {
        allAliases.add(concept.name);
        concept.aliases.forEach(alias => allAliases.add(alias));
      }
      allAliases.delete(merged.normalizedName); // Remove canonical name from aliases
      merged.aliases = Array.from(allAliases);

      // Combine all mentions
      merged.allMentions = group.flatMap(c => c.allMentions);

      // Take definition from any that has one
      if (!merged.definition) {
        merged.definition = group.find(c => c.definition)?.definition;
      }

      deduplicated.push(merged);

      logger.debug('Merged duplicate concepts', {
        normalizedName: merged.normalizedName,
        duplicateCount: group.length,
        aliases: merged.aliases
      });
    }

    return deduplicated;
  }

  /**
   * Validate extracted concept
   */
  validateConcept(
    concept: ExtractedEntity,
    paperText: string
  ): { valid: boolean; issues: string[] } {
    const issues: string[] = [];

    // Check if concept appears in paper
    const variants = [concept.name, concept.normalizedName, ...concept.aliases];
    const mentioned = variants.some(v =>
      paperText.toLowerCase().includes(v.toLowerCase())
    );

    if (!mentioned) {
      issues.push('Concept not found in paper text');
    }

    // Check confidence threshold
    if (concept.confidence < 0.5) {
      issues.push('Confidence below acceptable threshold');
    }

    // Check importance for single mentions
    if (concept.allMentions.length === 1 && concept.importance > 7) {
      issues.push('High importance but mentioned only once (possible hallucination)');
    }

    return {
      valid: issues.length === 0,
      issues
    };
  }
}

/**
 * Factory function to create ConceptExtractor with configuration
 */
export function createConceptExtractor(config: {
  apiKey: string;
}): ConceptExtractor {
  return new ConceptExtractor(config.apiKey);
}
