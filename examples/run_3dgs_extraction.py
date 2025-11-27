"""
Example: Extract Knowledge Graph from 3D Gaussian Splatting Paper

This script demonstrates the complete pipeline for processing the seminal
"3D Gaussian Splatting for Real-Time Radiance Field Rendering" paper.

Usage:
    python examples/run_3dgs_extraction.py

Expected output:
    - Extracted entities (concepts, methods, metrics, datasets)
    - Extracted relationships with evidence
    - Quality report
    - JSON output files
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

# Example mock data for demonstration
# In production, this would use actual API calls and database operations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_example_extraction_output():
    """
    Create example output showing what the system would extract from
    the 3D Gaussian Splatting paper.

    This demonstrates the complete data model and expected output format.
    """

    # Example paper metadata
    paper_metadata = {
        "arxiv_id": "2308.04079",
        "title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
        "authors": [
            {"name": "Bernhard Kerbl", "affiliation": "Inria"},
            {"name": "Georgios Kopanas", "affiliation": "Inria"},
            {"name": "Thomas Leimkühler", "affiliation": "Inria"},
            {"name": "George Drettakis", "affiliation": "Inria"}
        ],
        "abstract": "Radiance Field methods have recently revolutionized novel-view synthesis...",
        "publication_date": "2023-08-07",
        "categories": ["cs.CV", "cs.GR"],
        "venue": "ACM TOG (SIGGRAPH 2023)"
    }

    # Example extracted entities
    entities = {
        "concepts": [
            {
                "id": "concept_001",
                "name": "Neural Radiance Fields",
                "normalized_name": "Neural Radiance Fields",
                "aliases": ["NeRF", "NeRFs", "neural radiance field"],
                "type": "concept",
                "importance": 10,
                "confidence": 0.98,
                "definition": "Continuous volumetric scene representations encoded in neural network weights",
                "first_mention": {"section": "Introduction", "paragraph": 2}
            },
            {
                "id": "concept_002",
                "name": "real-time rendering",
                "normalized_name": "Real-Time Rendering",
                "aliases": ["realtime rendering", "interactive rendering"],
                "type": "concept",
                "importance": 9,
                "confidence": 0.95,
                "definition": "Rendering at interactive frame rates (≥30 FPS)",
                "first_mention": {"section": "Abstract", "paragraph": 1}
            },
            {
                "id": "concept_003",
                "name": "novel view synthesis",
                "normalized_name": "Novel View Synthesis",
                "aliases": ["view synthesis", "image synthesis"],
                "type": "concept",
                "importance": 8,
                "confidence": 0.92,
                "definition": "Generating new camera viewpoints from input images",
                "first_mention": {"section": "Introduction", "paragraph": 1}
            }
        ],
        "methods": [
            {
                "id": "method_001",
                "name": "3D Gaussian Splatting",
                "normalized_name": "3D Gaussian Splatting",
                "aliases": ["3D-GS", "3DGS", "Gaussian Splatting"],
                "type": "method",
                "category": "proposed",
                "importance": 10,
                "confidence": 0.97,
                "description": "Represents 3D scenes as explicit anisotropic 3D Gaussians with learnable parameters",
                "parameters": {
                    "primitive": "anisotropic 3D Gaussians",
                    "color_encoding": "spherical harmonics",
                    "rendering": "tile-based rasterization"
                },
                "first_mention": {"section": "Title", "paragraph": 0}
            },
            {
                "id": "method_002",
                "name": "Instant-NGP",
                "normalized_name": "Instant Neural Graphics Primitives",
                "aliases": ["Instant-NGP", "InstantNGP"],
                "type": "method",
                "category": "baseline",
                "importance": 7,
                "confidence": 0.94,
                "description": "Fast NeRF training using hash encoding",
                "first_mention": {"section": "Related Work", "paragraph": 5}
            }
        ],
        "metrics": [
            {
                "id": "metric_001",
                "name": "PSNR",
                "normalized_name": "Peak Signal-to-Noise Ratio",
                "aliases": ["PSNR", "peak signal to noise ratio"],
                "type": "metric",
                "unit": "dB",
                "direction": "higher_is_better",
                "importance": 8,
                "confidence": 0.99,
                "values": [
                    {"method": "3D Gaussian Splatting", "value": 28.5, "dataset": "Mip-NeRF 360"},
                    {"method": "NeRF", "value": 26.1, "dataset": "Mip-NeRF 360"},
                    {"method": "Instant-NGP", "value": 27.2, "dataset": "Mip-NeRF 360"}
                ]
            },
            {
                "id": "metric_002",
                "name": "rendering_fps",
                "normalized_name": "Rendering Frames Per Second",
                "aliases": ["FPS", "frames per second", "rendering speed"],
                "type": "metric",
                "unit": "fps",
                "direction": "higher_is_better",
                "importance": 10,
                "confidence": 0.98,
                "values": [
                    {"method": "3D Gaussian Splatting", "value": 30.0},
                    {"method": "NeRF", "value": 0.033},
                    {"method": "Instant-NGP", "value": 2.0}
                ]
            },
            {
                "id": "metric_003",
                "name": "training_time",
                "normalized_name": "Training Time",
                "aliases": ["training time", "optimization time"],
                "type": "metric",
                "unit": "minutes",
                "direction": "lower_is_better",
                "importance": 7,
                "confidence": 0.91,
                "values": [
                    {"method": "3D Gaussian Splatting", "value": 30},
                    {"method": "NeRF", "value": 240}
                ]
            }
        ],
        "datasets": [
            {
                "id": "dataset_001",
                "name": "Mip-NeRF 360",
                "normalized_name": "Mip-NeRF 360 Dataset",
                "aliases": ["MipNeRF 360", "Mip-NeRF360"],
                "type": "dataset",
                "dataset_type": "evaluation",
                "importance": 9,
                "confidence": 0.96,
                "description": "360-degree unbounded scene captures"
            },
            {
                "id": "dataset_002",
                "name": "Tanks and Temples",
                "normalized_name": "Tanks and Temples",
                "aliases": ["T&T", "Tanks&Temples"],
                "type": "dataset",
                "dataset_type": "evaluation",
                "importance": 7,
                "confidence": 0.94,
                "description": "Large-scale reconstruction benchmark"
            }
        ]
    }

    # Example extracted relationships
    relationships = [
        {
            "id": "rel_001",
            "source_entity": "3D Gaussian Splatting",
            "source_id": "method_001",
            "target_entity": "Neural Radiance Fields",
            "target_id": "concept_001",
            "relation_type": "improves_on",
            "confidence": 0.96,
            "evidence": {
                "quote": "Our method achieves real-time rendering (≥30 fps at 1080p) while maintaining competitive quality, compared to NeRF's 30-second per-frame rendering time.",
                "section": "Abstract",
                "paragraph": 1
            },
            "quantitative": {
                "metric": "rendering_speed",
                "source_value": 30.0,
                "target_value": 0.033,
                "improvement": "900x faster",
                "improvement_percent": 90000
            }
        },
        {
            "id": "rel_002",
            "source_entity": "3D Gaussian Splatting",
            "source_id": "method_001",
            "target_entity": "Instant-NGP",
            "target_id": "method_002",
            "relation_type": "improves_on",
            "confidence": 0.91,
            "evidence": {
                "quote": "We achieve higher quality than Instant-NGP on complex outdoor scenes (average PSNR 28.5 vs 27.1) while matching its rendering speed.",
                "section": "Results",
                "paragraph": 12
            },
            "quantitative": {
                "metric": "PSNR",
                "source_value": 28.5,
                "target_value": 27.1,
                "improvement": "1.4 dB higher",
                "improvement_percent": 5.15
            },
            "qualifications": ["on complex outdoor scenes"]
        },
        {
            "id": "rel_003",
            "source_entity": "3D Gaussian Splatting",
            "source_id": "method_001",
            "target_entity": "Mip-NeRF 360",
            "target_id": "dataset_001",
            "relation_type": "evaluates_on",
            "confidence": 0.99,
            "evidence": {
                "quote": "We evaluate on the Mip-NeRF 360 dataset",
                "section": "Experiments",
                "paragraph": 3
            }
        },
        {
            "id": "rel_004",
            "source_entity": "3D Gaussian Splatting",
            "source_id": "method_001",
            "target_entity": "PSNR",
            "target_id": "metric_001",
            "relation_type": "uses_metric",
            "confidence": 0.97,
            "evidence": {
                "quote": "We report PSNR, SSIM, and LPIPS for quantitative evaluation",
                "section": "Experiments",
                "paragraph": 4
            }
        }
    ]

    # Quality report
    quality_report = {
        "paper_id": "2308.04079",
        "timestamp": datetime.utcnow().isoformat(),
        "overall_score": 0.92,
        "validations": {
            "entities": {
                "total_entities": 47,
                "passed_validation": 45,
                "failed": [
                    {
                        "entity_id": "concept_x",
                        "entity_name": "generic term",
                        "reason": "Too generic, mentioned only once",
                        "severity": "low"
                    }
                ]
            },
            "relationships": {
                "total_relationships": 34,
                "passed_validation": 32,
                "failed": [
                    {
                        "relationship_id": "rel_x",
                        "reason": "Evidence quote not found in paper",
                        "severity": "high"
                    }
                ]
            },
            "completeness": {
                "has_main_method": True,
                "has_evaluation_metrics": True,
                "has_datasets": True,
                "has_baselines": True,
                "all_sections_processed": True,
                "score": 1.0
            }
        },
        "recommendation": "accept",
        "issues": []
    }

    # Processing metadata
    processing_metadata = {
        "start_time": "2024-01-15T10:30:00Z",
        "end_time": "2024-01-15T10:32:04Z",
        "duration_ms": 124000,
        "tokens_used": {
            "haiku": 5420,
            "sonnet": 54680
        },
        "cost_usd": 0.18,
        "agent_versions": {
            "concept_extractor": "1.0.0",
            "method_extractor": "1.0.0",
            "relationship_extractor": "1.2.0",
            "entity_linker": "1.1.0",
            "quality_control": "1.0.0"
        }
    }

    return {
        "paper_metadata": paper_metadata,
        "entities": entities,
        "relationships": relationships,
        "quality_report": quality_report,
        "processing_metadata": processing_metadata
    }


async def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Research Knowledge Graph Extraction Demo")
    logger.info("Paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering")
    logger.info("=" * 80)

    # Create output directory
    output_dir = Path("examples/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nGenerating example extraction output...")

    # Generate example output
    result = create_example_extraction_output()

    # Save to files
    files = {
        "paper_metadata": result["paper_metadata"],
        "entities": result["entities"],
        "relationships": result["relationships"],
        "quality_report": result["quality_report"],
        "processing_metadata": result["processing_metadata"]
    }

    for name, data in files.items():
        filepath = output_dir / f"{name}_3dgs.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved {filepath}")

    # Display summary
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 80)

    entities = result["entities"]
    logger.info(f"\nEntities Extracted:")
    logger.info(f"  • Concepts: {len(entities['concepts'])}")
    logger.info(f"  • Methods: {len(entities['methods'])}")
    logger.info(f"  • Metrics: {len(entities['metrics'])}")
    logger.info(f"  • Datasets: {len(entities['datasets'])}")
    logger.info(f"  • Total: {sum(len(v) for v in entities.values())}")

    logger.info(f"\nRelationships Extracted: {len(result['relationships'])}")

    logger.info(f"\nQuality Score: {result['quality_report']['overall_score']:.2f}")
    logger.info(f"Recommendation: {result['quality_report']['recommendation'].upper()}")

    logger.info(f"\nProcessing Time: {result['processing_metadata']['duration_ms'] / 1000:.1f}s")
    logger.info(f"Cost: ${result['processing_metadata']['cost_usd']:.2f}")

    # Display example relationships
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE RELATIONSHIPS")
    logger.info("=" * 80)

    for rel in result["relationships"][:3]:
        logger.info(f"\n{rel['source_entity']} --[{rel['relation_type']}]--> {rel['target_entity']}")
        logger.info(f"  Confidence: {rel['confidence']:.2f}")
        logger.info(f"  Evidence: \"{rel['evidence']['quote'][:100]}...\"")
        if 'quantitative' in rel:
            quant = rel['quantitative']
            logger.info(f"  Improvement: {quant['improvement']} ({quant['metric']})")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Extraction demo completed successfully!")
    logger.info(f"✓ Output files saved to: {output_dir.absolute()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
