# Corpus Selection Strategy

## Table of Contents
1. [Overview](#overview)
2. [Selection Algorithms](#selection-algorithms)
3. [Priority Scoring](#priority-scoring)
4. [Implementation](#implementation)
5. [Gaussian Splatting Corpus](#gaussian-splatting-corpus)

---

## Overview

### Challenge

Given a seed paper (e.g., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"), how do we automatically select 50-100 related papers that:
- Are scientifically relevant
- Cover the research landscape comprehensively
- Include foundational work, improvements, and applications
- Represent diverse perspectives and approaches

### Goals

1. **Relevance**: Papers should be directly related to the research topic
2. **Diversity**: Cover different aspects (methods, theory, applications)
3. **Impact**: Include influential and highly-cited work
4. **Recency**: Balance between foundational papers and recent developments
5. **Completeness**: Form a connected graph with meaningful relationships

---

## Selection Algorithms

### 1. Citation-Based BFS (Breadth-First Search)

**Best for**: Discovering research lineage and direct influences

**Algorithm**:
```python
def citation_bfs(seed_paper: str, max_papers: int = 100, max_depth: int = 3) -> List[str]:
    """
    Expand corpus using breadth-first traversal of citation graph.

    Args:
        seed_paper: arXiv ID of seed paper
        max_papers: Maximum papers to select
        max_depth: Maximum citation hops from seed

    Returns:
        List of selected arXiv IDs
    """
    selected = {seed_paper}
    queue = [(seed_paper, 0)]  # (paper_id, depth)
    visited = {seed_paper}

    while queue and len(selected) < max_papers:
        paper_id, depth = queue.pop(0)

        if depth >= max_depth:
            continue

        # Get papers this paper cites (backward citations)
        cited_papers = fetch_cited_papers(paper_id)

        # Get papers that cite this paper (forward citations)
        citing_papers = fetch_citing_papers(paper_id)

        # Combine and score
        candidates = cited_papers + citing_papers

        # Sort by relevance score
        candidates = sorted(
            candidates,
            key=lambda p: calculate_relevance_score(p, seed_paper),
            reverse=True
        )

        # Add top candidates
        for candidate in candidates[:10]:  # Top 10 per paper
            if candidate not in visited and len(selected) < max_papers:
                selected.add(candidate)
                queue.append((candidate, depth + 1))
                visited.add(candidate)

    return list(selected)
```

**Characteristics**:
- Explores both backward (references) and forward (citations) links
- Discovers papers across different time periods
- Can find foundational work (backward) and recent improvements (forward)

**Limitations**:
- May miss parallel work not cited by seed paper
- Biased toward citation-rich areas

---

### 2. Citation-Based DFS (Depth-First Search)

**Best for**: Following specific research threads deeply

**Algorithm**:
```python
def citation_dfs(seed_paper: str, max_papers: int = 100, max_depth: int = 5) -> List[str]:
    """
    Expand corpus using depth-first traversal.
    Follows citation chains deeply before breadth.
    """
    selected = {seed_paper}
    visited = {seed_paper}

    def dfs_helper(paper_id: str, depth: int):
        if depth >= max_depth or len(selected) >= max_papers:
            return

        # Get citations
        citations = fetch_cited_papers(paper_id)

        # Sort by relevance
        citations = sorted(
            citations,
            key=lambda p: calculate_relevance_score(p, seed_paper),
            reverse=True
        )

        for citation in citations:
            if citation not in visited and len(selected) < max_papers:
                visited.add(citation)
                selected.add(citation)
                dfs_helper(citation, depth + 1)  # Recursive depth-first

    dfs_helper(seed_paper, 0)
    return list(selected)
```

**Characteristics**:
- Follows methodological lineage deeply
- Good for understanding historical development

**Limitations**:
- May miss breadth of related work
- Can get stuck in one research thread

---

### 3. Semantic Similarity Search

**Best for**: Finding topically related papers regardless of citations

**Algorithm**:
```python
def semantic_similarity_search(
    seed_paper: str,
    max_papers: int = 100,
    similarity_threshold: float = 0.7
) -> List[str]:
    """
    Select papers based on semantic similarity to seed.

    Uses embeddings of title + abstract for similarity computation.
    """
    # Get seed paper embedding
    seed_metadata = fetch_paper(seed_paper)
    seed_text = f"{seed_metadata.title} {seed_metadata.abstract}"
    seed_embedding = embed_text(seed_text)

    # Search arXiv for candidate papers
    # Option 1: Search by category
    candidates = search_arxiv(
        categories=seed_metadata.categories,
        date_range=(seed_metadata.date - timedelta(days=365*3),
                   datetime.now())
    )

    # Compute similarities
    papers_with_scores = []
    for candidate in candidates:
        candidate_text = f"{candidate.title} {candidate.abstract}"
        candidate_embedding = embed_text(candidate_text)

        similarity = cosine_similarity(seed_embedding, candidate_embedding)

        if similarity >= similarity_threshold:
            papers_with_scores.append((candidate.arxiv_id, similarity))

    # Sort by similarity and return top N
    papers_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [paper_id for paper_id, _ in papers_with_scores[:max_papers]]
```

**Characteristics**:
- Finds topically similar work
- Not limited to citation network
- Can discover parallel independent developments

**Limitations**:
- Requires embedding model
- May include less influential work

---

### 4. Hybrid Approach (Recommended)

**Best for**: Comprehensive corpus with both citation links and semantic relevance

**Algorithm**:
```python
def hybrid_selection(
    seed_paper: str,
    max_papers: int = 100,
    citation_weight: float = 0.6,
    semantic_weight: float = 0.4
) -> List[str]:
    """
    Combine citation-based and semantic similarity approaches.

    Args:
        seed_paper: Seed arXiv ID
        max_papers: Target corpus size
        citation_weight: Weight for citation-based score
        semantic_weight: Weight for semantic similarity

    Returns:
        Selected papers ranked by combined score
    """
    # Phase 1: Get citation network (50% of budget)
    citation_papers = citation_bfs(seed_paper, max_papers=max_papers//2)

    # Phase 2: Get semantically similar papers
    semantic_papers = semantic_similarity_search(
        seed_paper,
        max_papers=max_papers//2
    )

    # Phase 3: Combine and re-rank
    all_candidates = set(citation_papers + semantic_papers)

    papers_with_scores = []
    for paper in all_candidates:
        # Citation score (based on graph centrality)
        citation_score = calculate_citation_score(paper, citation_papers)

        # Semantic score
        semantic_score = calculate_semantic_similarity(paper, seed_paper)

        # Combined score
        combined_score = (
            citation_weight * citation_score +
            semantic_weight * semantic_score
        )

        papers_with_scores.append((paper, combined_score))

    # Sort and return top N
    papers_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [paper for paper, _ in papers_with_scores[:max_papers]]
```

---

## Priority Scoring

### Multi-Factor Scoring Function

```python
def calculate_priority_score(paper: PaperMetadata, seed_paper: str) -> float:
    """
    Calculate priority score for paper selection.

    Higher score = higher priority for inclusion in corpus.

    Factors:
    - Citation count (influence)
    - Recency (balance old foundational with new developments)
    - Relevance to seed paper (semantic + citation)
    - Venue quality (top conferences weighted higher)
    - Author reputation (h-index)
    """
    scores = {}

    # 1. Citation impact (0-30 points)
    scores['citation'] = min(math.log10(paper.citation_count + 1) * 3, 30)

    # 2. Recency (0-20 points)
    days_since_published = (datetime.now() - paper.publication_date).days
    if days_since_published < 30:
        scores['recency'] = 20
    elif days_since_published < 90:
        scores['recency'] = 15
    elif days_since_published < 365:
        scores['recency'] = 10
    elif days_since_published < 365 * 3:
        scores['recency'] = 5
    else:
        scores['recency'] = 2  # Old foundational work still valuable

    # 3. Semantic relevance (0-30 points)
    similarity = calculate_semantic_similarity(paper, seed_paper)
    scores['relevance'] = similarity * 30

    # 4. Venue quality (0-15 points)
    venue_scores = {
        'CVPR': 15, 'ICCV': 15, 'ECCV': 15, 'NeurIPS': 15, 'ICML': 15,
        'SIGGRAPH': 15, 'TOG': 14, 'TPAMI': 14,
        'ICLR': 13, 'BMVC': 10, 'arXiv': 5
    }
    scores['venue'] = venue_scores.get(paper.venue, 5)

    # 5. Citation link strength (0-5 points)
    # Bonus if paper is directly cited by or cites seed paper
    if paper.arxiv_id in get_cited_papers(seed_paper):
        scores['citation_link'] = 5
    elif seed_paper in get_cited_papers(paper.arxiv_id):
        scores['citation_link'] = 5
    else:
        scores['citation_link'] = 0

    # Total: 0-100
    return sum(scores.values())
```

### Category-Based Balancing

Ensure corpus diversity across categories:

```python
def balance_corpus_by_category(
    papers: List[str],
    target_distribution: Dict[str, float]
) -> List[str]:
    """
    Ensure corpus has balanced representation across categories.

    Args:
        papers: Candidate papers
        target_distribution: Desired distribution
            e.g., {"methods": 0.4, "theory": 0.2, "applications": 0.2, "datasets": 0.1, "surveys": 0.1}

    Returns:
        Balanced selection
    """
    # Classify papers
    classified = defaultdict(list)
    for paper in papers:
        category = classify_paper_type(paper)
        classified[category].append(paper)

    # Select from each category
    balanced = []
    for category, target_frac in target_distribution.items():
        target_count = int(len(papers) * target_frac)
        category_papers = classified[category]

        # Sort by priority within category
        category_papers.sort(
            key=lambda p: calculate_priority_score(p),
            reverse=True
        )

        balanced.extend(category_papers[:target_count])

    return balanced
```

---

## Implementation

### Complete Selection Pipeline

```python
class CorpusSelector:
    """
    Orchestrates corpus selection with configurable strategies.
    """

    def __init__(
        self,
        strategy: str = "hybrid",
        max_papers: int = 100,
        max_depth: int = 3
    ):
        self.strategy = strategy
        self.max_papers = max_papers
        self.max_depth = max_depth

    async def select_corpus(self, seed_paper: str) -> List[str]:
        """
        Select corpus starting from seed paper.

        Args:
            seed_paper: arXiv ID of seed paper

        Returns:
            List of selected arXiv IDs
        """
        logger.info(f"Starting corpus selection from seed {seed_paper}")

        if self.strategy == "citation_bfs":
            papers = await self._citation_bfs(seed_paper)
        elif self.strategy == "citation_dfs":
            papers = await self._citation_dfs(seed_paper)
        elif self.strategy == "semantic":
            papers = await self._semantic_search(seed_paper)
        elif self.strategy == "hybrid":
            papers = await self._hybrid_selection(seed_paper)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Balance by category
        papers = self._balance_by_category(papers)

        # Final ranking by priority
        papers = self._rank_by_priority(papers, seed_paper)

        logger.info(f"Selected {len(papers)} papers for corpus")
        return papers[:self.max_papers]

    def _balance_by_category(self, papers: List[str]) -> List[str]:
        """Ensure category diversity."""
        target_dist = {
            "methods": 0.5,      # 50% methods papers
            "applications": 0.2,  # 20% applications
            "theory": 0.15,       # 15% theoretical analysis
            "surveys": 0.10,      # 10% surveys/reviews
            "datasets": 0.05      # 5% datasets/benchmarks
        }
        return balance_corpus_by_category(papers, target_dist)

    def _rank_by_priority(
        self,
        papers: List[str],
        seed_paper: str
    ) -> List[str]:
        """Rank papers by priority score."""
        papers_with_scores = [
            (paper, calculate_priority_score(paper, seed_paper))
            for paper in papers
        ]
        papers_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [paper for paper, _ in papers_with_scores]
```

---

## Gaussian Splatting Corpus

### Seed Paper

**"3D Gaussian Splatting for Real-Time Radiance Field Rendering"**
- arXiv ID: 2308.04079
- Published: August 2023
- Authors: Bernhard Kerbl et al. (Inria)
- Impact: 247+ citations in 6 months

### Selection Strategy for Gaussian Splatting

**Goal**: Build comprehensive corpus covering:
1. Foundational work (NeRF, point-based rendering)
2. The 3D-GS paper itself
3. Improvements and extensions of 3D-GS
4. Related real-time rendering methods
5. Applications of 3D-GS

**Configuration**:
```python
selector = CorpusSelector(
    strategy="hybrid",
    max_papers=100,
    max_depth=2
)

corpus = await selector.select_corpus(
    seed_paper="2308.04079",
    categories=["cs.CV", "cs.GR"],
    date_range=(datetime(2020, 1, 1), datetime.now())
)
```

### Expected Corpus Composition

**Foundational (15 papers)**:
- NeRF (Mildenhall et al., 2020)
- Mip-NeRF (Barron et al., 2021)
- Point-based rendering classics
- Volumetric rendering theory

**Direct Lineage (20 papers)**:
- Instant-NGP (Müller et al., 2022)
- Mip-NeRF 360 (Barron et al., 2022)
- Plenoxels (Yu et al., 2021)
- TensoRF (Chen et al., 2022)

**Extensions of 3D-GS (25 papers)**:
- Mip-Splatting (anti-aliasing)
- Scaffold-GS (memory optimization)
- Dynamic 3D-GS (temporal modeling)
- 4D-GS (spacetime representation)

**Applications (20 papers)**:
- Avatar generation with GS
- Large-scale scene reconstruction
- AR/VR applications
- Mobile/edge deployment

**Comparative Methods (15 papers)**:
- Alternative real-time methods
- Neural rendering surveys
- Benchmarking papers

**Datasets & Benchmarks (5 papers)**:
- Mip-NeRF 360 dataset
- Tanks and Temples
- DTU dataset

---

## Metrics for Evaluation

How do we know if corpus selection is good?

### Coverage Metrics

```python
def evaluate_corpus_coverage(corpus: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """
    Evaluate corpus quality against ground truth (e.g., manually curated survey).

    Returns:
        - recall: fraction of important papers included
        - precision: fraction of included papers that are important
        - diversity: entropy of paper categories
        - connectivity: fraction of papers with >1 citation link
    """
    return {
        "recall": len(set(corpus) & set(ground_truth)) / len(ground_truth),
        "precision": len(set(corpus) & set(ground_truth)) / len(corpus),
        "diversity": calculate_category_entropy(corpus),
        "connectivity": calculate_citation_connectivity(corpus)
    }
```

### Quality Indicators

- **Average citation count**: Should be >50 for good corpus
- **Recent paper fraction**: At least 30% from last 2 years
- **Citation connectivity**: >70% of papers should have citation links to others
- **Category balance**: No single category >60% of corpus

---

## Future Enhancements

1. **Active Learning**: Learn from human feedback on corpus quality
2. **Multi-hop Reasoning**: "Find papers that improve X which improves Y"
3. **Controversy Detection**: Include papers with conflicting claims
4. **Temporal Tracking**: Maintain corpus freshness as new papers publish
5. **Cross-Domain Links**: Include relevant papers from adjacent fields

---

## Summary

Corpus selection is critical for knowledge graph quality. Our **hybrid approach** combining citation-based traversal with semantic similarity provides:
- **Comprehensive coverage** of research landscape
- **Balanced representation** across time periods and paper types
- **Priority-based ranking** ensuring important work included
- **Scalable algorithms** supporting large-scale corpus construction

For the Gaussian Splatting domain, this produces a corpus of 100 high-quality papers spanning foundational work, the seminal paper, extensions, and applications.
