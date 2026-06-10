"""A small, self-contained protein-design golden set for evaluation.

Bundling a tiny corpus + labeled questions means ``chaperone eval`` runs out of the
box (no user-provided documents needed) and the retrieval metrics are deterministic.
Each corpus doc has a stable id; each question lists the ids that should be retrieved.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# (doc_id, text) — concise, accurate facts about common protein-design methods.
CORPUS: list[tuple[str, str]] = [
    (
        "rfdiffusion",
        "RFdiffusion is a generative diffusion model for protein backbone design. It can "
        "condition on binding hotspots, symmetry, and motif scaffolding to design de novo "
        "binders and scaffolds, and its outputs are typically sequence-designed with ProteinMPNN.",
    ),
    (
        "proteinmpnn",
        "ProteinMPNN is a message-passing neural network that designs amino-acid sequences for "
        "a fixed protein backbone. It improves sequence recovery and experimental success rates "
        "over physics-based methods like Rosetta and runs in seconds.",
    ),
    (
        "alphafold",
        "AlphaFold2 predicts 3D protein structure from sequence using an Evoformer over a multiple "
        "sequence alignment and a structure module. pLDDT scores per-residue confidence and PAE "
        "(predicted aligned error) scores confidence in relative domain positions.",
    ),
    (
        "esmfold",
        "ESMFold predicts protein structure directly from a single sequence using the ESM-2 protein "
        "language model, skipping the multiple sequence alignment. It is much faster than AlphaFold2 "
        "at some cost in accuracy, which is useful for metagenomic-scale folding.",
    ),
    (
        "esm",
        "ESM-2 is a transformer protein language model trained with masked language modeling on UniProt. "
        "Its embeddings capture structural and functional signal and support variant effect prediction "
        "and zero-shot fitness estimation.",
    ),
    (
        "rosetta",
        "Rosetta is a physics-based macromolecular modeling suite used for structure prediction, "
        "docking, and protein design through energy functions and Monte Carlo sampling.",
    ),
]


class GoldenItem(BaseModel):
    question: str
    relevant_doc_ids: list[str] = Field(description="Corpus ids that should be retrieved.")
    ideal_answer: str


GOLDEN: list[GoldenItem] = [
    GoldenItem(
        question="How does RFdiffusion design protein binders?",
        relevant_doc_ids=["rfdiffusion"],
        ideal_answer="RFdiffusion is a diffusion model that designs backbones, conditioning on "
        "binding hotspots and motifs to create de novo binders.",
    ),
    GoldenItem(
        question="What does ProteinMPNN do for a fixed backbone?",
        relevant_doc_ids=["proteinmpnn"],
        ideal_answer="ProteinMPNN designs an amino-acid sequence for a fixed backbone using a "
        "message-passing neural network, outperforming Rosetta on sequence recovery.",
    ),
    GoldenItem(
        question="What do pLDDT and PAE measure in AlphaFold?",
        relevant_doc_ids=["alphafold"],
        ideal_answer="pLDDT is per-residue confidence; PAE is confidence in the relative position "
        "of residues/domains.",
    ),
    GoldenItem(
        question="How is ESMFold different from AlphaFold?",
        relevant_doc_ids=["esmfold"],
        ideal_answer="ESMFold folds from a single sequence with a protein language model and no MSA, "
        "trading accuracy for speed.",
    ),
    GoldenItem(
        question="What is ESM-2 used for besides folding?",
        relevant_doc_ids=["esm"],
        ideal_answer="ESM-2 embeddings support variant effect prediction and zero-shot fitness "
        "estimation.",
    ),
]
