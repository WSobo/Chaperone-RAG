# Agent Profiles

This folder contains markdown-based agent profile definitions used by `AgentRouter`.

## Format

Each profile is a `.agent.md` file with YAML frontmatter, followed by prompt instructions.

Required frontmatter keys:
- `name`: unique profile key used by the router
- `description`: short purpose statement
- `keywords`: list of routing keywords

Optional keys:
- `priority`: tie-break weight when multiple profiles match

## Minimal Template

```markdown
---
name: your_agent_name
description: What this specialist does
keywords:
  - keyword one
  - keyword two
priority: 10
---

System prompt instructions for this specialist.
```

## Current Profiles

Core coding experts:
- `python_ooo_expert.agent.md` - Python architecture, OOP, typing, and maintainability
- `unix_shell_expert.agent.md` - robust shell pipelines and bash scripting
- `slurm_hpc_operator.agent.md` - SLURM scheduling and HPC job operations

Protein and biology domain experts:
- `pdb_api_expert.agent.md` - RCSB PDB metadata and coordinate retrieval
- `protein_structure_analyst.agent.md` - structure interpretation and quality caveats
- `protein_design_advisor.agent.md` - design strategy and candidate prioritization
- `sequence_analysis_expert.agent.md` - FASTA, alignment, conservation, and motif workflows
- `molecular_dynamics_planner.agent.md` - MD setup and trajectory planning

Research workflow experts:
- `literature_scout.agent.md` - evidence collection and benchmark comparison
- `rag_index_engineer.agent.md` - retrieval pipeline quality and grounding
- `reproducibility_auditor.agent.md` - provenance, seeds, manifests, and rerun reliability

## Suggested Minimal Starter Set

If you want to keep only a lean group, start with:
1. `python_ooo_expert.agent.md`
2. `unix_shell_expert.agent.md`
3. `pdb_api_expert.agent.md`
4. `slurm_hpc_operator.agent.md`
5. `reproducibility_auditor.agent.md`