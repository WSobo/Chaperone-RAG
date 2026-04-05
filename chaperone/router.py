from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from chaperone.utils.logger import logger


@dataclass(frozen=True)
class AgentProfile:
    """Container for one agent persona definition."""

    name: str
    description: str
    keywords: Tuple[str, ...] = field(default_factory=tuple)
    prompt_preamble: str = ""
    priority: int = 0

    def keyword_score(self, prompt_lower: str) -> int:
        """Simple keyword overlap score for routing."""
        score = 0
        for keyword in self.keywords:
            kw = keyword.lower().strip()
            if kw and kw in prompt_lower:
                score += max(1, len(kw.split()))
        return score


class AgentProfileRegistry:
    """Loads and matches agent profiles from markdown frontmatter files."""

    FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)$", re.DOTALL)

    def __init__(self, agents_dir: str = "configs/agents"):
        self.agents_dir = Path(agents_dir)
        self.profiles: Dict[str, AgentProfile] = self._default_profiles()
        self._load_from_disk()

    def _default_profiles(self) -> Dict[str, AgentProfile]:
        """Fallback profiles keep behavior stable when no config files exist."""
        return {
            "default_assistant": AgentProfile(
                name="default_assistant",
                description="General computational biology assistant",
                keywords=("protein", "biology", "analysis", "model"),
                prompt_preamble=(
                    "You are a senior computational biology assistant. "
                    "Provide concise, reproducible, and scientifically grounded guidance."
                ),
                priority=0,
            ),
            "orchestrator": AgentProfile(
                name="orchestrator",
                description="Chief scientist and HPC orchestration expert",
                keywords=("plan", "design", "architecture", "reproduce", "orchestrate", "experiment"),
                prompt_preamble=(
                    "You are the Chief Scientist and HPC Orchestrator.\n"
                    "Your role focuses on reproducibility, generating clean bash scripts, and tracking provenance.\n"
                    "Always use the `generate_reproducibility_bundle` tool when concluding an experiment."
                ),
                priority=10,
            ),
            "structural_engineer": AgentProfile(
                name="structural_engineer",
                description="Protein structure specialist",
                keywords=(
                    "pdb",
                    "pymol",
                    "alphafold",
                    "rfdiffusion",
                    "proteinmpnn",
                    "residue",
                    "boltz",
                ),
                prompt_preamble=(
                    "You are a Principal Structural Protein Engineer.\n"
                    "Your focus is 3D coordinates, folding algorithms (AlphaFold, RFdiffusion), and molecular dynamics.\n"
                    "When writing scripts to analyze PDBs, handle missing atoms and alternate locations correctly."
                ),
                priority=10,
            ),
            "literature_researcher": AgentProfile(
                name="literature_researcher",
                description="Scientific literature discovery specialist",
                keywords=("search", "find", "arxiv", "paper", "literature", "citation", "review"),
                prompt_preamble=(
                    "You are a literature research specialist.\n"
                    "Use literature and web tools to collect high-quality references before making claims."
                ),
                priority=5,
            ),
        }

    def _load_from_disk(self) -> None:
        if not self.agents_dir.exists():
            logger.info(f"Agent profile directory not found: {self.agents_dir}. Using built-in profiles.")
            return

        files = sorted(self.agents_dir.glob("*.agent.md"))
        for file_path in files:
            try:
                profile = self._parse_profile(file_path)
            except Exception as exc:
                logger.warning(f"Skipping invalid agent profile {file_path}: {exc}")
                continue

            self.profiles[profile.name] = profile
            logger.info(f"Loaded agent profile: {profile.name} ({file_path.name})")

    def _parse_profile(self, file_path: Path) -> AgentProfile:
        raw = file_path.read_text(encoding="utf-8")
        match = self.FRONTMATTER_RE.match(raw)

        if match:
            metadata = yaml.safe_load(match.group(1)) or {}
            body = match.group(2).strip()
        else:
            metadata = {}
            body = raw.strip()

        fallback_name = file_path.stem.replace(".agent", "")
        name = str(metadata.get("name", fallback_name)).strip()
        description = str(metadata.get("description", "Custom agent profile")).strip()
        priority = int(metadata.get("priority", 0))

        raw_keywords = metadata.get("keywords", [])
        if isinstance(raw_keywords, str):
            keywords = tuple(part.strip() for part in raw_keywords.split(",") if part.strip())
        elif isinstance(raw_keywords, list):
            keywords = tuple(str(item).strip() for item in raw_keywords if str(item).strip())
        else:
            keywords = tuple()

        return AgentProfile(
            name=name,
            description=description,
            keywords=keywords,
            prompt_preamble=body,
            priority=priority,
        )

    def best_profile_for_prompt(self, prompt: str) -> AgentProfile:
        prompt_lower = prompt.lower()
        scored: List[Tuple[int, int, AgentProfile]] = []

        for profile in self.profiles.values():
            keyword_score = profile.keyword_score(prompt_lower)
            scored.append((keyword_score, profile.priority, profile))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_keyword_score, _, best_profile = scored[0]

        if best_keyword_score <= 0:
            return self.profiles["default_assistant"]
        return best_profile


class AgentRouter:
    """
    Intelligently routes natural language to the correct Persona or specialized toolsets.
    Inspired by OmicsAgent and everything-claude-code.
    """
    def __init__(self, agents_dir: str = "configs/agents"):
        self.registry = AgentProfileRegistry(agents_dir=agents_dir)
        logger.info(f"Agent Router initialized with {len(self.registry.profiles)} profiles.")

    def route_intent(self, prompt: str) -> str:
        """Determines the best Persona to activate based on the user's prompt."""
        best_persona = self.registry.best_profile_for_prompt(prompt).name
        logger.info(f"Router mapped intent to Persona: [bold cyan]{best_persona}[/bold cyan]")
        return best_persona

    def load_persona_prompt(self, persona: str) -> str:
        """
        Dynamically generates the System Prompt depending on the active Persona.
        Using the "everything-claude-code" methodology of targeted roles and checklists.
        """
        base_instructions = """Answer the following questions as best you can. You are an expert computational biologist interacting with a HPC environment.
You have access to the following tools:

{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    profile = self.registry.profiles.get(persona, self.registry.profiles["default_assistant"])
    preamble = profile.prompt_preamble.strip()
    if not preamble:
        return base_instructions

    return f"{preamble}\n\n{base_instructions}"
