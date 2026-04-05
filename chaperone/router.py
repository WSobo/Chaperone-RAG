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
            if kw and re.search(rf"\b{re.escape(kw)}\b", prompt_lower):
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

        # Favor friendly chatter when conversational intent clearly outweighs domain intent.
        if self._should_prefer_friendly(prompt_lower) and "friendly_chatter" in self.profiles:
            return self.profiles["friendly_chatter"]

        if best_keyword_score <= 0:
            return self.profiles["default_assistant"]
        return best_profile

    def _should_prefer_friendly(self, prompt_lower: str) -> bool:
        """Decides whether conversational intent is stronger than specialist-domain intent."""
        domain_score = self._domain_signal_score(prompt_lower)
        convo_score = self._conversation_score(prompt_lower)

        if domain_score <= 0 and convo_score >= 2:
            return True

        return convo_score >= domain_score + 2

    def _conversation_score(self, prompt_lower: str) -> int:
        """Generalized conversation scoring for short social prompts."""
        score = 0

        social_patterns = (
            r"\bhow are you\b",
            r"\bhow r you\b",
            r"\bhow about you\b",
            r"\bhow'?s it going\b",
            r"\bwhat'?s it like\b",
            r"\bnice to meet you\b",
            r"\bcasual\b",
            r"\blet'?s keep it casual\b",
            r"\bcasual conversation\b",
            r"\bgood (morning|afternoon|evening|night)\b",
            r"\bgood\s+hbu\b",
            r"\bgood\s+wbu\b",
            r"\bhbu\b",
            r"\bwbu\b",
            r"\bsup\b",
            r"\bwhat'?s up\b",
            r"\bdo you like\b",
            r"\bwhat do you think about\b",
            r"\b(chat|chatting|conversation)\b",
            r"\bpick a conversation\b",
            r"\bthat wasn'?t\b",
            r"\byou said\b",
            r"\bquick biology fun fact\b",
            r"\bumm\b",
            r"\bthanks?\b",
            r"\bthank you\b",
            r"\bwho are you\b",
            r"\btell me about yourself\b",
            r"\bbeing gemma\b",
            r"^\s*[1-3](\)|\.)?\s*$",
            r"^\s*gemma(\s+\d+)?\s*\??\s*$",
        )

        if any(re.search(pattern, prompt_lower) for pattern in social_patterns):
            score += 2

        # General conversational question pattern (less brittle than hardcoded examples).
        if re.search(r"\b(do|did|can|could|would|will|are|were|have|has)\s+you\b", prompt_lower):
            score += 2

        # Keep very short social-like turns in chat mode (e.g., "good hbu", "lol", "nice").
        tokens = re.findall(r"[a-zA-Z0-9']+", prompt_lower)
        if 0 < len(tokens) <= 4:
            casual_tokens = {
                "good",
                "great",
                "fine",
                "okay",
                "ok",
                "hbu",
                "wbu",
                "lol",
                "nice",
                "cool",
                "yep",
                "yeah",
                "nah",
            }
            if any(token in casual_tokens for token in tokens):
                score += 2

        # Lightweight conversational score fallback.
        if "?" in prompt_lower:
            score += 1
        if len(tokens) <= 12:
            score += 1
        if re.search(r"\b(i|you|we|my|me)\b", prompt_lower):
            score += 1
        if re.search(r"\b(hey|hi|hello|yo|lol|haha|casual|chat)\b", prompt_lower):
            score += 1
        return score

    def _looks_like_small_talk(self, prompt_lower: str) -> bool:
        """Backward-compatible helper used by other modules."""
        if self._has_domain_signal(prompt_lower):
            return False
        return self._conversation_score(prompt_lower) >= 3

    def _has_domain_signal(self, prompt_lower: str) -> bool:
        """Detects specialist-domain intent using dynamic profile keywords plus core biology terms."""
        return self._domain_signal_score(prompt_lower) > 0

    def _domain_signal_score(self, prompt_lower: str) -> int:
        """Scores specialist-domain intent using profile keywords and core terms."""
        score = 0
        skip_names = {"friendly_chatter", "default_assistant"}
        for name, profile in self.profiles.items():
            if name in skip_names:
                continue
            for keyword in profile.keywords:
                kw = keyword.lower().strip()
                if kw and re.search(rf"\b{re.escape(kw)}\b", prompt_lower):
                    score += max(1, len(kw.split()))

        # Core domain terms that are often omitted from profile keyword lists.
        extra_domain_terms = (
            "protein",
            "pdb",
            "alphafold",
            "rfdiffusion",
            "slurm",
            "sbatch",
            "sequence",
            "docking",
            "ligand",
            "mutation",
            "bioinformatics",
            "citation",
            "arxiv",
            "paper",
        )
        for term in extra_domain_terms:
            if term in prompt_lower:
                score += 2
        return score

    def profile_names(self) -> List[str]:
        """Returns all known persona names sorted alphabetically."""
        return sorted(self.profiles.keys())


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

    def available_personas(self) -> List[str]:
        """Returns all available persona names."""
        return self.registry.profile_names()

    def resolve_persona_name(self, requested_name: str) -> str | None:
        """Resolves persona names case-insensitively for CLI overrides."""
        requested = requested_name.strip().lower()
        for name in self.registry.profile_names():
            if name.lower() == requested:
                return name
        return None

    def is_small_talk(self, prompt: str) -> bool:
        """Public helper so callers can preserve conversational mode."""
        return self.registry._looks_like_small_talk(prompt.lower())

    def has_domain_signal(self, prompt: str) -> bool:
        """Public helper to detect specialist-domain intent."""
        return self.registry._has_domain_signal(prompt.lower())

    def load_persona_prompt(self, persona: str) -> str:
        """
        Dynamically generates the System Prompt depending on the active Persona.
        Using the "everything-claude-code" methodology of targeted roles and checklists.
        """
        base_instructions = """Answer the following questions as best you can. You are an expert computational biologist interacting with a HPC environment.

    Workflow requirements:
    1. Plan before executing tools. State a concise plan in your Thought.
    2. Prefer the minimum number of tool calls needed for confidence.
    3. Validate assumptions and highlight uncertainty explicitly.
    4. For code or scripts, prioritize safe, reproducible operations.

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
        
        if persona == "friendly_chatter":
            return preamble

        if not preamble:
            return base_instructions

        return f"{preamble}\n\n{base_instructions}"
