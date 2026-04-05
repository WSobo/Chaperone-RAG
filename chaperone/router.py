import yaml
from typing import List, Dict
from collections import defaultdict
from chaperone.utils.logger import logger

class AgentRouter:
    """
    Intelligently routes natural language to the correct Persona or specialized toolsets.
    Inspired by OmicsAgent and everything-claude-code.
    """
    def __init__(self):
        self.personas_dir = "configs/personas"
        # Mappings of keywords to specific "Skills/Personas"
        self.skill_registry = {
            "orchestrator": ["plan", "design", "architecture", "reproduce", "orchestrate", "experiment"],
            "structural_engineer": ["pdb", "pymol", "alphafold", "rfdiffusion", "proteinmpnn", "residue", "boltz"],
            "literature_researcher": ["search", "find", "arxiv", "paper", "literature", "who", "what", "where"]
        }
        logger.info("Agent Router initialized.")

    def route_intent(self, prompt: str) -> str:
        """Determines the best Persona to activate based on the user's prompt."""
        prompt_lower = prompt.lower()
        matched = []
        
        for persona, keywords in self.skill_registry.items():
            if any(kw in prompt_lower for kw in keywords):
                matched.append(persona)
                
        # Default to a general scientific assistant if no clear match.
        best_persona = matched[0] if matched else "default_assistant"
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

        if persona == "structural_engineer":
            base_instructions = """You are a Principal Structural Protein Engineer. 
Your focus is 3D coordinates, folding algorithms (AlphaFold, RFdiffusion), and molecular dynamics.
When writing scripts to analyze PDBs (e.g., using BioPython in the sandbox), always ensure you handle missing atoms correctly.

""" + base_instructions
        
        elif persona == "orchestrator":
             base_instructions = """You are the Chief Scientist and HPC Orchestrator.
Your role focuses on reproducibility, generating clean bash scripts, and tracking provenance. 
Always use the `generate_reproducibility_bundle` tool when concluding an experiment to ensure MLOps best practices.

""" + base_instructions

        return base_instructions
