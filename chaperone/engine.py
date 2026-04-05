import torch
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

from chaperone.utils.logger import logger
from chaperone.tools.rcsb_fetcher import fetch_pdb_metadata, download_pdb_file
from chaperone.tools.slurm_runner import submit_job, create_slurm_script
from chaperone.tools.sandbox import execute_python_script
from chaperone.tools.literature import search_literature, web_search
from chaperone.tools.reproducibility import generate_reproducibility_bundle
from chaperone.router import AgentRouter

class GemmaEngine:
    def __init__(self):
        logger.info("Loading Gemma 4 weights into A5500 VRAM via HuggingFace for Agent loop...")
        
        self.model_id = "google/gemma-4-26b-a4b" 
        CACHE_DIR = "/private/groups/yehlab/wsobolew/02_projects/computational/Chaperone-RAG/model_cache"
        os.environ["HF_HOME"] = CACHE_DIR
        
        self.tools = [
            fetch_pdb_metadata, 
            download_pdb_file, 
            submit_job, 
            create_slurm_script,
            execute_python_script,
            search_literature,
            web_search,
            generate_reproducibility_bundle
        ]
        
        # Initialize the router to dynamically swap prompts based on intent
        self.router = AgentRouter()
        
        try:
            # Load tokenizer and model onto available GPUs
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=CACHE_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                cache_dir=CACHE_DIR
            )
            
            # Create standard HuggingFace pipeline for text generation
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.1, # Low temperature for accurate tool use reasoning
                return_full_text=False
            )
            
            # Wrap in LangChain's HuggingFacePipeline
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("Model pipeline loaded successfully.")
            
        except Exception as e:
            logger.warning(f"Failed to load full weights. Initializing Agent in MOCK mode. ERROR: {e}")
            self.model = None
            self.llm = None
            self.agent_executor = None

    def _get_agent_executor(self, user_prompt: str) -> AgentExecutor:
        """
        Dynamically constructs the AgentExecutor using a persona mapped by the Router.
        """
        if not self.llm:
            return None
            
        # Determine the appropriate persona from everything-claude-code concepts
        best_persona = self.router.route_intent(user_prompt)
        template_str = self.router.load_persona_prompt(best_persona)
        prompt = PromptTemplate.from_template(template_str)
        
        # Create ReAct agent and Executor to manage the loop
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5 # Prevent infinite loops
        )
        return agent_executor

    def chat(self, prompt: str) -> str:
        """
        Sends the augmented prompt through the LangChain Agent loop.
        """
        executor = self._get_agent_executor(prompt)
        if not self.model or not executor:
            return f"[Mock Mode - Agent Logic Disabled]: I would trigger my ReAct loop to analyze and utilize [{', '.join([t.name for t in self.tools])}] for your prompt: '{prompt}'"
            
        try:
            # Let the agent autonomously decide which tools to pull and call
            response = executor.invoke({"input": prompt})
            return response.get("output", "No response generated.")
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return f"An error occurred during agent reasoning: {e}"
