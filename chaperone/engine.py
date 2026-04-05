import torch
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

from chaperone.utils.logger import logger
from chaperone.tools.rcsb_fetcher import fetch_pdb_metadata, download_pdb_file
from chaperone.tools.slurm_runner import submit_job, create_slurm_script

class GemmaEngine:
    def __init__(self):
        logger.info("Loading Gemma 4 weights into A5500 VRAM via HuggingFace for Agent loop...")
        
        self.model_id = "google/gemma-4-26b-a4b" 
        CACHE_DIR = "/private/groups/yehlab/wsobolew/02_projects/computational/Chaperone-RAG/model_cache"
        os.environ["HF_HOME"] = CACHE_DIR
        
        self.tools = [fetch_pdb_metadata, download_pdb_file, submit_job, create_slurm_script]
        
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
            self._setup_agent()
            
        except Exception as e:
            logger.warning(f"Failed to load full weights. Initializing Agent in MOCK mode. ERROR: {e}")
            self.model = None
            self.llm = None
            self.agent_executor = None

    def _setup_agent(self):
        """Builds the ReAct Agent orchestration loop connecting the local Gemma LLM to the tools."""
        # Generic ReAct prompt template adapted for Open Source Models
        template = '''Answer the following questions as best you can. You are an expert computational biologist interacting with a HPC environment.
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
Thought:{agent_scratchpad}'''

        prompt = PromptTemplate.from_template(template)
        
        # Create ReAct agent and Executor to manage the loop
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5 # Prevent infinite loops
        )
        logger.info("AgentExecutor and ReAct routing successfully initialized.")

    def chat(self, prompt: str) -> str:
        """
        Sends the augmented prompt through the LangChain Agent loop.
        """
        if not self.model or not self.agent_executor:
            return f"[Mock Mode - Agent Logic Disabled]: I would trigger my ReAct loop to analyze and utilize [{', '.join([t.name for t in self.tools])}] for your prompt: '{prompt}'"
            
        try:
            # Let the agent autonomously decide which tools to pull and call
            response = self.agent_executor.invoke({"input": prompt})
            return response.get("output", "No response generated.")
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return f"An error occurred during agent reasoning: {e}"
