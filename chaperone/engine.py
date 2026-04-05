import torch
import os
import re
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.agents import create_agent
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import SystemMessage

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
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=None # Avoid max_length and max_new_tokens conflict
            )
            
            # Wrap in LangChain's HuggingFacePipeline
            
            # Ensure the tokenizer has a chat template for ChatHuggingFace
            if not getattr(self.tokenizer, "chat_template", None):
                self.tokenizer.chat_template = (
                    "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '\n' }}{% endif %}"
                    "{% for message in messages %}"
                    "{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}"
                    "{% elif message['role'] == 'assistant' %}{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}"
                    "{% elif message['role'] == 'tool' %}{{ '<start_of_turn>tool\n' + message['content'] + '<end_of_turn>\n' }}"
                    "{% endif %}{% endfor %}{{ '<start_of_turn>model\n' }}"
                )
                
            base_llm = HuggingFacePipeline(pipeline=pipe)
            self.llm = ChatHuggingFace(llm=base_llm, tokenizer=self.tokenizer, model_id=self.model_id)
            logger.info("Model pipeline loaded successfully.")
            
        except Exception as e:
            logger.warning(f"Failed to load full weights. Initializing Agent in MOCK mode. ERROR: {e}")
            self.model = None
            self.llm = None
            self.agent_executor = None

    def quick_chat_reply(self, prompt: str) -> str:
        """Fast local fallback for friendly chatter to keep latency low on large models."""
        text = prompt.strip().lower()

        if re.search(r"\b(nice to meet you|great to meet you)\b", text):
            return (
                "Really happy to meet you too. A local AI plus human collaboration is a powerful combo. "
                "If you want, we can keep chatting or pivot into a biology workflow next."
            )
        if re.search(r"\b(hi|hello|hey|yo)\b", text):
            return "Hey there. Good to see you. How is your day going so far?"
        if "gemma" in text:
            return (
                "Gemma 4 is the local model powering this assistant. It can chat, route to specialist personas, "
                "and use tools for biology workflows."
            )
        if re.search(r"\b(awesome|cool|nice|love this)\b", text):
            return "Love that energy. Want to stay in chat mode, or test a real protein design question?"
        if re.search(r"\b(thanks|thank you|thx)\b", text):
            return "You are welcome. I am glad this is working better now."
        if re.search(r"\b(good night|gn|sleep)\b", text):
            return "Good night. Rest well and we can pick this back up anytime."
        return (
            "I am here with you. Tell me what you want to explore and I will adapt, whether it is casual chat "
            "or a computational biology task."
        )

    def _get_agent_executor(self, user_prompt: str, best_persona: str):
        """
        Dynamically constructs the AgentExecutor using a persona mapped by the Router.
        """
        if not self.llm:
            return None
            
        template_str = self.router.load_persona_prompt(best_persona)
        
        # Create agent to manage the loop
        agent = create_agent(self.llm, tools=self.tools, system_prompt=SystemMessage(content=template_str))
        
        return agent

    def chat(self, prompt: str, forced_persona: str | None = None) -> str:
        """
        Sends the augmented prompt through the LangChain Agent loop.
        """
        # Determine the appropriate persona from everything-claude-code concepts
        best_persona = forced_persona if forced_persona else self.router.route_intent(prompt)
        
        if best_persona == "friendly_chatter":
            logger.info("Bypassing agent loop for friendly conversation mode.")
            if os.getenv("CHAPERONE_CHAT_LLM", "0") != "1":
                logger.info("Using fast local chat responses (set CHAPERONE_CHAT_LLM=1 for model chat).")
                return self.quick_chat_reply(prompt)
            if not self.llm:
                return "Mock Mode Direct Chat: Sure! I'm here to chat."
            # Call the LLM directly without tools
            try:
                template_str = self.router.load_persona_prompt(best_persona)
                sys_msg = SystemMessage(content=template_str)
                response = self.llm.invoke([sys_msg, {"role": "user", "content": prompt}])
                return response.content
            except Exception as e:
                logger.error(f"Direct text generation failed: {e}")
                return f"An error occurred during chat: {e}"
        
        executor = self._get_agent_executor(prompt, best_persona)
        if not self.model or not executor:
            return f"[Mock Mode - Agent Logic Disabled]: I would trigger my ReAct loop to analyze and utilize [{', '.join([t.name for t in self.tools])}] for your prompt: '{prompt}'"
            
        try:
            # Let the agent autonomously decide which tools to pull and call
            response = executor.invoke({"messages": [{"role": "user", "content": prompt}]})
            messages = response.get("messages", [])
            if messages:
                return messages[-1].content
            return "No response generated."
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return f"An error occurred during agent reasoning: {e}"
