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
        self.chat_state = {
            "menu": None,
            "last_topic": None,
            "turn": 0,
            "fact_idx": 0,
        }
        
        # Initialize the router to dynamically swap prompts based on intent
        self.router = AgentRouter()
        
        try:
            max_new_tokens = int(os.getenv("CHAPERONE_MAX_NEW_TOKENS", "256"))
            max_generation_time = float(os.getenv("CHAPERONE_MAX_GENERATION_TIME", "20"))

            # Load tokenizer and model onto available GPUs
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=CACHE_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                dtype=torch.bfloat16,
                cache_dir=CACHE_DIR
            )

            # Identify correct end-of-turn tokens for Gemma format
            stop_token_ids = [self.tokenizer.eos_token_id]
            
            # The t.vocab dict doesn't actually contain Gemma 4 special tokens; 
            # converting special tokens directly via convert_tokens_to_ids is the correct approach
            for token in ["<end_of_turn>", "<start_of_turn>", "<eos>"]:
                cid = self.tokenizer.convert_tokens_to_ids(token)
                # Some versions of transformers return an unk_token id if it fails to find the token
                if cid and cid != self.tokenizer.unk_token_id and cid not in stop_token_ids:
                    stop_token_ids.append(cid)

            # Keep generation options in one place to avoid pipeline/config conflicts.
            self.model.generation_config.max_new_tokens = max_new_tokens
            self.model.generation_config.do_sample = True
            self.model.generation_config.temperature = 0.1
            self.model.generation_config.pad_token_id = getattr(self.tokenizer, "pad_token_id", self.tokenizer.eos_token_id)
            self.model.generation_config.eos_token_id = stop_token_ids
            self.model.generation_config.max_time = max_generation_time
            self.model.generation_config.max_length = None
            
            # Create standard HuggingFace pipeline for text generation
            # Note: Do not pass generation_config along to pipeline constructor, as the pipeline wrapper
            # automatically extracts it from the model instance and passing it directly causes a conflict
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=False
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
            logger.info(
                f"Model pipeline loaded successfully (max_new_tokens={max_new_tokens}, "
                f"max_generation_time={max_generation_time}s)."
            )
            
        except Exception as e:
            logger.warning(f"Failed to load full weights. Initializing Agent in MOCK mode. ERROR: {e}")
            self.model = None
            self.llm = None
            self.agent_executor = None

    def quick_chat_reply(self, prompt: str) -> str:
        """Fast local fallback for friendly chatter to keep latency low on large models."""
        text = prompt.strip().lower()
        self.chat_state["turn"] += 1

        bio_fun_facts = [
            "Quick biology fun fact: octopuses edit their RNA in neurons, which may help them rapidly adapt brain function.",
            "Quick biology fun fact: tardigrades survive extreme dehydration using proteins that protect cell structures.",
            "Quick biology fun fact: some bacteria exchange DNA directly through tiny bridges in a process called conjugation.",
            "Quick biology fun fact: your gut microbiome can influence metabolism and even aspects of mood signaling.",
        ]

        def next_fun_fact() -> str:
            idx = self.chat_state["fact_idx"] % len(bio_fun_facts)
            self.chat_state["fact_idx"] += 1
            return bio_fun_facts[idx]

        # Handle follow-up menu selections for short conversational turns.
        if self.chat_state.get("menu") == "conversation_topics":
            if re.fullmatch(r"\s*(1|1\)|1\.|one)\s*", text):
                self.chat_state["last_topic"] = "sci_fi"
                self.chat_state["menu"] = None
                return "Nice pick. Sci-fi question: would you rather explore ocean planets, ringworlds, or Dyson spheres first?"
            if re.fullmatch(r"\s*(2|2\)|2\.|two)\s*", text):
                self.chat_state["last_topic"] = "productivity"
                self.chat_state["menu"] = None
                return "Productivity idea: try 25-minute focus sprints with a tiny written goal before each sprint."
            if re.fullmatch(r"\s*(3|3\)|3\.|three)\s*", text):
                self.chat_state["last_topic"] = "bio_fun_fact"
                self.chat_state["menu"] = None
                return next_fun_fact()
            return "Pick 1, 2, or 3 and I will jump into that topic."

        if re.search(r"\bhow are you\b|\bhow r you\b", text):
            return "I am doing well. Thanks for asking. Want to keep chatting or explore a fun science topic?"
        if re.fullmatch(r"\s*casual\s*[;.!?]*\s*", text):
            return "Casual mode on. I am in. Want light banter, random trivia, or a silly would-you-rather?"
        if re.fullmatch(r"\s*(1|1\)|1\.|chat)\s*", text):
            return "Perfect, chat mode it is. What should we riff on first?"
        if re.search(r"\bdo you like cats\b", text):
            return (
                "I do not have personal preferences, but cats are objectively elite: agile, curious, and dramatic in the best way. "
                "Are you a cat person?"
            )
        if re.search(r"\bdo you like\s+([a-zA-Z][a-zA-Z\s-]{1,40})", text):
            return "I do not have personal likes, but I can absolutely talk about that. Tell me your take and I will match your vibe."
        if re.search(r"\b(hbu|wbu|how about you)\b", text):
            return (
                "I am doing well, thanks for asking. Want to keep it casual, or do a fun bio topic like "
                "a weird protein fact?"
            )
        if re.search(r"\b(another one|another|again|one more)\b", text):
            if self.chat_state.get("last_topic") == "bio_fun_fact":
                return next_fun_fact()
            return "Sure. Tell me what lane: casual chat, sci-fi, productivity, or biology fun facts."
        if re.search(r"\b(that wasn'?t|you said|not a quick biology fun fact)\b", text):
            self.chat_state["last_topic"] = "bio_fun_fact"
            return f"Fair call. {next_fun_fact()}"
        if re.search(r"\b(chat|chatting|conversation|pick a conversation)\b", text):
            self.chat_state["menu"] = "conversation_topics"
            return (
                "Absolutely. Pick one and I will run with it: "
                "1) sci-fi ideas, 2) productivity habits, 3) a quick biology fun fact."
            )
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

        fallback_responses = [
            "I am here with you. We can keep it casual, or switch to a science topic whenever you want.",
            "I am with you. Want a fun question, a quick fact, or just open chat?",
            "Happy to keep chatting. Pick a lane: casual, productivity, sci-fi, or biology fun facts.",
        ]
        idx = self.chat_state["turn"] % len(fallback_responses)
        return fallback_responses[idx]

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

    def _direct_chat(self, persona: str, prompt: str) -> str:
        """Runs a direct model chat for non-tool responses."""
        if not self.llm:
            return "Mock Mode Direct Chat: I hear you."

        try:
            if persona == "default_assistant":
                template_str = (
                    "You are a helpful computational biology assistant speaking directly to a user. "
                    "Respond in plain natural language and do not output internal reasoning markers "
                    "like Thought, Action, Observation, or Final Answer labels."
                )
            else:
                template_str = self.router.load_persona_prompt(persona)
            sys_msg = SystemMessage(content=template_str)
            response = self.llm.invoke([sys_msg, {"role": "user", "content": prompt}])
            return response.content
        except Exception as e:
            logger.error(f"Direct text generation failed: {e}")
            return f"An error occurred during chat: {e}"

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
            return self._direct_chat(best_persona, prompt)

        if best_persona == "default_assistant":
            if self.router.is_small_talk(prompt):
                logger.info("Default assistant detected small-talk; using fast local chat reply.")
                return self.quick_chat_reply(prompt)
            logger.info("Using fast default assistant fallback response (no tool loop).")
            return (
                "I can help with either casual chat or a specific biology workflow. "
                "Tell me what you want next: 1) chat, 2) literature search, 3) PDB/structure analysis, "
                "4) SLURM script help."
            )
        
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
