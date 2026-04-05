from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_core.messages import SystemMessage
import sys
import time

model_id = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="model_cache")
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="model_cache")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
base_llm = HuggingFacePipeline(pipeline=pipe)
llm = ChatHuggingFace(llm=base_llm)

for chunk in llm.stream([SystemMessage(content="Count up:"), {"role": "user", "content": "hello"}]):
    print("Chunk:", repr(chunk.content))
    time.sleep(0.5)

