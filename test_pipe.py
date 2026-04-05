import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import SystemMessage
import os

model_id = "google/gemma-4-26b-a4b"
CACHE_DIR = "model_cache"
t = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
# loading in CPU or meta just for quick test of init
mod = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=CACHE_DIR, device_map="cpu", torch_dtype=torch.float16)

stop_ids = [t.eos_token_id]
for tk in ["<end_of_turn>", "<start_of_turn>", "<eos>"]:
    cid = t.convert_tokens_to_ids(tk)
    if cid and cid != t.unk_token_id and cid not in stop_ids:
        stop_ids.append(cid)

print("STOP IDS:", stop_ids)
p = pipeline("text-generation", model=mod, tokenizer=t, max_new_tokens=22, max_length=None, stop_strings=["<eos>", "<end_of_turn>"], eos_token_id=stop_ids, pad_token_id=t.eos_token_id, do_sample=True, temperature=0.1)

hfp = HuggingFacePipeline(pipeline=p)
c = ChatHuggingFace(llm=hfp, model_id=model_id)

print("PIPELINE TEST SUCCESSFUL")
