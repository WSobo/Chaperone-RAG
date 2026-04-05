import os
import kagglehub
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Set the kagglehub cache directory to a local folder in the project instead of the home directory
CACHE_DIR = "/private/groups/yehlab/wsobolew/02_projects/computational/Chaperone-RAG/model_cache"
os.environ["KAGGLEHUB_CACHE"] = CACHE_DIR

MODEL_PATH = kagglehub.model_download("google/gemma-4/transformers/gemma-4-26b-a4b")

# Load model
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto"
)