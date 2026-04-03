import os
from gemma import gm

class GemmaEngine:
    def __init__(self):
        print("Loading Gemma 4 weights into A5500 VRAM... (This takes a minute)")
        self.model = gm.nn.Gemma4_26B_A4B() 
        self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_26B_A4B_IT)
        
        self.sampler = gm.text.ChatSampler(
            model=self.model,
            params=self.params,
            multi_turn=True,
        )

    def chat(self, prompt):
        return self.sampler.chat(prompt)
