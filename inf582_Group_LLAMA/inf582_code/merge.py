import os
os.environ['CUDA_VISIBLE_DEVICES']='5'

import torch
torch.cuda.set_device(torch.cuda.current_device())
device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_folder = "llama2-7b-chat-inf582"

model = AutoPeftModelForCausalLM.from_pretrained(
    model_folder,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)

# merge the lora adapter and the base model
merged_model = model.merge_and_unload()

from transformers import AutoTokenizer

output_folder = 'merged-llama2-7b-chat-inf582'

# save the merged model and the tokenizer
merged_model.save_pretrained(output_folder, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(model_folder)
tokenizer.save_pretrained(output_folder)