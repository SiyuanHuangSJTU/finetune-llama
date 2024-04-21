import os
os.environ['CUDA_VISIBLE_DEVICES']='5'

import torch
torch.cuda.set_device(torch.cuda.current_device())
device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModelForCausalLM

model_folder = 'merged-llama2-7b-chat-inf582'

tokenizer = AutoTokenizer.from_pretrained(model_folder)

model = AutoModelForCausalLM.from_pretrained(
    model_folder,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    device_map={'':torch.cuda.current_device()}
)


from transformers import pipeline, GenerationConfig

gen_config = GenerationConfig.from_pretrained(model_folder)
gen_config.max_new_tokens = 150
gen_config.temperature = 0.5
gen_config.repetition_penalty = 1.1
gen_config.pad_token_id = tokenizer.eos_token_id

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map={'':torch.cuda.current_device()},
    generation_config=gen_config,
)

from datasets import load_dataset
from random import randrange

dataset = load_dataset("csv", data_files="test_text.csv", split='train')


def prompt_formatter(sample):
    return f"""### Instruction:
You are a helpful, respectful and honest assistant. \
Your task is to give the passage a title that fits the main idea. \
Your answer should be based on the provided passage only. \

### passage:
{sample['text']}

### title:
"""

results_list = []

dataset = [s for s in dataset]

from tqdm.auto import tqdm

for id, sample in tqdm(enumerate(dataset)):
    prompt = prompt_formatter(sample)
    output = pipe(prompt)
    results_list.append([str(id), output[0]['generated_text'][len(prompt):]])
    
    
    
import pandas as pd
submission = pd.DataFrame(results_list, columns=['ID', 'titles'])
submission.to_csv('submission_0326.csv', index=False)