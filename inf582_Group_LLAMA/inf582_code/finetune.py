import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import torch


torch.cuda.set_device(torch.cuda.current_device())
device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

from datasets import load_dataset

dataset = load_dataset("csv", data_files="train.csv")
train_dataset = dataset['train']

from huggingface_hub import login
login(token = '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')

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

model_id = "meta-llama/Llama-2-7b-chat-hf"

#
# load model in NF4 quantization with double quantization,
# set compute dtype to bfloat16
#
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    device_map={'':torch.cuda.current_device()}
)
model = prepare_model_for_kbit_training(model)


def prompt_formatter(sample):
	return f"""<s>### Instruction:
You are a helpful, respectful and honest assistant. \
Your task is to give the passage a title that fits the main idea. \
Your answer should be based on the provided passage only. \

### passage:
{sample['text']}

### title:
{sample['titles']} </s>"""


from transformers import TrainingArguments, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

#
# construct a Peft model.
# the QLoRA paper recommends LoRA dropout = 0.05 for small models (7B, 13B)
#
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

#
# set up the trainer
#
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

args = TrainingArguments(
    output_dir="llama2-7b-chat-inf582",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=4,
    save_strategy="epoch",
    learning_rate=2e-4,
    optim="paged_adamw_32bit",
    bf16=True,
    fp16=False,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=False,
)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=prompt_formatter,
    args=args,
)


trainer.train()
trainer.save_model()