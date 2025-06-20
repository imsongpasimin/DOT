import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig, EarlyStoppingCallback
from datasets import load_dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import find_all_linear_names, print_trainable_parameters
from huggingface_hub import login
from const import *

login(token='<Your_Huggingface_Token>')

model_names = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]

checkpoint_llama = ["final_checkpoint_llama_R15", "final_checkpoint_llama_R16", "final_checkpoint_llama_Lap", "final_checkpoint_llama_Rest", "final_checkpoint_llama_M-Rest", "final_checkpoint_llama_M-Lap", "final_checkpoint_llama_Books", "final_checkpoint_llama_Clothing", "final_checkpoint_llama_Hotel"]
checkpoint_qwen = ["final_checkpoint_qwen_R15", "final_checkpoint_qwen_R16", "final_checkpoint_qwen_Lap", "final_checkpoint_qwen_Rest", "final_checkpoint_qwen_M-Rest", "final_checkpoint_qwen_M-Lap", "final_checkpoint_qwen_Books", "final_checkpoint_qwen_Clothing", "final_checkpoint_qwen_Hotel"]
checkpoint_mistral = ["final_checkpoint_mistral_R15", "final_checkpoint_mistral_R16", "final_checkpoint_mistral_Lap", "final_checkpoint_mistral_Rest", "final_checkpoint_mistral_M-Rest", "final_checkpoint_mistral_M-Lap", "final_checkpoint_mistral_Books", "final_checkpoint_mistral_Clothing", "final_checkpoint_mistral_Hotel"]
checkpoints = [checkpoint_llama, checkpoint_qwen, checkpoint_mistral]

file_path = [   
    {
        'train': 'R15_train.json',
        'validation': 'R15_dev.json',
    },
    {
        'train': 'R16_train.json',
        'validation': 'R16_dev.json',
    },
    {
        'train': 'Lap_train.json',
        'validation': 'Lap_dev.json',
    },
    {
        'train': 'Rest_train.json',
        'validation': 'Rest_dev.json',
    },
    {
        'train': 'M-Rest_train.json',
        'validation': 'M-Rest_dev.json',
    },
    {
        'train': 'M-Lap_train.json',
        'validation': 'M-Lap_dev.json',
    },
    {
        'train': 'Books_train.json',
        'validation': 'Books_dev.json',
    },
    {
        'train': 'Clothing_train.json',
        'validation': 'Clothing_dev.json',
    },
    {
        'train': 'Hotel_train.json',
        'validation': 'Hotel_dev.json',
    },
]


def formatting_prompts_func_rest(example):
    path = f"prompts/rest_0shot.txt"
    with open(path, 'r', encoding='utf-8') as fp:
        instruction = fp.read().strip() + "\n"
    output_texts = []     
    for j in range(len(example['Text'])):
        text = f"{instruction}\nText: {example['Text'][j]}\nSentiment Elements: {example['Sentiment Elements'][j]}"
        output_texts.append(text)
    return output_texts

def formatting_prompts_func_lap(example):
    path = f"prompts/lap_0shot.txt"
    with open(path, 'r', encoding='utf-8') as fp:
        instruction = fp.read().strip() + "\n"
    output_texts = []        
    for j in range(len(example['Text'])):
        text = f"{instruction}\nText: {example['Text'][j]}\nSentiment Elements: {example['Sentiment Elements'][j]}"
        output_texts.append(text)
    return output_texts

def formatting_prompts_func_book(example):
    path = f"prompts/books_0shot.txt"
    with open(path, 'r', encoding='utf-8') as fp:
        instruction = fp.read().strip() + "\n"
    output_texts = []        
    for j in range(len(example['Text'])):
        text = f"{instruction}\nText: {example['Text'][j]}\nSentiment Elements: {example['Sentiment Elements'][j]}"
        output_texts.append(text)
    return output_texts

def formatting_prompts_func_cloth(example):
    path = f"prompts/clothing_0shot.txt"
    with open(path, 'r', encoding='utf-8') as fp:
        instruction = fp.read().strip() + "\n"
    output_texts = []         
    for j in range(len(example['Text'])):
        text = f"{instruction}\nText: {example['Text'][j]}\nSentiment Elements: {example['Sentiment Elements'][j]}"
        output_texts.append(text)
    return output_texts

def formatting_prompts_func_hotel(example):
    path = f"prompts/hotel_0shot.txt"
    with open(path, 'r', encoding='utf-8') as fp:
        instruction = fp.read().strip() + "\n"
    output_texts = []         
    for j in range(len(example['Text'])):
        text = f"{instruction}\nText: {example['Text'][j]}\nSentiment Elements: {example['Sentiment Elements'][j]}"
        output_texts.append(text)
    return output_texts

for m in range(len(model_names)):
    for d in range(len(checkpoints[m])):
        output_dir="./"
        cur_file_path = file_path[d]
        dataset = load_dataset("json", data_files=cur_file_path)
        train_dataset = dataset['train']
        val_dataset = dataset['validation']

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(model_names[m], torch_dtype=torch.bfloat16, quantization_config=bnb_config, device_map='auto')                                                
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(base_model)

        # Change the LORA hyperparameters accordingly to fit your use case
        peft_config = LoraConfig(
            r=128,
            lora_alpha=16,
            target_modules=find_all_linear_names(base_model),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        base_model = get_peft_model(base_model, peft_config)
        print_trainable_parameters(base_model)
        
        tokenizer = AutoTokenizer.from_pretrained(model_names[m])
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
        # Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
        training_args = TrainingArguments(
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            num_train_epochs=10, 
            learning_rate=1e-4,
            bf16=True,
            save_total_limit=5,
            logging_steps=10,
            output_dir=output_dir,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_strategy="epoch",
        )
        # Change the prompt based on data domain
        if d in [0, 1, 3, 4]:
            formatting_prompts_func = formatting_prompts_func_rest
        elif d in [2, 5]:
            formatting_prompts_func = formatting_prompts_func_lap
        elif d == 6:
            formatting_prompts_func = formatting_prompts_func_book
        elif d == 7:
            formatting_prompts_func = formatting_prompts_func_cloth
        elif d == 8:
            formatting_prompts_func = formatting_prompts_func_hotel
        
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            max_seq_length=2048,
            formatting_func=formatting_prompts_func,
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
            
        trainer.train() 
        trainer.save_model(output_dir)

        output_dir = os.path.join(output_dir, checkpoints[m][d])
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir) 
        torch.cuda.empty_cache()  