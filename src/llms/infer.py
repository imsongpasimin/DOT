import sys
import argparse
import time
import random

sys.path.append(".")
from data_utils import get_transformed_io
from eval_utils import extract_spans_para
from main import init_args, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import login

login(token='<Your_Hugginggface_Token>')

opinion2sentword = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}

def chat(_input, model, data):
    model_name = f"final_checkpoint_{model}_{data}"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "user", "content": _input}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Additional post-processing may be needed to convert to approprioate format.
    return response

def load_prompt(domain, prompt_type):
    if task == 'acos':
        prompt_path = f"llms/prompts/{domain}_{prompt_type}.txt"
    elif task == 'asqp':
        prompt_path = f"llms/prompts/{domain}_{prompt_type}.txt"
        
    with open(prompt_path, 'r', encoding='utf-8') as fp:
        prompt = fp.read().strip() + "\n\n"
    return prompt


def inference(args, domain, chat, model, start_idx=0, end_idx=200):
    data_path = f'{args.data_path}/{args.task}/{args.dataset}/{args.data_type}.txt'
    sources, _, targets, _ = get_transformed_io(args.data_path, args.data_name, args.data_type, args, n_tuples=0, wave=2)

    # sample `num_sample` samples from sources and targets
    samples = random.sample(list(zip(sources, targets)), args.num_sample)

    instruction = load_prompt(domain, args.prompt_type)
    prompt = f"{instruction}\n\n "
    
    for i, (source, target) in enumerate(samples):
        if i < start_idx or i > end_idx:
            continue
        print(i)
        source = " ".join(source)
        gold_list = extract_spans_para(target, 'gold')
            
        if args.task in ['asqp', 'acos', 'memd']:
            gold_list = [(at, ot, ac, opinion2sentword[sp]) for (ac, at, sp, ot) in gold_list]

        context = f"Text: {source}\n"
        context += "Sentiment Elements: "
        res = chat(prompt + context, model, args.dataset)
        # File path to storage the results
        file_path = f"llms/{model}_{args.data_type}_{args.num_sample}_{args.task}_{args.dataset}_{args.prompt_type}.txt"

        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(context + res + "\n")
            file.write(f"Gold: {gold_list}\n\n")

        print(context + res)
        print(f"Gold: {gold_list}\n")
        time.sleep(3)


if __name__ == "__main__":
    args = init_args()
    set_seed(args.seed)

    # default parameters
    args.data_type = "test"
    args.num_sample = 200
    args.prompt_type = "0shot"
    
    # Examples
    args.task = "asqp"
    args.dataset = "R15"
    inference(args, "rest", chat, llama, start_idx=0, end_idx=200)
    inference(args, "rest", chat, qwen, start_idx=0, end_idx=200)
    inference(args, "rest", chat, mistral, start_idx=0, end_idx=200)
    
    args.task = "memd"
    args.dataset = "Books"
    inference(args, "books", chat, llama, start_idx=0, end_idx=200)