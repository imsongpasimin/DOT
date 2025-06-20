import sys
import json
import ast
sys.path.append(".")

from eval_utils import extract_spans_para, compute_f1_scores

def f(span):
    ac, at, sp, ot = extract_spans_para(span, 'test')[0]
    return (at, ot, ac, sp)

opinion2sentword = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}

def eval_log(file_path):
    """
    read the LLMs log file and compute the F1 scores
    """
    all_labels, all_preds = [], []
    cnt = 0
    with open(file_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            if line.startswith("output: "):
                cnt += 1
                line = line.split("output: ")[1].strip()
                try:
                    lst = ast.literal_eval(line)
                    print(lst)
                    pred_list = [(at.lower(), ot.lower(), ac.lower(), opinion2sentword[sp.lower()]) if sp.lower() in opinion2sentword.keys() else (at.lower(), ot.lower(), ac.lower(), sp.lower()) for (at, ac, sp, ot) in lst]
                    
                    #pred_list = [(at.lower(), ot.lower(), sp.lower()) for (at, ot, sp) in lst]
                    #pred_list = [(at.lower(), ac.lower(), sp.lower()) for (at, ac, sp) in lst]
                    # pred_list = [f(item[1]) for item in lst]
                    print(pred_list)
                except:
                    print(">>>", line)
                    print(cnt)
                    pred_list = []
                all_preds.append(pred_list)
            elif line.startswith("Gold:"):
                line = line.split("Gold:")[1].strip()
                gold_list = ast.literal_eval(line)
                print(gold_list)
                all_labels.append(gold_list)

    scores = compute_f1_scores(all_preds, all_labels)
    print("Count:", len(all_preds))
    print(scores)
    print('\n\n\n')


if __name__ == "__main__":
    txt_files = [
        "src/llms/qwen_test_200_asqp_rest15_0shot.txt",
        "src/llms/qwen_test_200_asqp_rest16_0shot.txt",
        "src/llms/qwen_test_200_acos_laptop16_0shot.txt",
        "src/llms/qwen_test_200_acos_rest16_0shot.txt",
        #"src/llms/llama3_test_200_aste_laptop14_0shot.txt",
        #"src/llms/Mistral_test_200_aste_laptop14_10shot.txt",
        #"src/llms/llama3_test_200_tasd_rest16_0shot.txt",
        #"src/llms/Mistral_test_200_tasd_rest16_10shot.txt"
    ]

    for txt_file in txt_files:
        print(txt_file)
        eval_log(txt_file)
        print()
