# adapted from https://github.com/thu-ml/STAIR

from utils import read_json, write_json
import argparse
import random
from transformers import AutoTokenizer
from utils import apply_chat_template, trajectory_to_response
from tqdm import tqdm
import queue
import threading

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--seed', type=int, default=42, help="Random seed")
    args.add_argument('--mcts_path', type=str, default="../tree_data/tree_data.json")
    args.add_argument('--traindata_type', type=str, default='dpo', help="Choose from sft and dpo")
    args.add_argument('--v_threshold', type=float, default=0.5, help="Min delta v between chosen and rejected in dpo pairs. Only activate when dpo")
    args.add_argument('--actor_data_path', type=str, default="../actor_train_data/actor_data.jsonl", help="Path to write actor traindata")
    args.add_argument('--tokenizer_path', type=str, default="../actor", help="Tokenizer path")
    args.add_argument('--max_tokens', type=int, default=4096, help="Max tokens for node to be in actor traindata")
    args.add_argument('--max_child_num', type=int, default=4, help="Max children number for each node")
    args.add_argument('--good_value_threshold', type=float, default=0.8, help="Threshold for node to be a good node.")
    args = args.parse_args()
    return args

def generate_data(mcts_data, args, save_place):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    train_data = []
    
    for d in tqdm(mcts_data):
        tags = queue.Queue()
        tags.put("0")
        prompt = d["0"]["prompt_dict"]["templated_prompt"]

        node_type_list = ["strategy", "intent", "policy", "answer"]
        while not tags.empty():
            tag = tags.get()
            responses = []
            values = []
            labels = []
            node_types = []
            node_scores = []
            pre_steps = []
            this_steps = []
            
            is_terminal = []
            visited_times = []
            for i in range(args.max_child_num):
                idx = tag+"."+str(i)
                if idx in d.keys():
                    value = d[idx]["true_value"]
                    label = d[idx]["prompt_dict"]["label"]
                    node_type = node_type_list[d[idx]["depth"]-1]
                    if node_type != "answer":
                        node_score = d[idx]["node_reward"]
                    else:
                        if label=="safe":
                            node_score = (d[idx]["reward"]["helpfulness"]+1)/2
                        else:
                            node_score = d[idx]["reward"]["safety"]
                    pre_step = "".join(d[idx]["trajectory"][:-1])
                    this_step = d[idx]["trajectory"][-1]

                    labels.append(label)
                    node_types.append(node_type)
                    node_scores.append(node_score)
                    pre_steps.append(pre_step)
                    this_steps.append(this_step)
                    
                    response = trajectory_to_response(d[idx]["trajectory"])
                    prompt_with_template = apply_chat_template(prompt, response, tokenizer)
                    tensor = tokenizer(prompt_with_template, add_special_tokens=False, return_tensors="pt")
                    length = len(tensor["input_ids"][0])
                    if length <= args.max_tokens:
                        tags.put(idx)
                        responses.append(response)
                        values.append(value)
                        is_terminal.append(d[idx]["is_terminal"])
                        visited_times.append(d[idx]["visited_times"])
            _len = len(responses)
            for i in range(_len):
                for j in range(_len):
                    if node_scores[i] - node_scores[j] > args.v_threshold and node_scores[i] >= args.good_value_threshold:
                        train_data.append({
                            "prompt": prompt,
                            "label": labels[i],
                            "type":node_types[i],
                            "pre_step":pre_steps[i],
                            "chosen": this_steps[i],
                            "rejected": this_steps[j],
                            "chosen_score": node_scores[i],
                            "rejected_score": node_scores[j],
                            "chosen_is_terminal": is_terminal[i],
                            "rejected_is_terminal": is_terminal[j],
                            "chosen_visited_times": visited_times[i],
                            "rejected_visited_times": visited_times[j]
                        })

    save_place += train_data

def main():
    args = parse_args()
    print(args)
    random.seed(args.seed)
    mcts_data = read_json(args.mcts_path)
    print("TREE_LEN:", len(mcts_data))

    actor_partial_data = [[] for _ in range(200)]
    worker_prompt_num = (len(mcts_data) - 1) // 200 + 1
    threads = []
    for i in range(200):
        mcts_data_for_worker = mcts_data[min(i*worker_prompt_num,len(mcts_data)):min((i+1)*worker_prompt_num,len(mcts_data))]
        thread = threading.Thread(target=generate_data, args=(mcts_data_for_worker, args, actor_partial_data[i]))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    actor_data = []
    for actor_partial_d in actor_partial_data:
        actor_data += actor_partial_d


    print("ACTOR_PAIRS_CNT:", len(actor_data))
    terminal_terminal_cnt = 0
    terminal_non_terminal_cnt = 0
    non_terminal_non_terminal_cnt = 0
    for d in actor_data:
        if d["chosen_is_terminal"] and d["rejected_is_terminal"]:
            terminal_terminal_cnt += 1
        elif not d["chosen_is_terminal"] and not d["rejected_is_terminal"]:
            non_terminal_non_terminal_cnt += 1
        else:
            terminal_non_terminal_cnt += 1
    print("Terminal Terminal CNT:", terminal_terminal_cnt)
    print("Terminal Non-Terminal CNT:", terminal_non_terminal_cnt)
    print("Non-Terminal Non-Terminal CNT:", non_terminal_non_terminal_cnt)

    write_json(args.actor_data_path, actor_data)

if __name__ == '__main__':
    main()
