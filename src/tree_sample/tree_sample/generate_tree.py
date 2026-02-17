# adapted from https://github.com/thu-ml/STAIR

from tree_sample.tree import tree
from tree_sample.node import node
import argparse
from omegaconf import OmegaConf
from config import TSBaseConfig
import torch
import random
from utils import read_json, write_json, apply_chat_template, load_actor_model
from diskcache import Cache
import threading
import os
import logging



def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="../config/tree_generate_test.yaml")
    args = args.parse_args()
    return args

def generate_action(actor, tokenizer, node:node, config):
    terminators = [
        tokenizer.eos_token_id
    ] + [tokenizer.convert_tokens_to_ids(stop_token) for stop_token in config.stop_tokens]

    prompt = apply_chat_template(node.prompt_dict["templated_prompt"] , node.trajectory, tokenizer)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(actor.device)

    with torch.no_grad():
        outputs = actor.generate(
            input_ids,
            max_new_tokens=config.max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        ).cpu()
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=False)


def build_tree(node,config,mct, actor_model,actor_tokenizer):
    if node.depth >= 3:
        children = []
        for _ in range(config.generate_samples_number_list[node.depth]):
            action = generate_action(actor_model, actor_tokenizer, node, config)
            if action not in children:
                mct.add_node(node, action)
                children.append(action)
        return
    else:
        children = []
        scores = []
        for _ in range(config.generate_samples_number_list[node.depth]):
            action = generate_action(actor_model, actor_tokenizer, node, config)
            if action not in children:
                mct.add_node(node, action)
                children.append(action)
        for child in node.get_children():
            scores.append(child.get_node_reward())
        max_score = max(scores)
        min_score = min(scores)
        max_child = None
        min_child = None
        for child in node.get_children():
            if  child.get_node_reward() == max_score:
                max_child = child
            if  child.get_node_reward() == min_score:
                min_child = child
    build_tree(max_child,config,mct,actor_model,actor_tokenizer)
    build_tree(min_child,config,mct,actor_model,actor_tokenizer)

def thread_function(prompts_data, config, worker_order):
    logging.info("THREAD " + str(worker_order) +" BEGIN")


    actor_model, actor_tokenizer = load_actor_model(config.actor_model_dir)
    actor_model.eval()

    if config.use_cache:
        folder_path = os.path.join(config.cache_dir, str(worker_order))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        cache = Cache(folder_path)

    output_content = []
    for prompt_index, prompt in enumerate(prompts_data):
        if config.use_cache and prompt["templated_prompt"] in cache:
            mct_data = cache[prompt["templated_prompt"]]
            output_content.append(mct_data)
            continue

        root_node = node(None, "", 1)
        root_node.set_param(config.max_depth, prompt, 0, [])
        mct = tree(root_node)
        
        build_tree(root_node, config,mct, actor_model,actor_tokenizer)
        
        
        mct_data = mct.show_tree()
        if config.use_cache:
            cache[prompt["templated_prompt"]] = mct_data
        output_content.append(mct_data)

        write_json(os.path.join(config.output_path, str(worker_order)+".json"), output_content)
    logging.info("THREAD " + str(worker_order) +" END")
        



def main():
    args = parse_args()
    config = OmegaConf.structured(TSBaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    logging.basicConfig(filename=config.log_file, level=logging.INFO)
    logging.info("CONFIG IS:"+str(config))

    prompts_data = read_json(config.train_prompt_path)

    random.seed(0)
    random.shuffle(prompts_data)
    
    
    logging.info("PROMPT DATA LOADED")

    threads = []
    for i in range(config.worker_num):
        prompts_data_for_worker = prompts_data[min(i*config.worker_prompt_num,len(prompts_data)):min((i+1)*config.worker_prompt_num, len(prompts_data))]
        thread = threading.Thread(target=thread_function, args=(prompts_data_for_worker, config, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()
