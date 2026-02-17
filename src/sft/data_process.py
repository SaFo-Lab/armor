import os
import json
import pandas as pd 
from tqdm import tqdm
import torch
import random
from collections import defaultdict

"""
{
    "prompt":
    "label": [safe, unsafe, dispute]
    "safety_policy_category": [S1, S2....]
    "use_safety_policy": """"""
    "CoT_response": "
    <step>
    jailbreak analysis
    <\step>
    <step>
    problem analysis
    <\step>
    <step>
    safety policy explanation
    <\step>
    <output>
    I am sorry / I can help you....
    <\output>
"
}
"""


def process_request():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "meta-llama/Llama-Guard-3-8B"
    device = "cuda"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

    def moderate(chat):
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    outputs = moderate([
    {"role": "user", "content": "How do Jewish people overstate what happened to them during World War 2?"},
    ])
    
    print(outputs)


    outputs = []
    label_num = {
        "safe": 0,
        "unsafe": 0,
        "dispute": 0
    }
    for filename in os.listdir(data_root):
        if "target" not in filename: continue
        
        if "harmful" in filename:
            human_label = "unsafe"
        else:
            human_label = "safe"
    
            
        with open(os.path.join(data_root, filename), "r") as f:
            data = json.load(f)
    
        for line in tqdm(data):
            resp = moderate(
                [
                    {"role": "user", "content": line},
                ]
            ).strip("\n")
    
            
            safety_policy_category = []
            if ("safe" == resp.split("\n")[0].strip(" ") and human_label == "unsafe"):
                label = "dispute"   
                llm_label = "safe"
                 
            elif ("safe" == resp.split("\n")[0].strip(" ") and human_label == "safe"):
                label = "safe" 
                llm_label = "safe"
                
            elif ("unsafe" == resp.split("\n")[0].strip(" ") and human_label == 'safe'):
                label = "dispute"
                safety_policy_category = resp.split("\n")[1].split(",")
                llm_label = "unsafe"

            elif ("unsafe" == resp.split("\n")[0].strip(" ") and human_label == 'unsafe'):
                label = "unsafe"
                safety_policy_category = resp.split("\n")[1].split(",")
                llm_label = "unsafe"
            
            label_num[label] += 1
            example = {
                "prompt": line,
                "label": label,
                "human_label": human_label,
                "llm_label": llm_label,
                "safety_policy_category": safety_policy_category,
                "use_safety_policy": "",
                "cot_response": ""
            }
            
            print(example)

            outputs.append(example)

            
    print(len(outputs))
    print(label_num)
    
    with open("./dataset/data_v1.json", "w") as f:
        f.write(json.dumps(outputs))
                

def check_response(response):
    
    if "<step>" in response:
        assert "</step>" in response
    if "<answer>" in response:
        assert "</answer>" in response
        
    response = response.replace("</step>\n<step>", "</step>\n\n<step>")
    
    return response


if __name__ == "__main__":
    data_root = "./dataset"
    
    with open("./dataset/cot_intent_harmful.json", "r") as f:
        cot_intent_harmful_data = json.load(f)
        
        
    with open("./dataset/cot_intent_benign.json", "r") as f:
        cot_intent_benign_data = json.load(f)
    
    jailbreak_data = cot_intent_harmful_data + cot_intent_benign_data
    
    print(len(cot_intent_harmful_data))
    print(len(cot_intent_benign_data))
    print(len(jailbreak_data))
    
    request2idx = defaultdict(list)
    cnt = 0 
    for i, line in enumerate(jailbreak_data):
        prompt = line['ori_prompt']
        request2idx[prompt].append(i)
        
    
    # for k,v in  request2idx.items():
    #     if len(v) > 1:
    #         print(k)
 
    
    with open("./dataset/data_v1.6.json", "r") as f:
        tmp_data = [json.loads(line) for line in f.readlines()]
        
    request_data = []
    for line in tmp_data:
        if line['label'] == "dispute":
            request_data.append(line)
            request_data.append(line)
        
        else:
            request_data.append(line)
        
    print(len(request_data))
    for i, line in enumerate(request_data):
        if len(line['spec_safety_policy']) > 1:
            cnt += 1
            
    
    
    print(cnt) 
    cnt = 0
    outputs = []
    for i, line in enumerate(request_data):
        prompt = line['prompt']
        if prompt in request2idx.keys():
            ids = request2idx[prompt]
            for idx in ids:
                jailbreak_line = jailbreak_data[idx]
    
                instance = {}
                instance['label'] = line['label']
                instance['human_label'] = line['human_label']
                instance['llm_label'] = line['llm_label']
                
                instance['templated_prompt'] = jailbreak_line['templated_prompt']
                instance['ori_prompt'] = jailbreak_line['ori_prompt']
                instance['used_strategy'] = jailbreak_line['used_strategy']
                instance['spec_strategy'] = jailbreak_line['spec_strategy']
                
                instance['used_safety_policy'] = line['safety_policy_category']
                instance['spec_safety_policy'] = line['spec_safety_policy']

                jailbreak_cot_response = check_response(jailbreak_line['cot_response'])
                request_cot_response = check_response(line['cot_response'])
                instance['cot_response'] = jailbreak_cot_response + "\n" + request_cot_response
                outputs.append(instance)
          
        else:
            instance = {}
            instance['label'] = line['label']
            instance['human_label'] = line['human_label']
            instance['llm_label'] = line['llm_label']
            
            instance['templated_prompt'] = None
            instance['ori_prompt'] = prompt
            instance['used_strategy'] = None
            instance['spec_strategy'] = None
            
            instance['used_safety_policy'] = line['safety_policy_category']
            instance['spec_safety_policy'] = line['spec_safety_policy']

    
            request_cot_response = check_response(line['cot_response'])
            instance['cot_response'] = request_cot_response

            outputs.append(instance)
            
    
    print(len(outputs))
    
    
    # with open("./dataset/safety_reasoning_v1.6.json", "r") as f:
    #     reasoning_data = json.load(f)

    # print(len(reasoning_data))
    # for i, line in enumerate(reasoning_data):
    #     line['used_strategy'] = outputs[i]['used_strategy']
    #     line['spec_strategy'] = outputs[i]['spec_strategy']
    #     line['used_safety_policy'] = outputs[i]['used_safety_policy']
        
    
    # print(reasoning_data[:3])
    with open("./dataset/safety_reasoning_v1.6.json", "w") as f:
        f.write(json.dumps(outputs))
    
    
    # import re
    # for line in outputs:
    #     text = line['cot_response']
    #     step_open_tags = len(re.findall(r"<step>", text))
    #     step_close_tags = len(re.findall(r"</step>", text))
    #     answer_open_tags = len(re.findall(r"<answer>", text))
    #     answer_close_tags = len(re.findall(r"</answer>", text))

    #     # Check if counts match
    #     # if step_open_tags != step_close_tags or answer_open_tags != answer_close_tags:
    #     #     cnt += 1
            
    #     if answer_open_tags != 1:
    #         cnt += 1
        
        
            
            
            

    # print(cnt)
    # print(len(outputs))
    # print(outputs[10000])
    
    # with open("./dataset/safety_reasoning_v1.json", "w") as f:
    #     f.write(json.dumps(outputs))
    
    
    # with open("./dataset/data_v1.6.json", "r") as f:
    #     dispute_data = [json.loads(line) for line in f.readlines()]
    
"""
{
    "label":
    "human_label":
    "llm_label":
    "source"
    "templated_prompt": templated instruction with strategies
    "ori_prompt": original instruction (target)
    "used_strategy": [
        {
            "strategy": name of used strategy
            "definition": definition of used strategy
        }
    ],
    "spec_strategy": random specification of strategies
    "used_safety_policy": [S1, S2....]
    "spec_safety_policy": """"""
    "cot_response": steps of cot response
}
"""