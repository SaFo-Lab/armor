import json
import random

MAX_BEGIN=1000
MAX_JAIL=1000
MAX_HARM=200
random.seed(1)

strategy_list = []
strategy_dict = {}
with open("../strategy_list.json", "r", encoding="utf-8") as f:
    strategies = json.load(f)
for reg in strategies:
    strategy_list.append(reg["strategy"])
    strategy_dict[reg["strategy"]] = reg["definition"]

def get_spec_strategy():
    spec_strategy_text = ""
    for strategy in strategy_list:
        spec_strategy_text += "{} ({})\n".format(strategy, strategy_dict[strategy])
    return spec_strategy_text

spec_strategy = get_spec_strategy()



original_path = "../download/dataset/SafeReasoning-Train.json"
with open(original_path, "r") as f:
    original_data = json.load(f)
random.shuffle(original_data)
data_for_sample = []
jail_num = 0
harm_num = 0
benign_num = 0
no_strategy = None

for reg in original_data:
    if reg['label'] == "safe" and reg['llm_label'] == "safe" and reg['human_label'] == "safe"  and benign_num<MAX_BEGIN:
        data_for_sample.append(reg)
        benign_num += 1
        
    elif reg['label'] == "unsafe" and reg['llm_label']=='unsafe' and reg['human_label'] == "unsafe":
        if reg["templated_prompt"] != reg["ori_prompt"] and jail_num<MAX_JAIL:
            data_for_sample.append(reg)
            jail_num += 1
        elif reg["templated_prompt"] == reg["ori_prompt"] and harm_num<MAX_HARM:
            data_for_sample.append(reg)
            harm_num += 1

print(len(data_for_sample))
random.shuffle(data_for_sample)
with open("./data_for_sample.json", "w") as f:
    json.dump(data_for_sample, f, ensure_ascii=False, indent=4)