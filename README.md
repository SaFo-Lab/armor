# [ICLR 2026] ARMOR: Aligning Safe and Secure Large Language Models via Meticulous Reasoning
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2507.11500v2) 

![Overview.](assets/overview.png)

### Abstract
Large Language Models have shown impressive generative capabilities across diverse tasks, but their safety remains a critical concern. Existing post-training alignment methods, such as SFT and RLHF, reduce harmful outputs yet leave LLMs vulnerable to jailbreak attacks, especially advanced optimization-based ones. Recent system-2 approaches enhance safety by adding inference-time reasoning, where models assess potential risks before producing responses. However, we find these methods fail against powerful out-of-distribution jailbreaks, such as AutoDAN-Turbo and Adversarial Reasoning, which conceal malicious goals behind seemingly benign prompts. We observe that all jailbreaks ultimately aim to embed a core malicious intent, suggesting that extracting this intent is key to defense. To this end, we propose ARMOR, which introduces a structured three-step reasoning pipeline: (1) analyze jailbreak strategies from an external, updatable strategy library, (2) extract the core intent, and (3) apply policy-based safety verification. We further develop ARMOR-Think, which decouples safety reasoning from general reasoning to improve both robustness and utility. Evaluations on advanced optimization-based jailbreaks and safety benchmarks show that ARMOR achieves state-of-the-art safety performance, with an average harmful rate of 0.002 and an attack success rate of 0.06 against advanced optimization-based jailbreaks, far below other reasoning-based models. Moreover, ARMOR demonstrates strong generalization to unseen jailbreak strategies, reducing their success rate to zero. These highlight ARMOR’s effectiveness in defending against OOD jailbreak attacks, offering a practical path toward secure and reliable LLMs.


### Method
![Method.](assets/method.png)
The framework of ARMOR consists of the following steps: (1) Construct the Meticulous Reasoning steps with jailbreak prompts, their coordinate ground truth (GT) jailbreak strategy and intent, and the safety policy; (2) Format the reasoning steps with inputs involving the user's prompts and the system prompt consists of a dynamic strategy library and the safety policy; (3) Train the base model to get the ARMOR model; (4) Conduct inference of ARMOR with a custom strategy library and the safety policy; (5) Conduct test-time scaling with the DPO model and PRM trained on preference data generated from grounded tree sampling.


### ARMOR Data and Model

The training data is available at [🤗ARMOR-Data](https://huggingface.co/datasets/Ethan271/SafeReasoning-Train).

The model is available at [🤗ARMOR-7b](https://huggingface.co/Ethan271/Armor-7b).


### Preparation

Prepare the general environment.

```shell
conda create -n armor python=3.12

conda activate armor

pip install -r requirements.txt
```

Download models (the base Qwen2.5 model and the armor pretrained model) and datasets (training data of armor).


### Training

Conduct SFT on the base model with the armor training data.

```shell
cd sft
```

Modify the configuration of the training, especially the path to the base model and the training data.


Then run the training script:

```shell
./scripts/finetune.sh
```

The sft model will be saved in `./outputs`.

### Sampling Preference Data
First prepare the environment for the data sampling.

```shell
cd ../tree_sample

conda create -n armor_sample python=3.12

conda activate armor_sample

pip install -r requirements.txt
```

Then follow the `README.md` to sample preference data.

### DPO
Use the sampled preference data to conduct the step-wise dpo on the model.

First prepare the environment for step-dpo.

```shell
cd ../dpo

conda create -n armor_dpo python=3.12

conda activate armor_dpo

pip install -r requirements.txt
```

Prepare the traing configeration in `./bash_scrips/qwen2_7b_step_dpo.sh`, especially the path to the base model (e.g. sft model) and the path to the preference data from the tree-based sampling.

Then run the following script to conduct the step-wise DPO:

```shell
./run_dpo.sh
```

The DPO-tuned model can then be utilized for the tree-based sampling in the next iteration and then the iterative improvement can be leveraged.

### Inference
Use the example `inference.py` for the inference:
```shell
python inference.py
```

### TO DO
- [ ] Release the code for data construction.
- [x] Release ARMOR's training data
- [x] Release ARMOR's model weights
