# ARMOR: Aligning Safe and Secure Large Language Models via Meticulous Reasoning
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2507.11500) 

![Overview.](assets/overview.png)

### Abstract
Large Language Models (LLMs) have demonstrated remarkable generative capabilities. However, their susceptibility to misuse has raised significant safety concerns. While post-training safety alignment methods have been widely adopted, LLMs remain vulnerable to malicious instructions that can bypass safety constraints. Recent efforts have introduced inference-time safety reasoning (system-2 alignment), where LLMs conduct a reasoning process to perform safety verification before final response. We show, however, that these checks are driven by ad-hoc reasoning that diverges from the structured human process, where they first discern a user's true intent, then evaluate the associated risk based on the true intent. Consequently, these defenses remain vulnerable to sophisticated jailbreak prompts that cloak harmful goals in seemingly benign language. To build secure and safe LLMs, we propose a reasoning-based safety alignment framework, ARMOR, that replaces the ad-hoc chains of thought reasoning process with human-aligned, structured one. At inference, ARMOR (1) detects likely jailbreak strategies, (2) extracts the user's core intent while discarding deceptive instructions, and (3) applies a policy-grounded safety analysis to the purified request. ARMOR is evaluated on adaptive jailbreak attacks and multiple safety benchmarks, and a test-time scaling is conducted to further improve its performance. Results demonstrate that ARMOR significantly enhances the robustness against state-of-the-art adaptive jailbreak attacks and outperforms recent reasoning-based aligned models across various safety benchmarks.         



### Method
![Method.](assets/method.png)
The framework of ARMOR consists of the following steps: (1) Construct the Meticulous Reasoning steps with jailbreak prompts, their coordinate ground truth (GT) jailbreak strategy and intent, and the safety policy; (2) Format the reasoning steps with inputs involving the user's prompts and the system prompt consists of a dynamic strategy library and the safety policy; (3) Train the base model to get the ARMOR model; (4) Conduct inference of ARMOR with a custom strategy library and the safety policy; (5) Conduct test-time scaling with the DPO model and PRM trained on preference data generated from grounded tree sampling.


### ARMOR Training Data

The training data is avaliable at https://huggingface.co/datasets/Ethan271/SafeReasoning-Train.

### TO DO
- [ ] Release the code for data construction.
- [x] Release ARMOR's training data
- [ ] Release ARMOR's model weights
