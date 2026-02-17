This file is based on the implementation of [STAIR](https://github.com/thu-ml/STAIR).


### Run Grounded Tree Sampling

1. Replace the path to the actor model and data in `tree_generate.yaml`. The data for sampling can be randomly sampled from the training data, or any other data with ground truth labels: `python sample_data.py`.


2. Replace the key and the base of arzure api in `tree_sample/score.py`.

3. Sample data with grounded tree sampling by running `scripts/tree_sample.sh`.

4. Construct the training data by running `scripts/generate_data.sh`.




