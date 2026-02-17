# Modified from trl/trl/trainer/dpo_trainer.py
from typing import Dict, Optional, Union

import torch
from transformers import PreTrainedModel
from trl import DPOTrainer


class StepDPOTrainer(DPOTrainer):

    def tokenize_row(self, feature, processing_class, max_prompt_length, max_completion_length, add_special_tokens, model: Optional[Union[PreTrainedModel, torch.nn.Module]] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]


        chosen_tokens = processing_class(
            chosen, truncation=True, max_length=8192, add_special_tokens=True
        )
        rejected_tokens = processing_class(
            rejected, truncation=True, max_length=8192, add_special_tokens=True
        )
        prompt_tokens = processing_class(
            prompt, truncation=True, max_length=8192, add_special_tokens=True
        )

        batch["chosen_input_ids"] = chosen_tokens["input_ids"]
        batch["rejected_input_ids"] = rejected_tokens["input_ids"]
        batch["prompt_input_ids"] = prompt_tokens["input_ids"]
        batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

        if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
            batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch["rejected_input_ids"])
            )
            batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch["chosen_input_ids"])
            )

        return batch
