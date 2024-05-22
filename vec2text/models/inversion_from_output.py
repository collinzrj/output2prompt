import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel
from vec2text.models.inversion_from_logits_emb import InversionFromLogitsEmbModel
from transformers import PreTrainedTokenizer, AutoTokenizer
import numpy as np

class InversionFromOutputModel(InversionFromLogitsEmbModel):
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # tokenizer = AutoTokenizer.from_pretrained('t5-base')
        # print(tokenizer.batch_decode(inputs['input_ids']))
        return self.encoder_decoder.generate(
            input_ids=inputs['input_ids'],
            attention_mask=torch.ones_like(inputs['attention_mask']),
            **generation_kwargs
        )

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        return self.encoder_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )