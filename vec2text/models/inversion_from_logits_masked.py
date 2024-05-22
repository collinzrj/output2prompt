import copy
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from sentence_transformers import SentenceTransformer

from vec2text.models.config import InversionConfig
from vec2text.models.model_utils import (
    FREEZE_STRATEGIES,
    disable_dropout,
    freeze_params,
    load_embedder_and_tokenizer,
    load_encoder_decoder,
    load_tokenizer,
    mean_pool,
)
from vec2text.utils import embed_api
from torch import Tensor
import warnings

logger = logging.getLogger(__name__)


# TODO: can we make this class a HF pretrained model so it works nicely with
# .push_to_hub(), etc.?
# TODO: Need config to subclass transformers.PreTrainedModel.
class InversionMaskedLogitsModel(transformers.PreTrainedModel):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """
    config_class = InversionConfig
    def __init__(self, config: InversionConfig):
        config.is_decoder = True
        config.add_cross_attention = True
        super().__init__(config=config)
        # self.config.is_decoder = True
        # self.config.add_cross_attention = True
        masked_lm_config = transformers.RobertaConfig.from_pretrained('roberta-base')
        masked_lm_config.is_decoder = True
        masked_lm_config.add_cross_attention = True
        self.masked_lm = transformers.RobertaForMaskedLM(masked_lm_config)
        ## overwrite get_extended_attention_mask method
        self.use_logits = True

        bottleneck_dim = 1536
        self.embed_dim = 768

        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embed_dim, bottleneck_dim),
            nn.Dropout(0.0),
            nn.GELU(),  # TODO consider dropout or normalization here.
            nn.Linear(bottleneck_dim, self.embed_dim),
        )

    def embed_and_project(
        self,
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = frozen_embeddings
        # # TODO: what is unigram?
        # if self.training:
        #     # Update unigram.
        #     unigram_batch = embeddings.mean(dim=0, keepdim=True)
        #     self.unigram.data = self.unigram.data * (
        #         1 - self.unigram_beta
        #     ) + unigram_batch * (self.unigram_beta)
        # embeddings -= self.unigram

        embeddings = torch.cat([embeddings, torch.zeros((embeddings.shape[0], -embeddings.shape[1] % self.embed_dim), device=self.device)], dim=1)
        num_embeddings = embeddings.shape[1] // self.embed_dim
        embeddings = embeddings.reshape(
            (embeddings.shape[0], num_embeddings, self.embed_dim)
        )
        embeddings = embeddings.to(next(self.embedding_transform.parameters()).dtype)
        embeddings = self.embedding_transform(embeddings)
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )

        assert embeddings.shape == (
            attention_mask.shape[0],
            attention_mask.shape[1],
            self.embed_dim,
        )
        return embeddings, attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        logit_embeds, logit_attention_mask = self.embed_and_project(
            frozen_embeddings=logits,
        )
        if self.use_logits is False:
            logit_embeds = None
            logit_attention_mask = None
        return self.masked_lm(
            input_ids=input_ids,
            attention_mask=torch.ones((input_ids.shape[0], input_ids.shape[1]), device=self.device),
            labels=labels,
            encoder_hidden_states=logit_embeds,
            encoder_attention_mask=logit_attention_mask
        )


class InversionMaskedLogitsModelT5(transformers.PreTrainedModel):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """
    config_class = InversionConfig
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        t5_config = transformers.T5Config.from_pretrained('t5-base')
        self.masked_lm = transformers.T5ForConditionalGeneration(t5_config)
        self.t5_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base')

        bottleneck_dim = 1536
        self.embed_dim = 768

        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embed_dim, bottleneck_dim),
            nn.Dropout(0.0),
            nn.GELU(),  # TODO consider dropout or normalization here.
            nn.Linear(bottleneck_dim, self.embed_dim),
        )

    def embed_and_project(
        self,
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = frozen_embeddings
        # # TODO: what is unigram?
        # if self.training:
        #     # Update unigram.
        #     unigram_batch = embeddings.mean(dim=0, keepdim=True)
        #     self.unigram.data = self.unigram.data * (
        #         1 - self.unigram_beta
        #     ) + unigram_batch * (self.unigram_beta)
        # embeddings -= self.unigram

        embeddings = torch.cat([embeddings, torch.zeros((embeddings.shape[0], -embeddings.shape[1] % self.embed_dim), device=self.device)], dim=1)
        num_embeddings = embeddings.shape[1] // self.embed_dim
        embeddings = embeddings.reshape(
            (embeddings.shape[0], num_embeddings, self.embed_dim)
        )
        embeddings = embeddings.to(next(self.embedding_transform.parameters()).dtype)
        embeddings = self.embedding_transform(embeddings)
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )

        assert embeddings.shape == (
            attention_mask.shape[0],
            attention_mask.shape[1],
            self.embed_dim,
        )
        return embeddings, attention_mask

    # def generate(
    #     self,
    #     inputs: Dict[str, torch.Tensor],
    #     generation_kwargs: Dict[str, torch.Tensor],
    # ) -> torch.Tensor:
    #     generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
    #     inputs_embeds, attention_mask = self.embed_and_project(
    #         input_ids=inputs.get("input_ids"),
    #         attention_mask=inputs.get("attention_mask"),
    #         frozen_embeddings=inputs.get("frozen_embeddings"),
    #     )

    #     return self.encoder_decoder.generate(
    #         # required: input embeddings
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         # optional: input IDs (for starting generation).
    #         # typically not set unless generating prefixes for
    #         # reranking.
    #         decoder_input_ids=inputs["decoder_input_ids"],
    #         # decoder_attention_mask=inputs["decoder_attention_mask"],
    #         **generation_kwargs,
    #     )

    def forward(
        self,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        logit_embeds, logit_attention_mask = self.embed_and_project(
            frozen_embeddings=frozen_embeddings,
        )
        # is this correct?
        output = self.masked_lm(
            inputs_embeds=logit_embeds,
            attention_mask=logit_attention_mask,
            labels=labels,
        )
        # print(self.t5_tokenizer.decode(output[0]))
        return output
    

class InversionMaskedLogitsModelEncoder(transformers.PreTrainedModel):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """
    config_class = InversionConfig
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        masked_lm_config = transformers.RobertaConfig.from_pretrained('roberta-base')
        self.masked_lm = transformers.RobertaForMaskedLM(masked_lm_config)

        bottleneck_dim = 1536
        self.embed_dim = 768

        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embed_dim, bottleneck_dim),
            nn.Dropout(0.0),
            nn.GELU(),  # TODO consider dropout or normalization here.
            nn.Linear(bottleneck_dim, self.embed_dim),
        )

    def embed_and_project(
        self,
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = frozen_embeddings
        # # TODO: what is unigram?
        # if self.training:
        #     # Update unigram.
        #     unigram_batch = embeddings.mean(dim=0, keepdim=True)
        #     self.unigram.data = self.unigram.data * (
        #         1 - self.unigram_beta
        #     ) + unigram_batch * (self.unigram_beta)
        # embeddings -= self.unigram

        embeddings = torch.cat([embeddings, torch.zeros((embeddings.shape[0], -embeddings.shape[1] % self.embed_dim), device=self.device)], dim=1)
        num_embeddings = embeddings.shape[1] // self.embed_dim
        embeddings = embeddings.reshape(
            (embeddings.shape[0], num_embeddings, self.embed_dim)
        )
        embeddings = embeddings.to(next(self.embedding_transform.parameters()).dtype)
        embeddings = self.embedding_transform(embeddings)
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )

        assert embeddings.shape == (
            attention_mask.shape[0],
            attention_mask.shape[1],
            self.embed_dim,
        )
        return embeddings, attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        logit_embeds, logit_attention_mask = self.embed_and_project(
            frozen_embeddings=logits,
        )
        return self.masked_lm(
            input_ids=input_ids,
            attention_mask=torch.ones((input_ids.shape[0], input_ids.shape[1]), device=self.device),
            labels=labels,
        )