import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel
from vec2text.models.inversion_from_logits_emb import InversionFromLogitsEmbModel

LOGIT_FILTER_VALUE = -1 * 10**7

class InversionFromMultipleLogitsModel(InversionFromLogitsEmbModel):
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        input_ids = inputs.get("input_ids")
        inputs_embeds, attention_mask = self.embed_and_project(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                # decoder_attention_mask=inputs["decoder_attention_mask"],
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                **generation_kwargs,
            )

    def get_frozen_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.call_embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        next_predictions = torch.argmax(embeddings, dim=1)
        # TODO: second_input_ids are padded, should repad them?
        second_input_ids = torch.cat((input_ids, next_predictions.view(-1, 1)), dim=1)
        # TODO: check if second_attention_mask is correct
        second_attention_mask = torch.ones_like(second_input_ids)
        second_embeddings = self.call_embedding_model(
            input_ids=second_input_ids,
            attention_mask=second_attention_mask,
        )
        all_embeddings = torch.cat((embeddings, second_embeddings), dim=1)
        return all_embeddings

    def embed_and_project(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 2  # batch by d
        else:
            with torch.no_grad():
                embeddings = self.call_embedding_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                next_predictions = torch.argmax(embeddings, dim=1)
                second_input_ids = torch.cat((input_ids, next_predictions.view(-1, 1)), dim=1)
                second_attention_mask = torch.ones_like(second_input_ids)
                second_embeddings = self.call_embedding_model(
                    input_ids=second_input_ids,
                    attention_mask=second_attention_mask,
                )
                embeddings = torch.cat((embeddings, second_embeddings), dim=1)

        num_tokens = self.num_tokens
        # Remove any extraneous zeros
        embeddings = embeddings[:, : self.tokenizer_mapping.numel()]  # (B, V)

        # Map embeddings to our space.
        batch_size = embeddings.shape[0]
        new_embeddings = torch.zeros(
            (batch_size, self.encoder_decoder.config.vocab_size),
            device=embeddings.device,
            dtype=torch.double,
        )
        mapping = (
            self.tokenizer_mapping[None]
            .repeat((batch_size, 1))
            .to(new_embeddings.device)
        )
        embeddings = new_embeddings.scatter_add(
            dim=1, index=mapping, src=embeddings.to(torch.double).exp()
        ).log()
        embeddings = (
            embeddings.nan_to_num()
        )  # replace empty values from -inf to tiny neg number

        if self.training:
            unigram_batch = embeddings.mean(dim=0, keepdim=True)
            # Update unigram.
            if self.unigram.sum() == 0:
                print("INFO: resetting unigram.")
                self.unigram.data = unigram_batch
            else:
                self.unigram.data = self.unigram.data * (
                    1 - self.unigram_beta
                ) + unigram_batch * (self.unigram_beta)
        embeddings = embeddings - self.unigram
        embeddings = embeddings.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        logits_zeros = torch.zeros(
            (batch_size, self.num_zeros_to_add),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        logits = torch.cat((embeddings, logits_zeros), dim=1).to(
            self.sequence_weights.dtype
        )
        logits = logits.reshape((batch_size, num_tokens, -1))

        with torch.no_grad():
            # Minibatch
            embeddings_list = []
            i = 0
            while i < batch_size:
                batch_logits = logits[i : i + self.minibatch_size, ...]
                batch_embeddings = torch.einsum(
                    "smd,bsm -> bsd", self.word_embeddings, batch_logits
                )
                embeddings_list.append(batch_embeddings)
                i += self.minibatch_size
            embeddings = torch.cat(embeddings_list, dim=0)

        embeddings = self.embedding_proj(embeddings)
        assert embeddings.shape == (
            batch_size,
            num_tokens,
            self.encoder_hidden_dim,
        )
        attention_mask = torch.ones(
            (batch_size, num_tokens), dtype=torch.long, device=embeddings.device
        )
        return embeddings, attention_mask

        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        inputs_embeds, attention_mask = self.embed_and_project(
            input_ids=input_ids,
            attention_mask=attention_mask,
            frozen_embeddings=frozen_embeddings
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
        )