from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput, Seq2SeqLMOutput
import torch
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import T5Tokenizer
import torch
import inspect
from torch.nn import CrossEntropyLoss
from vec2text.models.inversion_from_logits_emb import InversionFromLogitsEmbModel


class T5SparseEncoder(T5ForConditionalGeneration):
    def my_encoder(self, input_ids, output_attentions=True):
        mode = self.mode
        ss = input_ids.shape
        if mode == 'single_sentence':
            input_ids = input_ids[:, -1, :]
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_attentions=output_attentions
            )
            hidden_states = encoder_outputs[0]
        elif mode == 'full_attention':
            input_ids = input_ids.reshape((ss[0], ss[1] * ss[2]))
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_attentions=output_attentions
            )
            hidden_states = encoder_outputs[0]
        else:
            input_ids = input_ids.reshape((-1, ss[-1]))
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_attentions=output_attentions
            )
            ## take last token of the hidden state of each output
            if mode == 'last_token':
                hidden_states = encoder_outputs[0].reshape((ss[0], ss[1], ss[2], -1))
                hidden_states = hidden_states[:, :, -1, :]
            ## take average of hidden state of each output
            elif mode == 'average_pooling':
                hidden_states = encoder_outputs[0].reshape((ss[0], ss[1], ss[2], -1))
                hidden_states = torch.mean(hidden_states, dim=1)
            ## concat hidden state of each output
            elif mode == 'just_sparse':
                hidden_states = encoder_outputs[0].reshape((ss[0], ss[1] * ss[2], -1))
        attentions = None
        if output_attentions:
            attentions = encoder_outputs['attentions']
        return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=None,
                attentions=attentions
            )

    def forward(
        self,
        input_ids: Optional[List[torch.LongTensor]] = None,
        attention_mask: Optional[List[torch.FloatTensor]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5Model

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5Model.from_pretrained("t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
        >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.my_encoder(input_ids)
        hidden_states = encoder_outputs.last_hidden_state
        

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        if decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            # encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()
        # # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # # as the inputs.
        # if hasattr(self, "hf_device_map"):
        #     if hasattr(encoder, "_hf_hook"):
        #         encoder._hf_hook.io_same_device = True
        #     else:
        #         add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        hidden_states = self.my_encoder(inputs_tensor)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None
        )
        model_kwargs["encoder_outputs"] = encoder_outputs

        return model_kwargs