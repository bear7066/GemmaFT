import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
import transformers.models.gemma3.modeling_gemma3
from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.utils import is_torchdynamo_compiling


"""
This forward function do these things: alignment, filter, shift and then calculate CE Loss
"""
def replace_forward():
    transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration.forward = (
        _gemma3_forward
    )


def _gemma3_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **lm_kwargs,
) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:

    # config 
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    is_training = token_type_ids is not None and labels is not None

    if input_ids is not None and self.config.image_token_index >= self.vocab_size:
        special_image_mask = input_ids == self.config.image_token_index
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = 0
    else:
        llm_input_ids = input_ids

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    # integrate vision features
    if pixel_values is None:
        dummy = torch.zeros([1, 3, 896, 896], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        dummy_features = self.get_image_features(dummy)
        inputs_embeds = inputs_embeds + dummy_features.mean() * 0 # padding 
        image_features = None
    else:
        image_features = self.get_image_features(pixel_values)
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
            )
        else:
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

        if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
            img_token_count = special_image_mask.sum(dim=1).sum(dim=0)[0]
            raise ValueError(
                f"圖像 token 數量不符：文字中有 {img_token_count} 個 image token，"
                f"但圖像 embedding 有 {image_features.shape[0] * image_features.shape[1]} 個 token。"
            )

        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # Causal Mask + Transformer Decoder
    causal_mask = self._update_causal_mask(
        attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
    )
    outputs = self.language_model.model(
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        **lm_kwargs,
    )

    # hiddent_state: after thinking in multi-layers, LLM output a abstract representation of image and text
    # logits: the unnormalized score of the last layer of LLM
    hidden_states = outputs[0]
    loss = None
    logits = None 

    # use lm_head to understand the abstract representation(by vector mapping)
    if labels is not None:
        logits = self.language_model.lm_head(hidden_states).float()  
        shift_logits = logits[..., :-1, :].contiguous() # shift to predict next token
        shift_labels = labels[..., 1:].contiguous() # shift to labels logits

        if attention_mask is not None:
            shift_attn = attention_mask[:, -shift_logits.shape[1]:].to(logits.device)
            shift_logits = shift_logits[shift_attn != 0].contiguous()
            shift_labels = shift_labels[shift_attn.to(shift_labels.device) != 0].contiguous()

        # use CE to calculate pred + label
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, self.config.text_config.vocab_size),
            shift_labels.view(-1).to(shift_logits.device), 
        )
    else:
        logits = self.language_model.lm_head(hidden_states) # without labels, no loss can be calculated

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Gemma3CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if image_features is not None else None,
    )
