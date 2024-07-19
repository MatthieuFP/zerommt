import copy
import inspect
import warnings

import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from logging import getLogger
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.generation.utils import GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput, \
                                          BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput, ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput, \
                                          BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput, GenerationMode


from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ExplicitEnum

from transformers.generation.logits_process import (
    LogitsProcessorList
)

from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from .streamers import BaseStreamer

import warnings
warnings.filterwarnings("ignore")

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]
GenerateOutput = Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, ContrastiveSearchOutput]

@torch.no_grad()
def generate_cfg(
    model = None,
    model_txt = None,
    cfg_value = None,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which had the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complement the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
            sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
            intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
            `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
            generating before other GPUs. Otherwise it'll be set to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The negative prompt needed for some processors such as CFG. The batch size must match the input batch
            size. This is an experimental feature, subject to breaking API changes in future versions.
        negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention_mask for `negative_prompt_ids`.
        kwargs (`Dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

        If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
        [`~utils.ModelOutput`] types are:

            - [`~generation.GreedySearchDecoderOnlyOutput`],
            - [`~generation.SampleDecoderOnlyOutput`],
            - [`~generation.BeamSearchDecoderOnlyOutput`],
            - [`~generation.BeamSampleDecoderOnlyOutput`]

        If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
        [`~utils.ModelOutput`] types are:

            - [`~generation.GreedySearchEncoderDecoderOutput`],
            - [`~generation.SampleEncoderDecoderOutput`],
            - [`~generation.BeamSearchEncoderDecoderOutput`],
            - [`~generation.BeamSampleEncoderDecoderOutput`]
    """

    generation_txt_config = None
    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    model._validate_model_class()
    model_txt._validate_model_class()

    # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    if generation_config is None:
        # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
        # two conditions must be met
        # 1) the generation config must have been created from the model config (`_from_model_config` field);
        # 2) the generation config must have seen no modification since its creation (the hash is the same).
        if model.generation_config._from_model_config and model.generation_config._original_object_hash == hash(
            model.generation_config
        ):
            new_generation_config = GenerationConfig.from_model_config(model.config)
            if new_generation_config != model.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. This is a"
                    " deprecated strategy to control generation and will be removed soon, in a future version."
                    " Please use and modify the model generation configuration (see"
                    " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                )
                model.generation_config = new_generation_config
        generation_config = model.generation_config

    if generation_txt_config is None:
        # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
        # two conditions must be met
        # 1) the generation config must have been created from the model config (`_from_model_config` field);
        # 2) the generation config must have seen no modification since its creation (the hash is the same).
        if model_txt.generation_config._from_model_config and model_txt.generation_config._original_object_hash == hash(
            model_txt.generation_config
        ):
            new_generation_config = GenerationConfig.from_model_config(model_txt.config)
            if new_generation_config != model_txt.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. This is a"
                    " deprecated strategy to control generation and will be removed soon, in a future version."
                    " Please use and modify the model generation configuration (see"
                    " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                )
                model_txt.generation_config = new_generation_config
        generation_txt_config = model_txt.generation_config

    generation_config = copy.deepcopy(generation_config)
    generation_txt_config = copy.deepcopy(generation_txt_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    model_txt_kwargs = generation_config.update(**kwargs)
    if "input_visual_features" in model_txt_kwargs.keys():
        _ = model_txt_kwargs.pop("input_visual_features")
    generation_config.validate()
    generation_txt_config.validate()
    model._validate_model_kwargs(model_kwargs.copy())
    model_txt._validate_model_kwargs(model_txt_kwargs.copy())

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id
        generation_txt_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    txt_inputs_tensor, model_txt_input_name, model_txt_kwargs = model_txt._prepare_model_inputs(
        inputs, generation_txt_config.bos_token_id, model_txt_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states

    model_txt_kwargs["output_attentions"] = generation_txt_config.output_attentions
    model_txt_kwargs["output_hidden_states"] = generation_txt_config.output_hidden_states
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
        model_txt_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache
        model_txt_kwargs["use_cache"] = generation_txt_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )
        model_txt_kwargs["attention_mask"] = model_txt._prepare_attention_mask_for_generation(
            txt_inputs_tensor, generation_txt_config.pad_token_id, generation_txt_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not model.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

        model_txt_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            txt_inputs_tensor, model_txt_kwargs, model_txt_input_name
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )

        txt_input_ids, model_txt_kwargs = model_txt._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_txt_input_name,
            model_kwargs=model_txt_kwargs,
            decoder_start_token_id=generation_txt_config.decoder_start_token_id,
            bos_token_id=generation_txt_config.bos_token_id,
            device=txt_inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        if not has_default_max_length and generation_config.max_length is not None:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        generation_txt_config.max_length = generation_txt_config.max_new_tokens + input_ids_length
    model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
    model_txt._validate_generated_length(generation_txt_config, input_ids_length, has_default_max_length)


    # 7. determine generation mode
    generation_mode = model._get_generation_mode(generation_config, assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if model.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{model.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. prepare stopping criteria
    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    # 10. go into different generation modes
    if generation_mode == GenerationMode.ASSISTED_GENERATION:
        if generation_config.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences has to be 1 when doing assisted generate, "
                f"but is {generation_config.num_return_sequences}."
            )
        if batch_size > 1:
            raise ValueError("assisted generate is only supported for batch_size = 1")
        if not model_kwargs["use_cache"]:
            raise ValueError("assisted generate requires `use_cache=True`")

        assistant_accepts_encoder_outputs = "encoder_outputs" in set(
            inspect.signature(assistant_model.forward).parameters.keys()
        )

        # 11. If the assistant model is an encoder-decoder, prepare its encoder outputs
        if assistant_model.config.is_encoder_decoder and "assistant_encoder_outputs" not in model_kwargs:
            assistant_model_kwargs = copy.deepcopy(model_kwargs)
            inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
                inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
            )
            assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, assistant_model_kwargs, model_input_name
            )
            model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

        if (
            not assistant_model.config.is_encoder_decoder
            and assistant_accepts_encoder_outputs
            and "encoder_outputs" in model_kwargs
        ):
            # some assistants might be assymetric (many more enc layers than dec layers)
            # encoder-decoder models that share the exact same encoder as the teacher
            # in this case the assistant only needs to load the light-weight decoder,
            # but still requires `encoder_outputs` to be passed
            model_kwargs["assistant_encoder_outputs"] = model_kwargs["encoder_outputs"]

        # 12. run assisted generate
        return model.assisted_decoding(
            input_ids,
            assistant_model=assistant_model,
            do_sample=generation_config.do_sample,
            logits_processor=logits_processor,
            logits_warper=model._get_logits_warper(generation_config) if generation_config.do_sample else None,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    if generation_mode == GenerationMode.GREEDY_SEARCH:
        # 11. run greedy search
        return model.greedy_search(
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
        if not model_kwargs["use_cache"]:
            raise ValueError("Contrastive search requires `use_cache=True`")

        return model.contrastive_search(
            input_ids,
            top_k=generation_config.top_k,
            penalty_alpha=generation_config.penalty_alpha,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            sequential=generation_config.low_memory,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.SAMPLE:
        # 11. prepare logits warper
        logits_warper = model._get_logits_warper(generation_config)

        # 12. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=model.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run sample
        return model.sample(
            input_ids,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.BEAM_SEARCH:
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )

        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=model.config.is_encoder_decoder,
            **model_kwargs,
        )

        txt_input_ids, model_txt_kwargs = model_txt._expand_inputs_for_generation(
            input_ids=txt_input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=model.config.is_encoder_decoder,
            **model_txt_kwargs,
        )

        model_kwargs["model_txt_kwargs"] = model_txt_kwargs
        # 13. run beam search
        return beam_search_cfg(
            model,
            model_txt,
            cfg_value,
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.BEAM_SAMPLE:
        # 11. prepare logits warper
        logits_warper = model._get_logits_warper(generation_config)

        # 12. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )

        # 13. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=model.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 14. run beam sample
        return beam_sample_cfg(
            model,
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            num_beam_groups=generation_config.num_beam_groups,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=model.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        return model.group_beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
        final_constraints = []
        if generation_config.constraints is not None:
            final_constraints = generation_config.constraints

        if generation_config.force_words_ids is not None:

            def typeerror():
                raise ValueError(
                    "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                    f"of positive integers, but is {generation_config.force_words_ids}."
                )

            if (
                not isinstance(generation_config.force_words_ids, list)
                or len(generation_config.force_words_ids) == 0
            ):
                typeerror()

            for word_ids in generation_config.force_words_ids:
                if isinstance(word_ids[0], list):
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any(not isinstance(token_ids, list) for token_ids in word_ids):
                        typeerror()
                    if any(
                        any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                        for token_ids in word_ids
                    ):
                        typeerror()

                    constraint = DisjunctiveConstraint(word_ids)
                else:
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                        typeerror()

                    constraint = PhrasalConstraint(word_ids)
                final_constraints.append(constraint)

        # 11. prepare beam search scorer
        constrained_beam_scorer = ConstrainedBeamSearchScorer(
            constraints=final_constraints,
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=model.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        return model.constrained_beam_search(
            input_ids,
            constrained_beam_scorer=constrained_beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )


def beam_search_cfg(
    model,
    model_txt,
    cfg_value,
    input_ids: torch.LongTensor,
   	beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
   	pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    r"""
   	Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

   	<Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
   	instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

   	Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
           	The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
           	sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
           	used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
           	used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
           	tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
           	The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
           	Whether or not to return the attentions tensors of all attention layers. See `attentions` under
           	returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
           	Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
           	for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
           	Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
           	Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
           	Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
           	an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values

    model_txt_kwargs = model_kwargs.pop("model_txt_kwargs")
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else model.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    output_scores = output_scores if output_scores is not None else model.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else model.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else model.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        model_txt_inputs = model_txt.prepare_inputs_for_generation(input_ids, **model_txt_kwargs)

        outputs = model(
           	**model_inputs,
            return_dict=True,
           	output_attentions=output_attentions,
           	output_hidden_states=output_hidden_states,
        )

        outputs_txt = model_txt(
            **model_txt_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        next_txt_token_logits = outputs_txt.logits[:, -1, :]

        # Perform CFG

        ## This is removed
        next_token_logits = next_txt_token_logits + cfg_value * (next_token_logits -  next_txt_token_logits)

        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
            next_token_scores_processed
        )

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
        n_eos_tokens = len(eos_token_id) if eos_token_id else 0
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = model._reorder_cache(model_kwargs["past_key_values"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if model.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]
