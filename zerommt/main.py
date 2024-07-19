import torch
import torch.nn as nn
from .modeling import MultimodalM2M100ForConditionalGeneration
from open_clip import create_model_from_pretrained
from transformers import M2M100Config, NllbTokenizer
from typing import Optional, List
from huggingface_hub import hf_hub_download
from PIL import Image
from .generate_utils import generate_cfg
from copy import deepcopy

lang_ids = {
    "fr": "fra_Latn",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "cs": "ces_Latn",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab"
}


class ZeroMMT(nn.Module):
    def __init__(self, mmt_model, tokenizer, vision_encoder, vision_processor, enable_cfg, device):
        super().__init__()

        self.mmt_model = mmt_model
        self.tokenizer = tokenizer

        self.vision_encoder = vision_encoder
        self.vision_processor = vision_processor

        self.device = device

        self.enable_cfg = enable_cfg
        if self.enable_cfg:
            self.model_txt = deepcopy(self.mmt_model)
            self.model_txt.model.set_active_adapters(None)

    def preprocess(self,
                   imgs: List[Image.Image],
                   src_text: List[str],
                   src_lang: str,
                   tgt_text: Optional[List[str]] = None,
                   tgt_lang: str = None,
                   ):

        processed_img = torch.stack([self.vision_processor(img).to(self.device) for img in imgs], dim=0)
        with torch.inference_mode():
            img_features = self.vision_encoder.encode_image(processed_img)

        self.tokenizer.src_lang = src_lang
        src_inps = self.tokenizer(src_text, padding=True, return_tensors="pt").to(self.device)

        if tgt_text is not None:
            self.tokenizer.src_lang = tgt_lang
            tgt_inps = self.tokenizer(tgt_text, padding=True, return_tensors="pt").input_ids.to(self.device)
            tgt_inps = torch.cat((torch.LongTensor([2] * tgt_inps.size(0)).unsqueeze(-1).to(self.device), tgt_inps),
                                 dim=-1)
        else:
            bsize = src_inps.input_ids.size(0)
            tgt_inps = torch.LongTensor([2] * bsize).unsqueeze(-1).to(self.device)

        return img_features, src_inps, tgt_inps

    def forward(self,
                imgs: List[Image.Image],
                src_text: List[str],
                src_lang: str,
                tgt_text: Optional[List[str]] = None,
                tgt_lang: Optional[str] = None,
                output_loss: bool = False,
                cfg_value: Optional[float] = None,
                ):
        if output_loss:
            assert tgt_text is not None and tgt_lang is not None, "You need to provide 'tgt_text' and 'tgt_lang' to output loss."

        img_features, src_inps, tgt_inps = self.preprocess(imgs,
                                                           src_text,
                                                           src_lang,
                                                           tgt_text,
                                                           tgt_lang)

        outputs = self.mmt_model(input_ids=src_inps.input_ids, input_visual_features=img_features,
                                 attention_mask=src_inps.attention_mask, decoder_input_ids=tgt_inps,
                                 return_dict=True)

        if self.enable_cfg and cfg_value is not None:
            outputs_text_only = self.model_txt(input_ids=src_inps.input_ids, input_visual_features=None,
                                               attention_mask=src_inps.attention_mask, decoder_input_ids=tgt_inps,
                                               return_dict=True)

            logits_text_only = outputs_text_only.logits
            outputs.logits = logits_text_only + cfg_value * (outputs.logits - logits_text_only)

        if output_loss:
            logits = outputs.logits[:, 1: -1].reshape(-1, outputs.logits.size(-1))
            labels = tgt_inps[:, 2:].reshape(-1)
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fn(logits, labels)

            return loss

        return outputs

    def generate(self,
                 imgs: List[Image.Image],
                 src_text: List[str],
                 src_lang: str,
                 tgt_lang: str,
                 length_penalty: Optional[int] = None,
                 max_len: Optional[int] = None,
                 early_stopping: Optional[bool] = True,
                 beam_size: Optional[int] = None,
                 cfg_value: Optional[float] = None,
                 ):
        # Only beam search is supported for generation

        img_features, src_inps, _ = self.preprocess(imgs,
                                                    src_text,
                                                    src_lang,
                                                    None,
                                                    tgt_lang)
        model_kwargs = {"input_visual_features": img_features}

        if cfg_value is None:
            generated = self.mmt_model.generate(input_ids=src_inps.input_ids,
                                                attention_mask=src_inps.attention_mask,
                                                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
                                                num_beams=beam_size,
                                                length_penalty=length_penalty if length_penalty is not None else self.mmt_model.generation_config.length_penalty,
                                                max_length=max_len if max_len is not None else self.mmt_model.generation_config.max_length,
                                                early_stopping=early_stopping,
                                                use_cache=True,
                                                return_dict=True,
                                                **model_kwargs)
        else:
            generated = generate_cfg(model=self.mmt_model,
                                     model_txt=self.model_txt,
                                     cfg_value=cfg_value,
                                     input_ids=src_inps.input_ids,
                                     attention_mask=src_inps.attention_mask,
                                     forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
                                     num_beams=beam_size,
                                     length_penalty=length_penalty,
                                     max_length=max_len,
                                     early_stopping=early_stopping,
                                     use_cache=True,
                                     return_dict=True,
                                     **model_kwargs)

        return generated


def create_model(model_path: str,
                 enable_cfg: bool = False,
                 cache_dir: Optional[str] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_encoder, vision_processor = create_model_from_pretrained("nllb-clip-base-siglip", "v1",
                                                                    cache_dir=cache_dir,
                                                                    device=device)

    if model_path == "matthieufp/ZeroMMT-600M":
        mt_name = "facebook/nllb-200-distilled-600M"
    elif model_path == "matthieufp/ZeroMMT-1.3B":
        mt_name = "facebook/nllb-200-1.3B"
    elif model_path == "matthieufp/ZeroMMT-3.3B":
        mt_name = "facebook/nllb-200-3.3B"
    else:
        raise ValueError(f"{model_path} is not a valid model path.")

    mt_config = M2M100Config.from_pretrained(mt_name, cache_dir=cache_dir)
    mmt_model = MultimodalM2M100ForConditionalGeneration(mt_config)
    tokenizer = NllbTokenizer.from_pretrained(mt_name, cache_dir=cache_dir)

    model = ZeroMMT(mmt_model, tokenizer, vision_encoder, vision_processor, enable_cfg, device)

    checkpoint_path = hf_hub_download(model_path, "checkpoint.pt")
    model.mmt_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()

    return model
