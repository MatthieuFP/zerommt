# ZeroMMT

ZeroMMT is compatible with Python 3.9, we did not test any other python versions.

[Read the paper (arXiv)](https://arxiv.org/abs/2407.13579)

### Model weights
[ZeroMMT-600M](https://huggingface.co/matthieufp/ZeroMMT-600M) | [ZeroMMT-1.3B](https://huggingface.co/matthieufp/ZeroMMT-1.3B) | [ZeroMMT-3.3B](https://huggingface.co/matthieufp/ZeroMMT-3.3B)


<p align="justify"> This package is intended to perform inference with the ZeroMMT model. ZeroMMT is a zero-shot multilingual multimodal machine translation system trained only on English text-image pairs. It starts from a pretrained NLLB (more info <a href="https://github.com/facebookresearch/fairseq/tree/nllb">here</a>) and adapts it using lightweight modules (<a href="https://github.com/adapter-hub/adapters">adapters</a> & visual projector) while keeping original weights frozen during training. It is trained using visually conditioned masked language modeling and KL divergence between original MT outputs and new MMT ones. ZeroMMT is available in 3 sizes: 600M, 1.3B and 3.3B. The largest model shows state-of-the-art performances on <a href="https://github.com/MatthieuFP/CoMMuTE">CoMMuTE</a>, benchmark intended to evaluate abilities of multimodal translation systems to exploit image information to disambiguate the English sentence to be translated. ZeroMMT is multilingual and available for English-to-{Arabic,Chinese,Czech,German,French,Russian}.</p>


If you use this package or like our work, please cite:
```
@inproceedings{futeral-etal-2025-towards,
    title = "Towards Zero-Shot Multimodal Machine Translation",
    author = "Futeral, Matthieu  and
      Schmid, Cordelia  and
      Sagot, Beno{\^i}t  and
      Bawden, Rachel",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.45/",
    doi = "10.18653/v1/2025.findings-naacl.45",
    pages = "761--778",
    ISBN = "979-8-89176-195-7",
    abstract = "Current multimodal machine translation (MMT) systems rely on fully supervised data (i.e sentences with their translations and accompanying images), which is costly to collect and prevents the extension of MMT to language pairs with no such data. We propose a method to bypass the need for fully supervised data to train MMT systems, using multimodal English data only. Our method ( ZeroMMT) consists in adapting a strong text-only machine translation (MT) model by training it jointly on two objectives: visually conditioned masked language modelling and the Kullback-Leibler divergence between the original MT and new MMT outputs. We evaluate on standard MMT benchmarks and on CoMMuTE, a contrastive test set designed to evaluate how well models use images to disambiguate translations. ZeroMMT obtains disambiguation results close to state-of-the-art MMT models trained on fully supervised examples. To prove that ZeroMMT generalizes to languages with no fully supervised training data, we extend CoMMuTE to three new languages: Arabic, Russian and Chinese. We also show that we can control the trade-off between disambiguation capabilities and translation fidelity at inference time using classifier-free guidance and without any additional data. Our code, data and trained models are publicly accessible."
}
```

### Installation

```
pip install zerommt
```

### Example

**without cfg**
```
import requests
from PIL import Image
import torch
from zerommt import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(model_path="matthieufp/ZeroMMT-600M",
                     enable_cfg=False).to(device)
model.eval()

image = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000002153.jpg", stream=True
    ).raw
)

src_text = "He's got a bat in his hands."
src_lang = "eng_Latn"
tgt_lang = "fra_Latn"

# Compute cross-entropy loss given translation
tgt_text = "Il a une batte dans ses mains."

with torch.inference_mode():
    loss = model(imgs=[image],
                 src_text=[src_text],
                 src_lang=src_lang,
                 tgt_text=[tgt_text],
                 tgt_lang=tgt_lang,
                 output_loss=True)

print(loss)

# Generate translation with beam search
beam_size = 4

image2 = Image.open(
    requests.get(
        "https://zupimages.net/up/24/29/7r3s.jpg", stream=True
    ).raw
)

with torch.inference_mode():
    generated = model.generate(imgs=[image, image2],
                               src_text=[src_text, src_text],
                               src_lang=src_lang,
                               tgt_lang=tgt_lang,
                               beam_size=beam_size)

translation = model.tokenizer.batch_decode(generated, skip_special_tokens=True)
print(translation)
```

**with cfg** (WARNING: enabling cfg will require approximately twice as much memory!)

```
import requests
from PIL import Image
import torch
from zerommt import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(model_path="matthieufp/ZeroMMT-600M",
                     enable_cfg=True).to(device)
model.eval()

image = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000002153.jpg", stream=True
    ).raw
)

src_text = "He's got a bat in his hands."
src_lang = "eng_Latn"
tgt_lang = "fra_Latn"

# Compute cross-entropy loss given translation
tgt_text = "Il a une batte dans ses mains."
cfg_value = 1.25

with torch.inference_mode():
    loss = model(imgs=[image],
                 src_text=[src_text],
                 src_lang=src_lang,
                 tgt_text=[tgt_text],
                 tgt_lang=tgt_lang,
                 output_loss=True,
                 cfg_value=cfg_value)
print(loss)

# Generate translation with beam search and cfg
beam_size = 4

with torch.inference_mode():
    generated = model.generate(imgs=[image],
                               src_text=[src_text],
                               src_lang=src_lang,
                               tgt_lang=tgt_lang,
                               beam_size=beam_size,
                               cfg_value=cfg_value)
                               
translation = model.tokenizer.batch_decode(generated)[0]
print(translation)
```
