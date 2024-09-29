# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import glob

import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import skimage.io as io
import PIL.Image
import pdb
import cog
import toml
from types import SimpleNamespace

from clip_cap_pqvae import Clip_autoencoder_cap
# from clip_cap_autoencoder import Clip_autoencoder_cap

from get_args import get_args
import torch.nn.functional as F
# import torch

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

# WEIGHTS_PATHS = {
#     "coco": "coco_weights.pt",
#     "conceptual-captions": "conceptual_weights.pt",
# }

D = torch.device
CPU = torch.device("cpu")

def load_config(path):
    with open(path, 'r') as f:
        config = toml.load(f)
    # dict to SimpleNamespace, i.e. cfg['CHANNEL'] → cfg.CHANNEL
    config = {k: SimpleNamespace(**v) if isinstance(v, dict) else v for k, v in config.items()}
    return SimpleNamespace(**config)

def scalar_quantize(tensor, levels, device):
    # flat_tensor = tensor.view(-1)
    min_val = -1
    max_val = 1
    range_val = max_val - min_val
    step = range_val / (levels-1)
    quantization_points = torch.tensor([min_val + i * step for i in range(levels)]).to(device)
    quantization_points_repeat = quantization_points.repeat(tensor.shape[0], tensor.shape[1], 1)
    quantization_points_repeat = quantization_points_repeat.view(-1,quantization_points_repeat.shape[-1])
    tensor_flat = tensor.reshape(-1).unsqueeze(1)
    diffs = torch.abs(quantization_points_repeat - tensor_flat)
    min_indices = torch.argmin(diffs, dim=1)
    quantized_tensor = quantization_points[min_indices].unsqueeze(1)
    quantized_tensor = quantized_tensor.view(tensor.shape[0], tensor.shape[1])

    # pdb.set_trace()
    quantized_tensor = tensor + (quantized_tensor - tensor).detach()
    return quantized_tensor

class Predictor(): # cog.Predictor
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda:0")
        self.clip_model, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        args = get_args()
        cfgs = load_config("cfg.toml")

        self.models = {}
        self.prefix_length = 10
        for key, weights_path in WEIGHTS_PATHS.items():
            # model = ClipCaptionModel(self.prefix_length) # original
            model =  Clip_autoencoder_cap(cfgs, args, self.device, args.prefix_length).to(self.device) # VQ-VAE
            model.load_state_dict(torch.load(weights_path, map_location=CPU))
            model = model.eval()
            model = model.to(self.device)
            self.models[key] = model

    # @cog.input("image", type=cog.Path, help="Input image")
    # @cog.input(
    #     "model",
    #     type=str,
    #     options=WEIGHTS_PATHS.keys(),
    #     default="coco",
    #     help="Model to use",
    # )
    # @cog.input(
    #     "use_beam_search",
    #     type=bool,
    #     default=False,
    #     help="Whether to apply beam search to generate the output text",
    # )

    def predict(self, image, model="coco", use_beam_search=False):
        """Run a single prediction on the model"""
        image = io.imread(image)
        model = self.models[model]
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        self.levels = 4
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(self.device, dtype=torch.float32)
            # autoencoder:
            prefix = F.normalize(prefix, p=2, dim=1).float()
            encoded_x = model.encoder(prefix)
            encoded_x = F.normalize(encoded_x, p=2, dim=1).float()
            min_val = -0.5; max_val = 0.5
            levels_tensor = torch.linspace(min_val, max_val, self.levels).to(self.device)
            quantized_tensor = levels_tensor[torch.argmin(torch.abs(encoded_x.unsqueeze(-1) - levels_tensor), dim=-1)]
            quantized_tensor = encoded_x + (quantized_tensor - encoded_x).detach()
            decoded_x = model.decoder(quantized_tensor)
            decoded_x = F.normalize(decoded_x, p=2, dim=1)
            prefix = decoded_x

            # pqvae:
            # feature_normed = F.normalize(prefix, p=2, dim=1).float()
            # encoded_x = model.encoder(feature_normed)
            # encoded_x = F.normalize(encoded_x, p=2, dim=1).float()
            # codebook_loss, z_q_reconstructed = model.quantize(encoded_x)
            # decoded_x = model.decoder(z_q_reconstructed)
            # decoded_x = F.normalize(decoded_x, p=2, dim=1)
            # prefix = decoded_x

            prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)

        if use_beam_search:
            return generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
        else:
            return generate2(model, self.tokenizer, embed=prefix_embed)


class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):
    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefix_length
            )
        else:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )


# class ClipCaptionPrefix(ClipCaptionModel):
#     def parameters(self, recurse: bool = True):
#         return self.clip_project.parameters()

#     def train(self, mode: bool = True):
#         super(ClipCaptionPrefix, self).train(mode)
#         self.gpt.eval()
#         return self


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in range(entry_count): # entry_count=1
            # pdb.set_trace()
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


if __name__ == '__main__':
    # coco dataset 是imageset，不是featureset，因此需要先经过feature提取
    # folder_path = "clipcap_models_indices"
    # pt_files = glob.glob(os.path.join(folder_path, "*.pt"))

    # all_files = os.listdir(folder_path)
    # filtered_files = [f for f in all_files if f.startswith('-00') and 'codebook_dither' in f]
    
    # for pt_file in pt_files:
    # model_path = "clipcap_models/-006-codebook.pt"
    # model_path = "clipcap_models/-008-codebook_dither_pretrain_all_train.pt"
    # model_path = "clipcap_models/-009-codebook_dither_pretrain_e_d.pt"
    # model_path = "clipcap_models/-009-codebook_dither_pretrain_all_freeze.pt"
    # model_path = "clipcap_models/-009-codebook_dither.pt"

    # for model_path in filtered_files:
    #     model_path = os.path.join(folder_path, model_path)


    # model_path = "clipcap_autoencoder/freeze_ed_new/encoder_output_dimension-12-levels-4/-009-codebook.pt"
    model_path = "clipcap_pqvae/freeze_all/16-n_e-8/-009-codebook.pt"

    WEIGHTS_PATHS = {
        "coco": model_path,
        # "conceptual-captions": "conceptual_weights.pt",
    }

    predictor = Predictor()

    predictor.setup()

    # default: 
    # model: "coco"
    # use_beam_search: False (i.e. generate2)

    current_folder = os.getcwd()
    for filename in sorted(os.listdir(current_folder)):
        if filename.startswith("COCO_val2014"):
            generated_list = predictor.predict(filename)
            print(generated_list)
            print(filename)
            

    # image_path = "COCO_val2014_000000000074.jpg"
    # generated_list = predictor.predict(image_path)
    # print(generated_list)
    # print(model_path)

    # image_path = "COCO_val2014_000000000257.jpg"
    # generated_list = predictor.predict(image_path)
    # print(generated_list)
    # print(model_path)

    # image_path = "COCO_val2014_000000000502.jpg"
    # generated_list = predictor.predict(image_path)
    # print(generated_list)
    # print(model_path)

    # image_path = "COCO_val2014_000000000164.jpg"
    # generated_list = predictor.predict(image_path)
    # print(generated_list)
    # print(model_path)

    # image_path = "COCO_val2014_000000000283.jpg"
    # generated_list = predictor.predict(image_path)
    # print(generated_list)
    # print(model_path)

    # pdb.set_trace()
        