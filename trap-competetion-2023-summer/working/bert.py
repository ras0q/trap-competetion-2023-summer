from functools import cache
from os import path
from typing import Literal

import torch
import transformers
from torch import cuda, nn


class BertSequenceVectorizer(nn.Module):
    device: Literal["cuda", "cpu"]
    model_name: Literal["bert-base-uncased"]
    tokenizer: transformers.BertTokenizer
    bert_model: transformers.BertModel
    max_len: int

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "bert-base-uncased"
        self.cache_dir = path.join(path.dirname(__file__), "..", ".cache/")
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.bert_model = transformers.BertModel.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128

    @cache
    def vectorize(self, sentence: str):
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[: self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        return tuple(inputs), tuple(masks)

    def forward(self, all_inputs: tuple[tuple[int], tuple[int]]):
        return self.forward_cache(all_inputs)

    @cache
    def forward_cache(self, all_inputs: tuple[tuple[int], tuple[int]]):
        inputs, masks = all_inputs
        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model.forward(inputs_tensor, masks_tensor)
        seq_out, _ = bert_out["last_hidden_state"], bert_out["pooler_output"]

        if cuda.is_available():
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()
