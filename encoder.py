import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from transformers import AutoTokenizer, PreTrainedModel, AutoModel
from config import Arguments
from typing import Dict
import os

class DualEncoder(nn.Module):
    """ Encode query and document into dense vectors """

    def __init__(
        self, 
        args: Arguments,
        lm_q: PreTrainedModel,
        lm_d: PreTrainedModel,
        tokenizier: AutoTokenizer
    ):
        super().__init__()
        self.args = args
        self.lm_q = lm_q
        self.lm_d = lm_d
        self.tokenizier = tokenizier
        self.linear = nn.Linear(self.lm_q.config.hidden_size, args.out_dim) if args.add_linear else nn.Identity()


    def forward(
        self, 
        query: Dict[str, T], 
        passage: Dict[str, T]
    ): 
        """
        """
        q_out = self._encode()
        ctx_out = None
        return q_out, ctx_out

    def _compute_score(
        self, 
        
    ):
        """
        """
        pass

    def _encode(
        self,
        encoder: PreTrainedModel,
        input_dict: dict
    ):
        """
        """
        output = encoder(input_dict)
        hidden_state = output.last_hidden_state # [B, N, D]
        embeds = hidden_state[:, 0, :]
        embeds = self.linear(embeds)
        if self.args.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)
        return embeds


    @classmethod
    def build(cls, args: Arguments):
        qry_model_path = os.path.join(args.model_path, 'query_model')
        psg_model_path = os.path.join(args.model_path, 'passage_model')
        lm_q = AutoModel.from_pretrained(qry_model_path)
        lm_p = AutoModel.from_pretrained(psg_model_path)
        return cls(args, lm_q, lm_p)
    