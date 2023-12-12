"""
Encoder model wrappers based on HuggingFace code
"""

import torch
from transformers import BertConfig, BertModel, AdamW, BertTokenizer
import torch.nn as nn
import logging
from torch import Tensor as T
from models.biencoder import BiEncoder
from typing import List
from utils.data_utils import Tensorize


logger = logging.getLogger(__name__)

def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    query_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )
    ctx_encoder = HFBertEncoder(
        cfg.encoder.pretrained_model_cfg,
        project_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    fix_ctx_encoder = cfg.encoder.fix_ctx_encoder if hasattr(cfg.encoder, "fix_ctx_encoder") else False
    biencoder = BiEncoder(
        query_encoder,
        ctx_encoder,
        fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = get_optimizer(
        biencoder,
        learning_rate=cfg.train.learning_rate,
        adam_eps=cfg.train.adam_eps,
        weight_decay=cfg.train.weight_decay,
        ) if not inference_only else None

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


def get_bert_tensorizer(cfg):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg
    tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=cfg.do_lower_case)
    if cfg.special_tokens:
        _add_special_tokens(tokenizer, cfg.special_tokens)
    
    return BertTensorizer(tokenizer, sequence_length)


def _add_special_tokens(tokenizer, special_tokens):
    logger.info("Adding special tokens %s", special_tokens)
    logger.info("Tokenizer: %s", type(tokenizer))
    special_tokens_num = len(special_tokens)

    assert special_tokens_num < 500
    unused_ids = [tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)]
    logger.info("Utilizing the following unused token ids %s", unused_ids)

    for idx, id in enumerate(unused_ids):
        old_token = "[unused{}]".format(idx)
        del tokenizer.vocab[old_token]
        new_token = special_tokens[idx]
        tokenizer.vocab[new_token] = id
        tokenizer.ids_to_tokens[id] = new_token
        logging.debug("new token %s id=%s", new_token, id)

    tokenizer.additional_special_tokens = list(special_tokens)
    logger.info("additional_special_tokens %s", tokenizer.additional_special_tokens)
    logger.info("all_special_tokens_extended: %s", tokenizer.all_special_tokens_extended)
    logger.info("additional_special_tokens_ids: %s", tokenizer.additional_special_tokens_ids)
    logger.info("all_special_tokens %s", tokenizer.all_special_tokens)


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size cannot be 0"
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim else None
        self.init_weights()


    @classmethod
    def init_encoder(
        cls, 
        cfg_name: str,
        projection_dim: int = 0, 
        dropout: float = 0.1, 
        pretrained: bool = True,
        **kwargs
    ) -> BertModel:
        logger.info("Initializing HF Bert Encoder. cfg_name=%s", cfg_name)
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout !=0 :
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        if pretrained:
            return cls.from_pretrained(cfg_name, config=cfg, projection_dim=projection_dim, **kwargs)
        return HFBertEncoder(cfg, project_dim=projection_dim)
    
    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0
    ):
        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # HF > 4.0
        sequence_output = out.last_hidden_state
        pooler_output = None
        hidden_states = out.hidden_states

        if isinstance(representation_token_pos, int):
            pooler_output = sequence_output[:, representation_token_pos, :]
        else: # treat as tensor
            bsz = sequence_output.size(0)
            assert representation_token_pos.size(0) == bsz, "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooler_output = torch.stack(
                [sequence_output[i, representation_token_pos[i, 1], :] for i in range(bsz)]
            )
        
        if self.encode_proj:
            pooler_output = self.encode_proj(pooler_output)
        return sequence_output, pooler_output, hidden_states
    

class BertTensorizer(Tensorize):
    def __init__(
        self,
        tokenizer: BertTokenizer,
        max_length: int,
        pad_to_max: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self, 
        text: str, 
        title: str = None, 
        add_special_tokens: bool = True, 
        apply_max_len: bool = True
    ):
        text = text.strip()
        if title:
            token_ids = self.tokenizer.encode(
                text,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                truncation=True,
            )
        sequence_len = self.max_length
        if self.pad_to_max and len(token_ids) < sequence_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (sequence_len - len(token_ids))
        if len(token_ids) >= sequence_len:
            token_ids = token_ids[:sequence_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id
        
        return torch.tensor(token_ids)
    
    def get_separator_id(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])
    
    def get_pad_id(self) -> T:
        return torch.tensor([self.tokenizer.pad_token_id])
    
    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")
    
    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def set_pad_to_max(self, pad: bool):
        self.pad_to_max = pad
    
    def get_token_id(self, token: str) -> int:
        """ Convert token to ids"""
        return self.tokenizer.vocab[token]
    

        


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,  
) -> torch.optim.Optimizer:
    optimizer_grouped_parameters = get_hf_model_param_grouping(model, weight_decay)
    return get_optimizer_grouped(optimizer_grouped_parameters, learning_rate, adam_eps)


def get_hf_model_param_grouping(model: nn.Module, weight_decay: float = 0.0):
    no_decay = ["bias", "LayerNorm.weight"]

    return [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


def get_optimizer_grouped(
    optimizer_grouped_parameters: List,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
) -> torch.optim.Optimizer:
    return AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    """ Get Tokenizer"""
    return BertTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)