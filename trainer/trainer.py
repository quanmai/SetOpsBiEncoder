from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments


def _unpack_qp(inputs):
    pass



class DualEncoderTrainer(Trainer):
    """"""

    def __init__(self, *args, **kwargs):
        super(DualEncoderTrainer).__init__(*args, **kwargs)
        self.model

    
    def compute_loss(self, model, inputs):
        query, passage = _unpack_qp(inputs)