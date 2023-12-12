import torch
import torch.nn as nn
from encoder import DualEncoder
from config import Arguments
from transformers import AutoTokenizer, PreTrainedModel


class SetOps(nn.Module):
    """ TODO: add here
    Sample: title, text
        - document id (DID)
        - text: paragraphs -> sentence (will bind to DID)
    """

    def __init__(
        self,
        lm_q: PreTrainedModel,
        lm_p: PreTrainedModel,

        
    ) -> None:
        super().__init__()
        self.DE = DualEncoder.build(None)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_q)
        

    def forward(
        self,

    ):
        """
        """
        pass

