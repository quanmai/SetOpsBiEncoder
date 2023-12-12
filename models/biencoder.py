import torch
import torch.nn as nn
from torch import Tensor as T
import collections
from typing import List, Tuple
from data.biencoder_data import BiEncoderSample
from utils.data_utils import Tensorizer
import numpy as np
import random
import torch.nn.functional as F
from utils.model_utils import CheckpointState



BiEncoderBatch = collections.namedtuple(
    "BiEncoderBatch",
    [
        "query_ids",
        "query_segments",
        "context_ids",
        "ctx_segments",
        "is_positives",
        "hard_negatives",
        "encoder_type",
    ]
)



class BiEncoder(nn.Module):
    """
    Bi-Encoder model component. 
    """

    def __init__(
        self,
        query_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False
    ):
        super(BiEncoder, self).__init__()
        self.query_model = query_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder


    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_masks: T,
        fix_encoder: bool = True,
        representation_token_pos=0,
    ) -> (T, T, T):
        sequence_output = None
        pooler_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooler_output, hidden_states = sub_model(
                        ids,
                        segments,
                        attn_masks,
                        representation_token_pos
                    )

                if sub_model.training:
                    sequence_output.required_grad_(required_grad=True)
                    pooler_output.required_grad_(required_grad=True)
            else:
                sequence_output, pooler_output, hidden_states = sub_model(
                    ids,
                    segments,
                    attn_masks,
                    representation_token_pos
                )

        return sequence_output, pooler_output, hidden_states

    def forward(
        self,
        query_ids: T,
        query_segments: T,
        query_attn_mask: T,
        ctx_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos = 0
    ) -> (T, T):
        query_encoder = self.query_model if encoder_type is None or encoder_type=="query" else self.ctx_model
        _, query_out, _ = self.get_representation(
            query_encoder,
            query_ids,
            query_segments,
            query_attn_mask,
            self.fix_q_encoder,
            representation_token_pos
        )

        ctx_encoder = self.ctx_model if encoder_type is None or encoder_type=="ctx" else self.query_model
        _, ctx_out, _ = self.get_representation(
            ctx_encoder,
            ctx_ids,
            ctx_segments,
            ctx_attn_mask,
            self.fix_ctx_encoder,
            representation_token_pos
        )

        return query_out, ctx_out
    
    def create_biencoder_input(
        self,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int =0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        query_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctx[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.hard_negative_passages[0]
            
            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            query = sample.query

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[: num_hard_negatives]
            
            neg_ctxs = neg_ctxs[: num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[: num_hard_negatives]

            # context
            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(
                    ctx.text, 
                    title=ctx.title if (insert_title and ctx.title) else None
                ) for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx
                    )
                ]
            )

            if query_token:
                if query_token == "[START_ENT]":
                    # TODO: Revisit later
                    pass
                else:
                    query_tensors.append(
                        tensorizer.text_to_tensor(" ".join(query_token, query))
                    )
            else:
                query_tensors.append(tensorizer.text_to_tensor(query))
        
        ctx_tensor = torch.cat(
            [ctx.view(1, -1) for ctx in ctx_tensors], dim=0
        )
        query_tensor = torch.cat(
            [q.view(1, -1) for q in query_tensors], dim=0
        )

        ctx_segments = torch.zeros_like(ctx_tensor)
        query_segments = torch.zeros_like(query_tensor)

        return BiEncoderBatch(
            query_tensor,
            query_segments,
            ctx_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "query",
        )
    
    def get_state_dict(self):
        return self.state_dict()
    
    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        return self.load_state_dict(saved_state.model_dict, strict=strict)
    

class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)
        
        softmax_scores = F.log_softmax(scores, dim=-1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduce="mean",
        )

        _, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
            max_idxs == torch.tensor(positive_idx_per_question
        ).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)
        
        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vector: T) -> T:
        f = BiEncoderNllLoss.get_similarity_functioin()
        return f(q_vector, ctx_vector)
    
    @staticmethod
    def get_similarity_fuction():
        return dot_product_scores

def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vector: n2 x D -> n1 x n2
    r = torch.einsum('ij,kj->ik', q_vectors, ctx_vectors)
    return r
