import logging
import hydra
import torch
import sys
import time
from omegaconf import DictConfig, OmegaConf
import os
from typing import Tuple
from torch import Tensor as T
import random
from torch import nn
from models.biencoder import BiEncoderBatch, BiEncoderNllLoss
from utils.options import (
    set_cfg_gpu,
    set_seed,
    setup_logger,
    set_cfg_params_from_state,
    get_encoder_params_state_from_cfg,
)
from utils.model_utils import (
    get_model_file,
    load_states_from_checkpoint,
    setup_for_distributed_mode,
    CheckpointState,
    get_model_obj,
    get_scheduler_linear,
    move_to_device,

)
from utils.data_utils import (
    ShardedDataIterator,
    MultiSetDataIterator,
    LocalShardedDataIterator,
    Tensorizer,
)
from utils.dist_utils import all_gather_list
import math


from utils.conf_utils import BiencoderDatasetsCfg

from models import init_biencoder_components

logger = logging.getLogger()
setup_logger(logging)


class BiEncoderTrainer(object):
    """
    """

    def __init__(self, cfg: DictConfig):
        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        # if model file is specified, encoder parameters from saved state should be used for initialization
        model_file = get_model_file(cfg, cfg.checkpoint_file_name)
        if model_file:
            saved_state = load_states_from_checkpoint(cfg, cfg.checkpoint_file_name)
            set_cfg_params_from_state(saved_state.encoder_params, cfg)

        tensorizer, model, optimizer = init_biencoder_components(cfg.encoder.encoder_model_type, cfg)

        model, optimizer = setup_for_distributed_mode(
            model,
            optimizer,
            cfg.device,
            cfg.n_gpu,
            cfg.local_rank,
            cfg.fp16,
            cfg.fp16_opt_level,
        )
        self.biencoder = model
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        self.cfg = cfg
        self.ds_cfg = BiencoderDatasetsCfg(cfg)

        if saved_state:
            self._load_saved_state(saved_state)
        
        self.dev_iterator = None

    def get_data_iterator(
        self,
        batch_size: int,
        is_train_set: bool,
        shuffle: bool = True,
        shuffle_seed: int = 0,
        offset: int = 0,
        rank: int = 0,
    ):
        hydra_datasets = self.ds_cfg.train_datasets if is_train_set else self.ds_cfg.dev_datasets
        sampling_rates = self.ds_cfg.sampling_rates

        logger.info(
            "Initializing task/set data %s",
            self.ds_cfg.train_datasets_names if is_train_set else self.ds_cfg.dev_datasets_names
        )

        single_ds_iterator_cls = LocalShardedDataIterator if self.cfg.local_shareds_dataloader else ShardedDataIterator

        sharded_iterators = [
            single_ds_iterator_cls(
                ds,
                shard_id=self.shard_id,
                num_shards=self.distributed_factor,
                batch_size=batch_size,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
                offset=offset,
            )
            for ds in hydra_datasets
        ]

        return MultiSetDataIterator(
            sharded_iterators,
            shuffle_seed=shuffle_seed,
            shuffle=shuffle,
            sampling_rate=sampling_rates if is_train_set else [1],
            rank=rank,
        )


    def run_train(self):
        cfg = self.cfg

        train_iterator = self.get_data_iterator(
            cfg.train.batch_size,
            is_train_set=True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=self.start_batch,
            rank=cfg.local_rank,
        )
        max_iterations = train_iterator.get_max_iterations()
        logger.info("Total iterations per epoch = %d", max_iterations)
        if max_iterations == 0:
            logger.warning("No data found for training")
            return

        update_per_epoch = train_iterator.max_iterations // cfg.train.gradient_accumulation_steps

        total_updates = update_per_epoch * cfg.train.num_train_epochs
        logger.info("Total updates=%d", total_updates)
        warmup_steps = cfg.train.warmup_steps

        if self.scheduler_state:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            shift = int(self.scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            scheduler = get_scheduler_linear(
                self.optimizer,
                warmup_steps,
                total_updates,
                steps_shift=shift,
            )
        else:
            scheduler = get_scheduler_linear(self.optimizer, warmup_steps, total_updates)
        
        eval_step = math.ceil(update_per_epoch / cfg.train.eval_per_epoch)
        logger.info("  Evall step=%d", eval_step)
        logger.info("***** Training *****")

        for epoch in range(self.start_epoch, int(cfg.train.num_train_epochs)):
            logger.info("***** Epoch %d*****", epoch)
            self._train_epoch(scheduler, epoch, eval_step, train_iterator)

        if cfg.local_rank in [-1, 0]:
            logger.info("Training finish. Best validation checkpoint %s", self.best_cp_name)

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        eval_step: int,
        train_data_iterator: MultiSetDataIterator
    ):
        cfg = self.cfg
        rolling_train_loss = 0.0
        epoch_loss = 0
        epoch_correct_prediction = 0

        log_results_step = cfg.train.log_batch_step
        rolling_loss_step = cfg.train.train_rolling_loss_step
        num_hard_negatives = cfg.train.hard_negatives
        num_other_negatives = cfg.train.other_negatives
        seed = cfg.seed
        self.biencoder.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0

        biencoder = get_model_obj(self.biencoder)
        dataset = 0
        for i, samples_batch in enumerate(train_data_iterator.iterate_ds_data(epoch)):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            
            ds_cfg = self.ds_cfg.train_datasets[dataset]
            special_token = ds_cfg.special_token
            encoder_type = ds_cfg.encoder_type
            shuffle_positives = ds_cfg.shuffle_positives

            data_iteration = train_data_iterator.get_iteration()
            random.seed(seed + epoch + data_iteration)

            biencoder_batch = biencoder.create_biencoder_input(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=True,
                shuffle_positives=shuffle_positives,
                query_token=special_token,
            )

            # get the token to be used for representation selection
            from utils.data_utils import DEFAULT_SELECTOR
            selector = ds_cfg.selector if ds_cfg else DEFAULT_SELECTOR
            
            rep_positions = selector.get_positions(biencoder_batch, self.tensorizer)
            loss_scale = cfg.loss_scale_factors[dataset] if cfg.loss_scale_factors else None
            loss, correct_cnt = _do_biencoder_fwd_pass(
                self.biencoder,
                biencoder_batch,
                self.tensorizer,
                cfg,
                encoder_type=encoder_type,
                rep_positions=rep_positions,
                loss_scale=loss_scale,
            )

            epoch_correct_prediction += correct_cnt
            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            if cfg.fp16:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), cfg.train_max_grad_norm
                    )
            else:
                loss.backward()
                if cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.biencoder.parameters(), cfg.train.max_grad_norm
                    )

            if (i + 1) % cfg.train_gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.biencoder.zero_grad()
            
            if i % log_results_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch: %d: Step: %d/%d, loss=%f, lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    loss.item(),
                    lr,
                )

            if (i + 1) % rolling_loss_step == 0:
                logger.info("Training batch: %d", data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info(
                    "Avg. loss per last %d batches: %f",
                    rolling_loss_step,
                    latest_rolling_train_av_loss,
                )
                rolling_train_loss = 0.0
            
            if data_iteration % eval_step == 0:
                logger.info(
                    "rank=%d, Validation: Epoch: %d Step: %d/%d",
                    cfg.local_rank,
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(epoch, train_data_iterator.get_iteration(), scheduler)
                self.biencoder.train()
        
        logger.info("Epoch finished on %d", cfg.local_rank)
        self.validate_and_save(epoch, data_iteration, scheduler)

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av loss per epoch=%f", epoch_loss)
        logger.info("epoch total correct prediction=%d", epoch_correct_prediction)

    def validate_and_save(
        self,
        epoch: int,
        iteration: int,
        scheduler,
    ):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = cfg.local_rank in [-1, 0]

        if epoch == cfg.val_av_rank_start_epoch:
            self.best_validation_result = None

        if not cfg.dev_datasets:
            validation_loss = 0
        else:
            if epoch >= cfg.val_av_rank_start_epoch:
                validation_loss = self.validate_average_rank()
            else:
                validation_loss = self.validate_nll()
        
        if save_cp:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)

            if validation_loss < (self.best_validation_result or validation_loss + 1):
                self.best_validation_result = validation_loss
                self.best_cp_name = cp_name
                logger.info("New Best validation checkpoint %s", cp_name)


    def validate_nll(self) -> float:
        logger.info("NLL validation ...")
        cfg = self.cfg
        self.biencoder.eval()

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_sizem
                is_train_set=False,
                shuffle=Falsem
                rank=cfg.local_rank,
            )
        data_iterator = self.dev_iterator

        total_loss = 0.0
        start_time = time.time()
        total_correct_predictions = 0
        num_hard_negatives = cfg.train.hard_negatives
        num_other_negatives = cfg.train.other_negatives
        log_result_step = cfg.train.log_batch_step
        batches = 0
        dataset = 0
        biencoder = get_model_obj(self.biencoder)

        for i, samples_batch in enumerate(data_iterator.iterate_ds_data):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            logger.info("Eval step: %d, rank=%s", i, cfg.local_rank)

            biencoder_input = biencoder.create_biencoder_input(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
            )

            ds_cfg = self.ds_cfg.dev_datasets[dataset]
            rep_positions = ds_cfg.selector.get_positions(
                biencoder_input,
                self.tensorizer,
            )
            encoder_type = ds_cfg.encoder_type

            loss, correct_cnt = _do_biencoder_fwd_pass(
                self.biencoder,
                biencoder_input,
                self.tensorizer,
                cfg,
                encoder_type=encoder_type,
                rep_positions=rep_positions,
            )
            total_loss += loss.item()
            total_correct_predictions += correct_cnt
            batches += 1
            if (i + 1)% log_result_step == 0:
                logger.info(
                    "Eval step: %d, used_time=%f sec., loss=%f",
                    i,
                    time.timte() - start_time,
                    loss.item(),
                )
            
        total_loss /= batches
        total_samples = batches * cfg.train.dev_batch_size * self.distributed_factor
        correct_ratio = float(total_correct_predictions / total_samples)
        logger.info(
            "NLL Validation: loss=%f. correct prediction ratio %d/%d ~ %f",
            total_loss,
            total_correct_predictions,
            total_samples,
            correct_ratio,
        )
        return total_loss

    def validate_average_rank(self) -> float:
        """
        Validates biencoder model using each question's gold passage's rank across the set of passages from the dataset.
        It generates vectors for specified amount of negative passages from each question (see --val_av_rank_xxx params)
        and stores them in RAM as well as question vectors.
        Then the similarity scores are calculted for the entire
        num_questions x (num_questions x num_passages_per_question) matrix and sorted per quesrtion.
        Each question's gold passage rank in that  sorted list of scores is averaged across all the questions.
        :return: averaged rank number
        """
        logger.info("Average rank validation ...")

        cfg = self.cfg
        self.biencoder.eval()
        distributed_factor = self.distributed_factor

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_size,
                is_train_set=False,
                shuffle=False,
                rank=cfg.local_rank,
            )
        data_iterator = self.dev_iterator

        sub_batch_size = cfg.train.val_av_rank_bsz
        sim_score_f = BiEncoderNllLoss.get_similarity_fuction()
        q_representations = []
        ctx_representations = []
        positive_idx_per_question = []

        num_hard_negatives = cfg.train.val_av_rank_hard_neg
        num_other_negatives = cfg.train.val_av_rank_other_neg

        log_result_step = cfg.train.log_batch_step
        dataset = 0
        biencoder = get_model_obj(self.biencoder)
        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            if len(q_representations) > cfg.train.val_av_rank_max_qs / distributed_factor:
                break

            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            
            biencoder_input = biencoder.creat_biencoder_input(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
            )
            total_ctxs = len(ctx_representations)
            ctxs_ids = biencoder_input.context_ids
            ctxs_segments = biencoder_input.ctx_segments
            bsz = ctxs_ids.size(0)

            ds_cfg = self.ds_cfg.dev_datasets[dataset]
            encoder_type = ds_cfg.encoder_type
            rep_positions = ds_cfg.selector.get_positions(biencoder_input.query_ids, self.tensorizer)

            # split contexts batch into sub batches 
            # since it is supposed to be too large to be processed in one batch
            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):
                q_ids, q_segments = (
                    (biencoder_input.query_ids, biencoder_input.query_segments)
                    if j == 0 else (None, None)
                )

                if j == 0 and cfg.n_gpu > 1 and q_ids.size(0) == 1:
                    # if we are in DP (but not in DDP) mode, all model input tensors should have batch size >1 or 0,
                    # otherwise the other input tensors will be split but only the first split will be called
                    continue

                ctx_ids_batch = ctxs_ids[batch_start: batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[batch_start: batch_start + sub_batch_size]

                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch)
                with torch.no_grad():
                    q_dense, ctx_dense = self.biencoder(
                        q_ids,
                        q_segments,
                        q_attn_mask,
                        ctx_ids_batch,
                        ctx_seg_batch,
                        ctx_attn_mask,
                        encoder_type=encoder_type,
                        representation_token_pos=rep_positions,
                    )
                if q_dense is not None:
                    q_representations.extend(q_dense.cpu().split(1, dim=0))
                ctx_representations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_ids = biencoder_input.is_positive
            positive_idx_per_question.extend([total_ctxs + v for v in batch_positive_ids])

            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Av.rank validation: step %d, computed ctx_vectors %d, q_vectors %d",
                    i,
                    len(ctx_representations),
                    len(q_representations),
                )
        
        ctx_representations = torch.cat(ctx_representations, dim=0)
        q_representations = torch.caat(q_representations, dim=0)

        logger.info("Av.rank validation: total q_vectors size=%s", q_representations.size())
        logger.info("Av.rank validation: total ctx_vectors size=%s", ctx_representations.size())

        q_num = q_representations.size(0)
        assert q_num == len(positive_idx_per_question)


        scores = sim_score_f(q_representations, ctx_representations)
        values, indices = torch.sort(scores, dim=1, descending=True)

        rank = 0
        for i, idx in enumerate(positive_idx_per_question):
            gold_idx = (indices[i] == idx).nonzero()
            rank += gold_idx.item()
        
        if distributed_factor > 1:
            eval_stats = all_gather_list([rank, q_num], max_size=100)
            for i, item in enumerate(eval_stats):
                remote_rank, remote_q_num = item
                if i != cfg.local_rank:
                    rank += remote_rank
                    q_num += remote_q_num
        
        av_rank = float(rank / q_num)
        logger.info("Av.rank validation: average rank %s, total questions=%d", av_rank, q_num)
        return av_rank
    
    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.biencoder)
        cp = os.path.join(cfg.output_dir, cfg.checkpoint_file_name + "." + str(epoch))
        meta_params = get_encoder_params_state_from_cfg(cfg)
        state = CheckpointState(
            model_to_save.get_state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        torch.save(state._asdict(), cp)
        logger.info("Saved checkpoint at %s", cp)
        return cp

    def _load_saved_state(self, saved_state: CheckpointState):
        """ Load saved state settings"""
        epoch = saved_state.epoch
        offset = saved_state.offset
        if offset == 0: # epoch has been complated
            epoch += 1
        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)

        if self.cfg.ignore_checkpoint_offset:
            self.start_batch = 0
            self.start_epoch = 0
        else:
            self.start_epoch = epoch
            self.start_batch = 0

        model_to_load = get_model_obj(self.biencoder)
        logger.info("Loading saved model state ...")

        model_to_load.load_state(saved_state, strict=True)
        logger.info("Saved state loaded")
        if not self.cfg.ignore_checkpoint_optimizer:
            if saved_state.optimizer_dict:
                logger.info("Using saved optimizer state")
                self.optimizer.load_state_dict(saved_state.optimizer_dict)
        
        if not self.cfg.ignore_checkpoint_lr and saved_state.scheduler_dict:
            logger.info("Using saved scheduler_state")
            self.scheduler_state = saved_state.scheduler_dict
        

def _do_biencoder_fwd_pass(
    model: nn.Module,
    input: BiEncoderBatch,
    tensorizer: Tensorizer,
    cfg,
    encoder_type: str,
    rep_positions = 0,
    loss_scale: float = None,
) -> Tuple[torch.Tensor, int]: 
    # move to device
    input = BiEncoderBatch(**move_to_device(input._asdict(), cfg.device))

    q_attn_mask = tensorizer.get_attn_mask(input.query_ids)
    ctx_attn_mask = tensorizer.get_attn_mask(input.context_ids)

    if model.training:
        model_out = model(
            input.query_ids,
            input.query_segments,
            q_attn_mask,
            input.context_ids,
            input.ctx_segments,
            ctx_attn_mask,
            encoder_type=encoder_type,
            representation_token_pos=rep_positions,
        )
    else:
        with torch.no_grad():
            model_out = model(
                input.query_ids,
                input.query_segments,
                q_attn_mask,
                input.context_ids,
                input.ctx_segments,
                ctx_attn_mask,
                encoder_type=encoder_type,
                representation_token_pos=rep_positions,
            )
    local_q_vector, local_ctx_vectors = model_out
    
    loss_function = BiEncoderNllLoss()

    loss, is_correct = _calc_loss(
        cfg,
        loss_function,
        local_q_vector,
        local_ctx_vectors,
        input.is_positives,
        input.hard_negatives,
        loss_scale=loss_scale,
    )
    is_correct = is_correct.sum().item()

    if cfg.n_gpu > 1:
        loss = loss.mean()
    if cfg.train.gradient_accumulation_steps > 1:
        loss = loss / cfg.train.gradient_accumulation_steps
    return loss, is_correct



def _calc_loss(
    cfg,
    loss_function,
    local_q_vector,
    local_ctx_vectors,
    local_positive_idxs,
    local_hard_negative_idxs: list = None,
    loss_scale: float = None,
) -> Tuple[T, bool]:
    """
    Calculate In-batch negatives schema loss and supports to run it in DDP mode 
    by exchanging the representations accross all the nodes
    """
    distributed_world_size = cfg.distributed_world_size or 1
    if distributed_world_size > 1:
        q_vector_to_send = torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach()
        ctx_vector_to_send = torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach()

        global_question_ctx_vectors = all_gather_list(
            [
                q_vector_to_send,
                ctx_vector_to_send,
                local_positive_idxs,
                local_hard_negative_idxs,
            ],
            max_size=cfg.global_loss_buf_sz,
        )

        global_q_vector = []
        global_ctxs_vector = []

        positive_idx_per_question = []
        hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx, hard_negative_idxs = item

            if i != cfg.local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vector_to_send.to(local_ctx_vectors.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in hard_negative_idxs])
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                positive_idx_per_question.extend([v + total_ctxs for v in local_positive_idxs])
                hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in local_hard_negative_idxs])
            total_ctxs += ctx_vectors.size(0)
    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negative_idxs

    loss, is_correct = loss_function.calc(
        global_q_vector,
        global_ctxs_vector,
        positive_idx_per_question,
        hard_negatives_per_question,
        loss_scale=loss_scale,
    )

    return loss, is_correct




@hydra.main(config_path="conf", config_name="biencoder_train_cfg"):
def main(cfg: DictConfig):
    if cfg.train.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, must be >= 1".format(
                cfg.train.gradient_accumulation_steps
            )
        )

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)
    
    cfg = set_cfg_gpu(cfg)
    set_seed(cfg)

    if cfg.local_rank in [-1, 0]:
        logger.info("CFG (after gpu configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

    trainer = BiEncoderTrainer(cfg)

    if cfg.train_datasets and len(cfg.train_datasets) > 0:
        trainer.run_train()
    elif cfg.model_file and cfg.dev_datasets:
        logger.info("No train files are specified. Run 2 types of validation for specified model file")
        trainer.validate_nll()
        trainer.validate_average_rank()
    else:
        logger.warning("Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do.")
    

if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args

    main()