import os
import glob
import logging
import collections
import torch
from torch.serialization import default_restore_location
from torch import nn

logger = logging.getLogger()

CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        "model_dict",
        "optimizer_dict",
        "scheduler_dict",
        "offset",
        "epoch",
        "encoder_params",
    ],
)


def setup_for_distributed_mode(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: object,
    n_gpu: int = 1,
    local_rank: int = -1,
    fp16: bool = False,
    fp16_opt_level: str = "01",
) -> (nn.Module, torch.optim.Optimizer):
    model.to(device)
    if fp16:
        try:
            import apex
            from apex import amp
            
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex to use fp16 training")
        
        mode, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if local_rank != -1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device if device else local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    return model, optimizer


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model


def get_model_file(args, file_prefix) -> str:
    if args.model_file and os.path.exists(args.model_file):
        return args.model_file

    out_checkpoint_file = glob.glob(
        os.path.join(
            args.output_dir, file_prefix + "*"
        )
    ) if args.output_dir else []
    logger.info("Checkpoint file %s", out_checkpoint_file)
    model_file = max(out_checkpoint_file, key=os.path.getctime) if len(out_checkpoint_file) > 0 else None
    return model_file


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info("Reading saved model from %s", model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    logger.info("model_state_dict_key %s", state_dict.keys())
    return CheckpointState(**state_dict)


def get_scheduler_linear(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_training_steps: int,
    steps_shift: int = 0,
    last_epoch: int = -1,
):
    """
    Create a scheduler with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def move_to_device(sample, device) -> dict:
    if len(sample) == 0:
        return {}
    
    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_device(value, device) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(t, device) for t in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_device(t, device) for t in maybe_tensor]
        else:
            return maybe_tensor
    
    return _move_to_device(sample, device)