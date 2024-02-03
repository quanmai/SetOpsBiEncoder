import torch
import json
import logging
import types
import functools
import random
import os
import math
import torch.nn.functional as F
from typing import List, Union
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator
from torch import (
    nn, 
    Tensor as T,
    distributed as dist,
)
from utils.utils import (
    normalize_query, 
    set_seed,
    get_yaml_file,
    get_linear_scheduler,
    extract_query,
    extract_query_or,
)
from collate.collate_fn import collate_multi123_fn, collate_extract_query_fn
logging.disable(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="config/train_multi_dpr.yaml",
    )
    args = parser.parse_args()
    yaml_config = get_yaml_file(args.config_file)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    return args


class DualEncoder(nn.Module):
    def __init__(
        self, 
        query_encoder: nn.Module, 
        passage_encoder: nn.Module,
    ):
        super().__init__()
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
    
    def _get_representation(
        self,
        encoder: BertModel,
        ids: T,
        attn_mask: T,
        token_type_ids: T,
        representation_token_pos: int = 0,
    ) -> T:
        out = encoder(
            input_ids=ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = out.last_hidden_state
        return sequence_output[:, representation_token_pos, :]

    def forward(
        self,
        query_ids: Union[T, List[T]],
        query_attn_mask: Union[T, List[T]],
        query_token_type_ids: Union[T, List[T]],
        passage_ids: T,
        passage_attn_mask: T,
        passage_token_type_ids: T,
        representation_token_pos: int = 0,
        **kwargs,
    ):
        if kwargs['templates']:
            query_embedding = []
            qembs = [self._get_representation(
                encoder=self.query_encoder,
                ids=q_ids,
                attn_mask=q_attn_mask,
                token_type_ids=q_token_type_ids,
                representation_token_pos=representation_token_pos,
            ) for (q_ids, q_attn_mask, q_token_type_ids) in zip(query_ids, query_attn_mask, query_token_type_ids)]
            for idx, q in enumerate(qembs):
                q_emb = get_query_embedding(
                    kwargs['templates'][idx],
                    q,
                )
                query_embedding.append(q_emb)
            query_embedding = torch.stack(query_embedding, dim=0)
        else:
            query_embedding = self._get_representation(
                encoder=self.query_encoder,
                ids=query_ids,
                attn_mask=query_attn_mask,
                token_type_ids=query_token_type_ids,
                representation_token_pos=representation_token_pos,
            )

        pos_passage_embeddings = self._get_representation(
            encoder=self.passage_encoder,
            ids=passage_ids,
            attn_mask=passage_attn_mask,
            token_type_ids=passage_token_type_ids,
            representation_token_pos=representation_token_pos,
        )

        pos_passage_embedding = torch.cat([
            torch.mean(
                pos_passage_embeddings[idx: idx + 10][:], 
                dim=0, 
                keepdim=True
            ) for idx in range(0, pos_passage_embeddings.shape[0], 10)
        ], dim = 0)
        
        neg_passage_embedding = self._get_representation(
            encoder=self.passage_encoder,
            ids=kwargs['neg_passage_ids'],
            attn_mask=kwargs['neg_passage_attn_mask'],
            token_type_ids=kwargs['neg_passage_token_type_ids'],
            representation_token_pos=representation_token_pos,
        )

        passage_embedding = torch.cat((pos_passage_embedding, neg_passage_embedding), dim = 0)

        return query_embedding, passage_embedding

def get_query_embedding(
    template: str,
    qembs: List[T],
) -> T:
    if template == '_':
        return qembs[0]
    elif template == '_ or _':
        return qembs[0] + qembs[1]
    elif template == '_ that are not _':
        return qembs[0] - qembs[1]
    elif template == '_ that are also _':
        return qembs[0] * qembs[1]
    elif template == '_ or _ or _':
        return qembs[0] + qembs[1] + qembs[2]
    elif template == '_ that are also _ but not _':
        return qembs[0] * qembs[1] - qembs[2]
    elif template == '_ that are also both _ and _':
        return qembs[0] * qembs[1] * qembs[2]
    else:
        raise ValueError(f"Unknown template: {template}")

def calculate_dpr_loss(matching_score,labels):
    return F.nll_loss(input=F.log_softmax(matching_score,dim=1),target=labels)

def calculate_hit_cnt(matching_score,labels):
    _, max_ids = torch.max(matching_score,1)
    return (max_ids == labels).sum()

def calculate_average_rank(matching_score,labels):
    _,indices = torch.sort(matching_score,dim=1,descending=True)
    ranks = []
    for idx,label in enumerate(labels):
        rank = ((indices[idx] == label).nonzero()).item() + 1  ##  rank starts from 1
        ranks.append(rank)
    return ranks


class QADataset(torch.utils.data.Dataset):
    def __init__(self,file_path):
        self.data = json.load(open(file_path))
        print(f"load {len(self.data)} samples from {file_path}")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

def validate(model,dataloader,accelerator):
    model.eval()
    query_embeddings = []
    positive_doc_embeddings = []
    negative_doc_embeddings = []
    for batch in dataloader:
        with torch.no_grad():
            query_embedding,doc_embedding  = model(**batch)
        query_num,_ = query_embedding.shape
        query_embeddings.append(query_embedding.cpu())
        positive_doc_embeddings.append(doc_embedding[:query_num,:].cpu())
        negative_doc_embeddings.append(doc_embedding[query_num:,:].cpu())

    query_embeddings = torch.cat(query_embeddings,dim=0)
    doc_embeddings = torch.cat(positive_doc_embeddings+negative_doc_embeddings,dim=0)
    matching_score = torch.matmul(query_embeddings,doc_embeddings.permute(1,0)) # bs, num_pos+num_neg
    labels = torch.arange(query_embeddings.shape[0],dtype=torch.int64).to(matching_score.device)
    loss = calculate_dpr_loss(matching_score,labels=labels).item()
    ranks = calculate_average_rank(matching_score,labels=labels)
    
    if accelerator.use_distributed and accelerator.num_processes>1:
        ranks_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(ranks_from_all_gpus,ranks)
        ranks = [x for y in ranks_from_all_gpus for x in y]

        loss_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(loss_from_all_gpus,loss)
        loss = sum(loss_from_all_gpus)/len(loss_from_all_gpus)
    
    return sum(ranks)/len(ranks),loss

def main():
    args = parse_args()
    set_seed(args.seed)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='wandb',
        mixed_precision='no',
        kwargs_handlers=[kwargs]
    )

    accelerator.init_trackers(
        project_name="dpr", 
        config=args,
    )
    if accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        LOG_DIR = wandb_tracker.run.dir

    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    query_encoder = BertModel.from_pretrained(args.base_model,add_pooling_layer=False)
    passage_encoder = BertModel.from_pretrained(args.base_model,add_pooling_layer=False)
    dual_encoder = DualEncoder(query_encoder,passage_encoder)
    dual_encoder.train()

    train_dataset = QADataset(args.train_file)
    train_collate_fn = functools.partial(collate_extract_query_fn,tokenizer=tokenizer,stage='train',args=args,)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.per_device_train_batch_size,shuffle=True,collate_fn=train_collate_fn,num_workers=4,pin_memory=True)
    
    dev_dataset = QADataset(args.dev_file)
    dev_collate_fn = functools.partial(collate_extract_query_fn,tokenizer=tokenizer,stage='dev',args=args,)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,batch_size=args.per_device_eval_batch_size,shuffle=False,collate_fn=dev_collate_fn,num_workers=4,pin_memory=True)

    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in dual_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in dual_encoder.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr, eps=args.adam_eps)
    
    dual_encoder, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
        dual_encoder, optimizer, train_dataloader, dev_dataloader,
    )
    
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_train_epochs
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    EVAL_STEPS = args.val_check_interval if isinstance(args.val_check_interval,int) else int(args.val_check_interval * NUM_UPDATES_PER_EPOCH)
    lr_scheduler = get_linear_scheduler(optimizer,warmup_steps=args.warmup_steps,total_training_steps=MAX_TRAIN_STEPS)

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num dev examples = {len(dev_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    logger.info(f"  Per device eval batch size = {args.per_device_eval_batch_size}")
    completed_steps = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=100)

    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(args.seed+epoch)
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        for step,batch in enumerate(train_dataloader):
            with accelerator.accumulate(dual_encoder):
                with accelerator.autocast():
                    query_embedding,doc_embedding = dual_encoder(**batch)
                    single_device_query_num,_ = query_embedding.shape
                    single_device_doc_num,_ = doc_embedding.shape
                    if accelerator.use_distributed:
                        doc_list = [torch.zeros_like(doc_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=doc_list, tensor=doc_embedding.contiguous())
                        doc_list[dist.get_rank()] = doc_embedding
                        doc_embedding = torch.cat(doc_list, dim=0)

                        query_list = [torch.zeros_like(query_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=query_list, tensor=query_embedding.contiguous())
                        query_list[dist.get_rank()] = query_embedding
                        query_embedding = torch.cat(query_list, dim=0)

                    matching_score = torch.matmul(query_embedding,doc_embedding.permute(1,0))
                    labels = torch.cat(
                        [
                            torch.arange(single_device_query_num) + gpu_index * single_device_doc_num 
                            for gpu_index in range(accelerator.num_processes)
                        ],
                        dim=0
                    ).to(matching_score.device)
                    loss = calculate_dpr_loss(matching_score,labels=labels)

                accelerator.backward(loss)

                ## one optimization step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss:.4f}",lr=f"{lr_scheduler.get_last_lr()[0]:6f}")
                    completed_steps += 1
                    accelerator.clip_grad_norm_(dual_encoder.parameters(), args.max_grad_norm)
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                    accelerator.log({"training_loss": loss}, step=completed_steps)
                    accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=completed_steps)
                    
                    if completed_steps % EVAL_STEPS == 0:
                        avg_rank,loss = validate(dual_encoder,dev_dataloader,accelerator)
                        dual_encoder.train()
                        accelerator.log({"avg_rank": avg_rank, "loss":loss}, step=completed_steps)
                        accelerator.wait_for_everyone()
                        if accelerator.is_local_main_process:
                            unwrapped_model = accelerator.unwrap_model(dual_encoder)
                            unwrapped_model.query_encoder.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/query_encoder"))
                            tokenizer.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/query_encoder"))
                            
                            unwrapped_model.passage_encoder.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/passage_encoder"))
                            tokenizer.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/passage_encoder"))

                        accelerator.wait_for_everyone()
                
                optimizer.step()
                optimizer.zero_grad()
    
    if accelerator.is_local_main_process:wandb_tracker.finish()
    accelerator.end_training()

if __name__ == '__main__':
    main()