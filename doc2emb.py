import csv
import os
import json, jsonlines
import numpy as np
import torch
from tqdm import tqdm
from accelerate import PartialState
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    BertTokenizer,
    BertModel,
)
from utils.utils import normalize_document
import transformers
transformers.logging.set_verbosity_error()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--wikipedia_path", default="downloads/data/wikipedia_split/psgs_w100.tsv")
    # parser.add_argument("--num_docs", type=int, default=21015324)
    parser.add_argument("--document_path", default="data/documents.jsonl")
    parser.add_argument("--num_docs", type=int, default=325505)
    parser.add_argument("--encoding_batch_size", type=int, default=1024)
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    distributed_state = PartialState()
    device = distributed_state.device

    # Load encoder
    if args.pretrained_model_path == 'facebook/dpr-ctx_encoder-single-nq-base':
        doc_encoder = DPRContextEncoder.from_pretrained(args.pretrained_model_path)
        tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.pretrained_model_path)
    else:
        doc_encoder = BertModel.from_pretrained(args.pretrained_model_path, add_pooling_layer=False)
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    doc_encoder.eval()
    doc_encoder.to(device)

    # Load document passages
    progress_bar = tqdm(
        total=args.num_docs, 
        disable=not distributed_state.is_main_process, 
        ncols=100, 
        desc='loading document passages...'
    )

    documents = []
    with jsonlines.open(args.document_path) as reader:
        for line in reader:
            documents.append([line['title'], normalize_document(line['text'])])
            progress_bar.update(1)
    

    with distributed_state.split_between_processes(documents) as sharded_documents:
        sharded_documents = [
            sharded_documents[idx:idx+args.encoding_batch_size]
            for idx in range(0, len(sharded_documents), args.encoding_batch_size)
        ]
        encoding_progress_bar = tqdm(
            total=len(sharded_documents), 
            disable=not distributed_state.is_main_process, 
            ncols=100, 
            desc='encoding test data...'
        )
        doc_embeddings = []
        for doc in sharded_documents:
            title = [x[0] for x in doc]
            passage = [(x[1]) for x in doc]
            model_input = tokenizer(
                title, 
                passage, 
                max_length=256, 
                padding='max_length', 
                return_tensors='pt', 
                truncation=True
            ).to(device)
            with torch.no_grad():
                if isinstance(doc_encoder, BertModel):
                    CLS_POS = 0
                    output = doc_encoder(**model_input).last_hidden_state[:, CLS_POS, :].cpu().numpy()
                else:
                    output = doc_encoder(**model_input).pooler_output.cpu().numpy()
            doc_embeddings.append(output)
            encoding_progress_bar.update(1)
        doc_embeddings = np.concatenate(doc_embeddings, axis=0)
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(f'{args.output_dir}/documents_shard_{distributed_state.process_index}.npy', doc_embeddings)
