#!/bin/bash

python eval.py \
    --topk 100 \
    --pretrained_model_path ./wandb/run-20240202_023348-zvpbvh2s/files/step-6520/query_encoder/ \
    --embedding_dir embedding/multi_extract_query \
    --retriever "learned_query" \
    # --pretrained_model_path ./wandb/run-20240130_024727-l24bi3jv/files/step-6520/query_encoder/ \
    # --embedding_dir embedding/multi_123 \
    # --sbert \
    # --pretrained_model_path ./wandb/run-20240124_034520-e8pvzzs8/files/step-1640/query_encoder/ \
    # --embedding_dir embedding \