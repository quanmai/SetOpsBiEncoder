#!/bin/bash

accelerate launch doc2emb.py \
    --pretrained_model_path ./wandb/run-20240202_023348-zvpbvh2s/files/step-6520/passage_encoder/ \
    --output_dir embedding/multi_extract_query \
    # --pretrained_model_path ./wandb/run-20240130_024727-l24bi3jv/files/step-6520/passage_encoder/ \
    # --output_dir embedding/multi_123 \
    # --pretrained_model_path ./wandb/run-20240127_005746-yl1ux05l/files/step-840/passage_encoder/ \ # multi1
    # --output_dir embedding_multi_1 \
    # --pretrained_model_path ./wandb/run-20240126_032632-pjijjzu5/files/step-840/passage_encoder/ \ # multi question
    # --output_dir embedding_multi
    # --pretrained_model_path ./wandb/run-20240124_034520-e8pvzzs8/files/step-1640/passage_encoder/ \ # single
    # --output_dir embedding \