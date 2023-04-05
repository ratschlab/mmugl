#!/bin/bash

# set the paths accordingly
PROJECT="" # root directory of the project
MIMIC_DIR="" # root directory of MIMIC-III data
UMLS_GRAPH="" # file path to extracted KG, README suggests: $PROJECT/data/umls_graph.pkl
NOTES_CONCEPTS="" # file path to extracted medical concepts from clinical reports


python run_training_concepts.py \
    --name medication \
    --logdir $PROJECT/dev/runs \
    --data_dir $PROJECT/data \
    --code_mappings $PROJECT/data/code_mappings \
    --epochs 1000 \
    --num_workers 4 \
    --grad_acc 4 \
    --batch_size 16 \
    --down_batch_size 2 \
    --down_grad_acc 16 \
    --down_learning_rate 0.00005 \
    --down_es 25 \
    --embedding_dim 255 \
    --graph_num_layers 2 \
    --graph_num_filters 2 \
    --gnn_operator SAGEConv \
    --attention_heads 2 \
    --feedforward_dim 512 \
    --num_blocks 1 \
    --agg_mask_loss 0.25 \
    --verbose \
    --mlp_dim 128 \
    --mlp_num_layers 2 \
    --freeze_embeddings \
    --pretrain \
    --pretrain_es loss_total \
    --pretrain_patience 12 \
    --split_mode cgl \
    --icd_codes $PROJECT/data/gbert/dx-vocab.txt \
    --down_icd_codes $PROJECT/data/gbert/dx-vocab-multi.txt \
    --atc_codes $PROJECT/data/gbert/px-vocab-multi.txt \
    --diagnosis_csv $MIMIC_DIR/DIAGNOSES_ICD.csv \
    --prescription_csv $MIMIC_DIR/PRESCRIPTIONS.csv \
    --admission_csv $MIMIC_DIR/ADMISSIONS.csv \
    --umls_graph $UMLS_GRAPH \
    --notes_concepts $NOTES_CONCEPTS \
    --max_visit_text_length 1279 \
    --downstream_gbert \
    --gbert_visit_model avg \
    --gbert_data $PROJECT/data/gbert \
    --random_state 1111
    