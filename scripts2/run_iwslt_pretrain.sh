#!/usr/bin/env bash
python ez_run.py \
                --prefix [time] \
                --gpu 2 \
                --eval-every 500 \
                --dataset iwslt \
                --tensorboard \
                --data_prefix "/data0/data/transformer_data/" \
                --workspace_prefix "/data0/workspace/simultrans/" \
                --use_wo \
                --share_embeddings \
                --causal_enc \
                #--debug
                #--params base-t2t \
                #--fast --use_alignment --diag --positional_attention \
