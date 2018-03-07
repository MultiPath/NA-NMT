#!/usr/bin/env bash
python ez_run.py \
                --prefix [time] \
                --gpu 1 \
                --eval-every 500 \
                --dataset iwslt \
                --tensorboard \
                --data_prefix "/data0/data/transformer_data/" \
                --use_wo \
                --share_embeddings \
                --realtime \
                --pretrained_from "02.25_08.34.iwslt_subword_278_507_5_2_0.079_746_uni_" \
                --debug
                #--params base-t2t \
                #--fast --use_alignment --diag --positional_attention \
