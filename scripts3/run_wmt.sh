#!/usr/bin/env bash
python meta_nmt.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every 1000 \
                --data_prefix "/data1/ywang/" \
                --workspace_prefix "/data0/workspace/unitrans/" \
                --load_vocab \
                --dataset meta_europarl\
                --tensorboard \
                --batch_size 1250 \
                --inter_size 4 \
                --use_wo \
                -s ro -t en -a es pt it fr \
                --universal \
                --debug \

                #--share_universal_embedding \
                # --load_dataset \
                # --debug \
                # --dataset europarl_6k_fv \

