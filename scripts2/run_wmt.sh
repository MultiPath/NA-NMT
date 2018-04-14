#!/usr/bin/env bash
python ez_run.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every 1200 \
                --data_prefix "/data1/ywang/" \
                --workspace_prefix "/data0/workspace/unitrans2/" \
                --load_vocab \
                --dataset europarl0 \
                --tensorboard \
                --batch_size 1000 \
                --inter_size 4 \
                --use_wo \
                --language roen \
                --universal \

                #--debug \

                #--share_universal_embedding \
                # --load_dataset \
                # --debug \
                # --dataset europarl_6k_fv \

