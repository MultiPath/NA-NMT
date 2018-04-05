#!/usr/bin/env bash
python ez_run.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every 2000 \
                --data_prefix "/data1/ywang/" \
                --workspace_prefix "/data0/workspace/unitrans/" \
                --load_vocab \
                --dataset europarl1 \
                --tensorboard \
                --batch_size 3072 \
                --use_wo \
                --language roen \
                --debug \
                --universal \
                # --load_dataset \


