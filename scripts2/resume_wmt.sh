#!/usr/bin/env bash
python ez_run.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every 1000 \
                --data_prefix "/data1/ywang/" \
                --workspace_prefix "/data0/workspace/unitrans/" \
                --load_vocab \
                --dataset europarl2 \
                --tensorboard \
                --batch_size 1250 \
                --inter_size 4 \
                --use_wo \
                --language roen \
                --universal \
                --load_from "04.06_18.54.europarl2_subword_512_512_6_8_0.100_16000_" \
                --resume \
                # --debug \
                # --load_dataset \
                # --debug \


