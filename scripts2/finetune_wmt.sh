#!/usr/bin/env bash
python ez_run.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every-examples 6000 \
                --data_prefix "/data1/ywang/" \
                --workspace_prefix "/data0/workspace/unitrans/" \
                --load_vocab \
                --dataset europarl0 \
                --tensorboard \
                --batch_size 1250 \
                --inter_size 4 \
                --use_wo \
                --language roen \
                --universal \
                --load_from "04.08_20.40.europarl0_subword_512_512_6_8_0.100_16000_" \
                --resume \
                --finetune \
                --universal_option "no_update_self" \
                # --debug
                # --debug \
                # --load_dataset \
                # --debug \


