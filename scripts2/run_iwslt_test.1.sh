#!/usr/bin/env bash
python ez_run.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every 2000 \
                --dataset iwslt \
                --tensorboard \
                --data_prefix "/data0/data/transformer_data/" \
                --workspace_prefix "/data0/workspace/simultrans/" \
                --use_wo \
                --share_embeddings \
                --realtime \
                --traj_size 5 \
                --inter_size 5 \
                --delay_weight 0.2 \
                --batch_size 500 \
                --pretrained_from "03.10_10.13.iwslt_subword_278_507_5_2_0.079_746_uni_" \
                --load_from "03.10_10.13.iwslt_subword_278_507_5_2_0.079_746_uni_" \
                --mode train \
                --p_steps 500 \
                --q_steps 500 \
                --disable_lr_schedule \
                --debug
                # --fix_model \
                # --load_sampler_from "03.16_08.24.iwslt_subword_278_507_5_2_0.079_746_simul_.Q." \