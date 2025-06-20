set -ex

export CUDA_VISIBLE_DEVICES=0

declare -A TASK_DATA
TASK_DATA[asqp]="R15 R16"
TASK_DATA[acos]="Lap Rest"
TASK_DATA[memd]="M-Rest M-Lap Books Clothing Hotel"

cd src

# for SVP_TYPE in heuristic rand rank 
for TASK in asqp acos memd
do
for DATA in ${TASK_DATA[${TASK}]}
do
for SEED in 5 10 15 20 25
do
OUT_DIR="../outputs/$TASK/${DATA}/top${K}_${CTRL_TOKEN}_data${DATA_RATIO}"

mkdir -p $OUT_DIR


python main.py \
    --data_path "../data/" \
    --dataset $DATA \
    --model_name t5-base \
    --output_dir $OUT_DIR \
    --num_train_epochs 30 \
    --save_top_k 0 \
    --task $TASK \
    --first_stage_views exclude_O \
    --seed $SEED \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lowercase \
    --sort_label \
    --check_val_every_n_epoch 10  \
    --eval_batch_size 64 \
    --constrained_decode \
    --do_train \
    | tee ${OUT_DIR}/train.log \
    2> ${OUT_DIR}/train.err
    # --model_name_or_path "PATH TO THE CHECKPOINT" \ # configure the checkpoint path to eval

    # --load_path_cache \
    # --single_view_type $SVP_TYPE \
    # --load_ckpt_name "ckpt path" \
    # > $OUT_DIR/train.log 2>&1&
done
done
done
done
# done
