export output_dir="qwen2.5-dpo"
export prompt="qwen2-boxed"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --mixed_precision bf16 \
    --num_processes 5 \
    train.py configs/config_full.yaml \
    --model_name_or_path="path-to-base-model" \
    --data_path="path-to-preference-data" \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=32 \
    --torch_dtype=bfloat16 \
    --bf16=True \
    --beta=0.5 \
    --num_train_epochs=3 \
    --save_strategy='steps' \
    --save_steps=30 \
    --save_total_limit=5 \
    --output_dir=outputs/$output_dir \
    --hub_model_id=$output_dir \
    --prompt=$prompt
