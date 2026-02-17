DS_SKIP_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" nohup accelerate launch \
    --config_file config/accelerate_config.yaml \
    finetune.py  >  "train.out" 2>&1 &