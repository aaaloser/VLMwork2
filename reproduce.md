# Reproduction

## W4A8
### n_samples 32
```bash
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -W ignore main_quant.py \
    --model internvl2 \
    --model_args pretrained="/data/liyifan24/VLM/models/InternVL2-8B" \
    --calib_data coco \
    --data_path "/data/liyifan24/VLM/data/calib_coco_128.json" \
    --image_folder "/data/liyifan24/VLM/data/coco/train2017" \
    --n_samples 32 \
    --method qig \
    --run_process \
    --w_bit 4 \
    --a_bit 8 \
    --reweight \
    --distort \
    --loss_mode mae \
    --scale_path "scale_cache/qig/internvl2_w4a8.pt" > nohup_log/nohup_internvl2_w4a8_n32_reweight_distort.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -W ignore main.py \
    --model internvl2 \
    --model_args pretrained="/data/liyifan24/VLM/models/InternVL2-8B" \
    --tasks mmmu \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmmu \
    --method qig \
    --pseudo_quant \
    --w_bit 4 \
    --a_bit 8 \
    --w_group 128 \
    --output_path "output" \
    --scale_path "scale_cache/qig/internvl2_w4a8.pt" > nohup_log/nohup_internvl2_w4a8_n32_reweight_distort_eval.log 2>&1 &
```

### n_samples 64
```bash
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -W ignore main_quant.py \
    --model internvl2 \
    --model_args pretrained="/data/liyifan24/VLM/models/InternVL2-8B" \
    --calib_data coco \
    --data_path "/data/liyifan24/VLM/data/calib_coco_128.json" \
    --image_folder "/data/liyifan24/VLM/data/coco/train2017" \
    --n_samples 64 \
    --method qig \
    --run_process \
    --w_bit 4 \
    --a_bit 8 \
    --reweight \
    --distort \
    --loss_mode mae \
    --scale_path "scale_cache/qig/internvl2_w4a8_n64.pt" > nohup_log/nohup_internvl2_w4a8_n64_reweight_distort.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -W ignore main.py \
    --model internvl2 \
    --model_args pretrained="/data/liyifan24/VLM/models/InternVL2-8B" \
    --tasks mmmu \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmmu \
    --method qig \
    --pseudo_quant \
    --w_bit 4 \
    --a_bit 8 \
    --w_group 128 \
    --output_path "output" \
    --scale_path "scale_cache/qig/internvl2_w4a8_n64.pt" > nohup_log/nohup_internvl2_w4a8_n64_reweight_distort_eval.log 2>&1 &
```

## W3A16
### n_samples 32
```bash
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -W ignore main_quant.py \
    --model internvl2 \
    --model_args pretrained="/data/liyifan24/VLM/models/InternVL2-8B" \
    --calib_data coco \
    --data_path "/data/liyifan24/VLM/data/calib_coco_128.json" \
    --image_folder "/data/liyifan24/VLM/data/coco/train2017" \
    --n_samples 32 \
    --method qig \
    --run_process \
    --w_bit 3 \
    --a_bit 16 \
    --reweight \
    --loss_mode mae \
    --scale_path "scale_cache/qig/internvl2_w3a16_n32.pt" > nohup_log/nohup_internvl2_w3a16_n32_reweight_distort.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -W ignore main.py \
    --model internvl2 \
    --model_args pretrained="/data/liyifan24/VLM/models/InternVL2-8B" \
    --tasks mmmu \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmmu \
    --method qig \
    --pseudo_quant \
    --w_bit 3 \
    --a_bit 16 \
    --w_group 128 \
    --output_path "output" \
    --scale_path "scale_cache/qig/internvl2_w3a16_n32.pt" > nohup_log/nohup_internvl2_w3a16_n32_reweight_eval.log 2>&1 &
```

### n_samples 64
```bash
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -W ignore main_quant.py \
    --model internvl2 \
    --model_args pretrained="/data/liyifan24/VLM/models/InternVL2-8B" \
    --calib_data coco \
    --data_path "/data/liyifan24/VLM/data/calib_coco_128.json" \
    --image_folder "/data/liyifan24/VLM/data/coco/train2017" \
    --n_samples 64 \
    --method qig \
    --run_process \
    --w_bit 3 \
    --a_bit 16 \
    --reweight \
    --loss_mode mae \
    --scale_path "scale_cache/qig/internvl2_w3a16_n64.pt" > nohup_log/nohup_internvl2_w3a16_n64_reweight_distort.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -W ignore main.py \
    --model internvl2 \
    --model_args pretrained="/data/liyifan24/VLM/models/InternVL2-8B" \
    --tasks mmmu \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmmu \
    --method qig \
    --pseudo_quant \
    --w_bit 3 \
    --a_bit 16 \
    --w_group 128 \
    --output_path "output" \
    --scale_path "scale_cache/qig/internvl2_w3a16_n64.pt" > nohup_log/nohup_internvl2_w3a16_n64_reweight_eval.log 2>&1 &
```