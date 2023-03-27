CUDA_VISIBLE_DEVICES=0 python augmentation.py \
            --root_folder ./data_root/SemanticKITTI/sequences \
            --dst_folder  ./save_root/wet_ground/light \
            --water_height  0.0002  \
            --noise_floor  0.3   \