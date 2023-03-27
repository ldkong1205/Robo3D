CUDA_VISIBLE_DEVICES=0 python gen_beam_missing.py \
            --root_folder ./data_root/SemanticKITTI/sequences \
            --dst_folder  ./save_root/incomplete_echo/light \
            --drop_ratio 0.75