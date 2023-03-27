CUDA_VISIBLE_DEVICES=0 python gen_beam_missing.py \
            --root_folder ./data_root/SemanticKITTI/sequences \
            --dst_folder  ./save_root/fog/light \
            --num_beam_to_drop 16
