CUDA_VISIBLE_DEVICES=0 python gen_beam_missing.py \
            --root_folder ./data_root/Kitti \
            --dst_folder  ./save_root/beam_missing/light \
            --num_beam_to_drop 16
