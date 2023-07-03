CUDA_VISIBLE_DEVICES=0 python gen_motion_blur.py \
            --root_folder ./data_root/nuScenes \
            --dst_folder  ./save_root/motion_blur/light \
            --trans_std 0.2