CUDA_VISIBLE_DEVICES=0 python gen_incomplete_echo.py \
            --root_folder ./data_root/nuScenes \
            --dst_folder  ./save_root/incomplete_echo/light \
            --drop_ratio 0.75