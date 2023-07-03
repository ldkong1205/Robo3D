CUDA_VISIBLE_DEVICES=0 python gen_cross_sensor.py \
            --root_folder ./data_root/nuScenes \
            --dst_folder  ./save_root/cross_sensor/light \
            --num_beam_to_drop 8