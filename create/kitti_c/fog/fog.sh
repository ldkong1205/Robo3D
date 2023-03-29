CUDA_VISIBLE_DEVICES=0 python fog_simulation.py \
            --root_folder ./data_root/Kitti \
            --dst_folder  ./save_root/fog/light \
            --inte_folder  integral_lookup_tables_seg_light_0.008beta \
            --beta  0.008