## train
# BI, scale 2, 3, 4, 8
##################################################################################################################################
# BI, scale 2
# SAN_BIX2_G10R20P48, input=48x48, output=96x96

CUDA_VISIBLE_DEVICES=0 python san_main.py --model SAN --save SAN_BIX2_G20R10P48 --scale 2 --n_resgroups 20 --n_resblocks 10 --n_feats 64  --reset --chop --save_results --print_model --patch_size 96 
# BI, scale 3
# SAN_BIX3_G10R20P48, input=48x48, output=144x144
CUDA_VISIBLE_DEVICES=0 python san_main.py --model SAN --save SAN_BIX3_G20R10P48 --scale 3 --n_resgroups 20 --n_resblocks 10 --n_feats 64  --reset --chop --save_results --print_model --patch_size 144 
# BI, scale 4
# SAN_BIX4_G10R20P48, input=48x48, output=192x192
CUDA_VISIBLE_DEVICES=0 python san_main.py --model SAN --save SAN_BIX4_G20R10P48 --scale 4 --n_resgroups 20 --n_resblocks 10 --n_feats 64  --reset --chop --save_results --print_model --patch_size 192 
# BI, scale 8
# SAN_BIX8_G10R20P48, input=48x48, output=384x384
CUDA_VISIBLE_DEVICES=0 python san_main.py --model SAN --save SAN_BIX8_G20R10P48 --scale 8 --n_resgroups 20 --n_resblocks 10 --n_feats 64  --reset --chop --save_results --print_model --patch_size 384 

