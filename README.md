# Second-order Attention Network for Single Image Super-resolution (CVPR-2019)
[]()
"[Second-order Attention Network for Single Image Super-resolution](http://openaccess.thecvf.com/content_CVPR_2019/html/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.html)" is published on CVPR-2019.
The code is built on [RCAN(pytorch)](https://github.com/yulunzhang/RCAN) and tested on Ubuntu 16.04 (Pytorch 0.4.0)

## Main Contents
### 1. Introduction
- **Abstract:**
Recently, deep convolutional neural networks (CNNs) have been widely explored in single image super-resolution (SISR) and obtained remarkable performance. However, most of the existing CNN-based SISR methods mainly focus on wider or deeper architecture design, neglecting to explore the feature correlations of intermediate layers, hence hindering the representational power of CNNs. To address this issue, in this paper, we propose a second-order attention network (SAN) for more powerful feature expression and feature correlation learning. Specifically, a novel train- able second-order channel attention (SOCA) module is developed to adaptively rescale the channel-wise features by using second-order feature statistics for more discriminative representations. Furthermore, we present a non-locally enhanced residual group (NLRG) structure, which not only incorporates non-local operations to capture long-distance spatial contextual information, but also contains repeated local-source residual attention groups (LSRAG) to learn increasingly abstract feature representations. Experimental results demonstrate the superiority of our SAN network over state-of-the-art SISR methods in terms of both quantitative metrics and visual quality.


### 2. Train code
#### Prepare training datasets
- 1. Download the **DIV2K** dataset (900 HR images) from the link [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
- 2. Set '--dir_data' as the HR and LR image path.

#### Train the model
- You can retrain the model: 
  - 1. CD to 'TrainCode/code'; 
  - 2. Run the following scripts to train the models:
  

>  
> 
>  ## BI degradation, scale 2, 3, 4,8
>  ## input= 48x48, output = 96x96
>  python main.py  --model san  --save `save_name`  --scale 2 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --patch_size 96
>  ## input= 48x48, output = 144x144
>  python main.py  --model san  --save `save_name`  --scale 3 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --patch_size 96
>  ## input= 48x48, output = 192x192
>  python main.py  --model san  --save `save_name`  --scale 4 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --patch_size 96
>  ## input= 48x48, output = 392x392
>  python main.py  --model san  --save `save_name`  --scale 8 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --patch_size 96
>

### 3. Test code
-  1. You can [Download the pretrained model first](https://pan.baidu.com/s/1aTYG4Wy72MI-gCRGnJgkvQ), password: eq1v
-  2. CD to 'TestCode/code', run the following scripts

>  
>  ## BI degradation, scale 2, 3, 4,8
>  ## SAN_2x
> 
>  python main.py  --model san  --data_test MyImage  --save `save_name`  --scale 2 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --test_only --testpath 'your path' --testset Set5  --pre_train ../model/SAN_BIX2.pt  
> 
>  # SAN_3x   
> 
>  python main.py  --model san --data_test MyImage  --save `save_name`  --scale 3 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --test_only --testpath 'your path' --testset Set5  --pre_train ../model/SAN_BIX3.pt  
> 
>  # SAN_4x
>  python main.py  --model san --data_test MyImage  --save `save_name`  --scale 4 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --test_only --testpath 'your path' --testset Set5  --pre_train ../model/SAN_BIX4.pt
> 
>  # SAN_8x
> 
>  python main.py  --model san --data_test MyImage  --save `save_name`  --scale 8 --n_resgroups 20 --n_resblocks 10 --n_feats 64 --reset --chop --save_results --test_only --testpath 'your path' --testset Set5  --pre_train ../model/SAN_BIX8.pt
> 
### 4. Results
- Some of [the test results can be downloaded.](https://pan.baidu.com/s/1j0ZgfbGKyYZqsSCLOb3nUg)  Password:w3da

### 5. Citation
If the the work or the code is helpful, please cite the following papers

> @inproceedings{dai2019second,
> 
> title={Second-order Attention Network for Single Image Super-Resolution},
  author={Dai, Tao and Cai, Jianrui and Zhang, Yongbing and Xia, Shu-Tao and Zhang, Lei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={11065--11074},
  year={2019}
}

> @inproceedings{zhang2018image,
> 
  title={Image super-resolution using very deep residual channel attention networks},
  author={Zhang, Yulun and Li, Kunpeng and Li, Kai and Wang, Lichen and Zhong, Bineng and Fu, Yun},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={286--301},
  year={2018}
}

> @inproceedings{li2017second,
>  title={Is second-order information helpful for large-scale visual recognition?},
  author={Li, Peihua and Xie, Jiangtao and Wang, Qilong and Zuo, Wangmeng},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2070--2078},
  year={2017}
}

### 6. Acknowledge
The code is built on [RCAN (Pytorch)](https://github.com/yulunzhang/RCAN) and [EDSR (Pytorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors  for sharing the codes.