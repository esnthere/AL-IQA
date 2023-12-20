# AL-IQA: Active Learning-Based Sample Selection for Label-Efficient Blind Image Quality Assessment
This is the source code for [AL-IQA: Active Learning-Based Sample Selection for Label-Efficient Blind Image Quality Assessment]([https://ieeexplore.ieee.org/document/10003665](https://ieeexplore.ieee.org/document/10355923)).![KG-IQA Framework](https://github.com/esnthere/AL-IQA/blob/main/framework.png)

## Dependencies and Installation
Pytorch: 1.8.1  
CUDA: 10.2  

## For test:
### 1. Data preparation  
   To ensure high speed, save images and lables of each dataset with 'mat' files. Only need to run '**data_preparation_example.py**' once for each dataset.
   
### 2. Load pre-trained weight for test  
   The models pre-trained on KonIQ-10k with 5%, 10%, 25%, 80% samples are released. The dataset are randomly splitted several times during training, and each released model is obtained from the first split (numpy. random. seed(1)). The model file '**my_vision_transformer.py**' is modified from open accessed source code of [DEIT](https://github.com/facebookresearch/deit) and [TIMM](https://github.com/huggingface/pytorch-image-models/tree/main/timm). 
   
   The pre-trained models can be downloaded from: [Pre-trained models](https://pan.baidu.com/s/1kKGTp1iS0QGhuYGSJQVhTg?pwd=o80k). Please download these files and put them in the same folder of code and then run '**test_example_koniq_*n*percent.py**' to make intra/cross dataset test for models trained on *n%* samples.
   
   
## For train:  
The training code can be available at the 'training' folder.


## If you like this work, please cite:

@ARTICLE{10355923,
  author={Song, Tianshu and Li, Leida and Cheng, Deqiang and Chen, Pengfei and Wu, Jinjian},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Active Learning-Based Sample Selection for Label-Efficient Blind Image Quality Assessment}, 
  year={2023, Early Access},  
  doi={10.1109/TCSVT.2023.3341611}}

  
## License
This repository is released under the Apache 2.0 license. 
