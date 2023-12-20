# AL-IQA: Active Learning-Based Sample Selection for Label-Efficient Blind Image Quality Assessment
This is the source code for [AL-IQA: Active Learning-Based Sample Selection for Label-Efficient Blind Image Quality Assessment]([https://ieeexplore.ieee.org/document/10003665](https://ieeexplore.ieee.org/document/10355923)).![KG-IQA Framework](https://github.com/esnthere/AL-IQA/blob/main/framework.png)

## Dependencies and Installation
Pytorch: 1.8.1  
CUDA: 10.2  
Python: 3.7

## For test:
### 1. Data preparation  
   To ensure high speed, save images and lables of each dataset with 'mat' files. Only need to run '**data_preparation_example.py**' once for each dataset.
   
### 2. Load pre-trained weight for test  
   The models pre-trained on KonIQ-10k with 5%, 10%, 25%, 80% samples are released. The files in clip are obtained from open accessed source code of [CoOP]([https://github.com/facebookresearch/deit](https://github.com/KaiyangZhou/CoOp)) . 
   
   The pre-trained models can be downloaded from: [Pre-trained models](https://pan.baidu.com/s/111iPWcQ7baaC5b771ZQ3Aw?pwd=j7pq). Please download these files and put them in the same folder of code and then run '**test_koniq_rt'*n*'.py**' to make intra/cross dataset test for models trained on *n%* samples.
   
   
## For train:  
The training code can be available at the 'training' folder.


## If you like this work, please cite:

{

  author={Song, Tianshu and Li, Leida and Cheng, Deqiang and Chen, Pengfei and Wu, Jinjian},
  
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  
  title={Active Learning-Based Sample Selection for Label-Efficient Blind Image Quality Assessment}, 
  
  year={2023, Early Access},  
  
  doi={10.1109/TCSVT.2023.3341611}
  
}

  
## License
This repository is released under the Apache 2.0 license. 
