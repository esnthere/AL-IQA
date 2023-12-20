# Training Code for AL-IQA: Knowledge-Guided Blind Image Quality Assessment with Few Training Samples
This is the training example of KG-IQA on the RBID dataset, which is small enough to re-train. The trainning process is the same for other datasets:

## 1. Data preparation

   To ensure high speed, save training images and lables, JND images, and NSS features into 'mat/npz' files. The preparation process please refer to the published paper [KG-IQA](https://ieeexplore.ieee.org/document/10003665).  Please run '**data_preparation_example_for_rbid.py**' to save the training images and labels, and other necessary files can be downloaded from [Trainng files](https://pan.baidu.com/s/1EerM_rvNVo8Eevw74p3TNQ?pwd=z3oh). Please download these files and put them into the same folder of the training code.
   
## 2. Training the model

   Run '**training_example_of_rbid_25percent.py**' to train the model. The pre-trained weight and model file '**my_vision_transformer.py**' is modified from open accessed source code of [DEIT](https://github.com/facebookresearch/deit) and [TIMM](https://github.com/huggingface/pytorch-image-models/tree/main/timm). The training example can be seen from '**run training_example_of_rbid_25percent.ipynb**' .

## If you like this work, please cite:

{   
     author={Song, Tianshu and Li, Leida and Wu, Jinjian and Yang, Yuzhe and Li, Yaqian and Guo, Yandong and Shi, Guangming},  
     journal={IEEE Transactions on Multimedia},   
     title={Knowledge-Guided Blind Image Quality Assessment With Few Training Samples},   
     year={2023},
     volume={25},  
     pages={8145-8156},
     doi={10.1109/TMM.2022.3233244}   
  }


