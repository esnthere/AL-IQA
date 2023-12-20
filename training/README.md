# Training Code for AL-IQA: Active Learning-Based Sample Selection for Label-Efficient Blind Image Quality Assessment
This is the training example of AL-IQA on the LIVEW dataset, which is small enough to re-train. The trainning process is the same for other datasets:

## 1. Data preparation

   To ensure high speed, save training images and lables into 'mat' files. The preparation process please refer to the published paper [AL-IQA](https://ieeexplore.ieee.org/document/10355923).  Please run '**data_preparation_example_for_livew.py**' to save the training images and labels, and '**livew_ds.mat**' contains the distance between samples and distilled images.
   
## 2. Training the model

   Run'**save_first_vote.py**' to save predicted scores for calculating uncertainty and then run '**livew_rt25_first_step.py**' to perform the first round selection and train the model for ther first round.
  
   Run'**save_second_vote.py**' to save predicted scores for calculating uncertainty and then run '**livew_rt25_second_step.py**' to perform the second round selection and train the model for the second round.
   
   Run'**save_third_vote.py**' to save predicted scores for calculating uncertainty and then run '**livew_rt25_third_step.py**' to perform the third round selection and train the model for the third round.
    
   Finally, run '**livew_rt25_fourth_step_ftencoders.py**' to fine-tune the model.
   
   The training example can be seen from '**run_livew_rt25.ipynb**' .

## If you like this work, please cite:

@ARTICLE{AL-IQA,
  author={Song, Tianshu and Li, Leida and Cheng, Deqiang and Chen, Pengfei and Wu, Jinjian},
  
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  
  title={Active Learning-Based Sample Selection for Label-Efficient Blind Image Quality Assessment}, 
  
  year={2023, Early Access},  
  
  doi={10.1109/TCSVT.2023.3341611}}


