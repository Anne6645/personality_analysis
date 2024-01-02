# Problem: to do list

1. x_tensor is longer than y_tensor for one data
 Check for Result and set baseline
2. lr update
3. meta loss add
4. acc show
5. multimodal Fusion
6. MLP must have the same data shape for each data(try to find another can adapt differeent data in each modal)

#  Code Explain

code structure: this code is divided in two parts:
1. Feature extraction
2. model prediction
3. meta adaptive

Feature extraction:
Feature used in this work includes 4 parts:

Face_encoding_list :dimension(image_selected,128), show feature of face of interviewee

audio_encoding_list:(_,1024), which is the global video feature

audio_mfcc_list: (_,24), 梅尔倒谱系数（Mel-scaleFrequency Cepstral Coefficients，简称MFCC）

'text_encoding_list':(70,1024), which is the text feature 


```
FE Stage 1:
    Use the text to classify data into 10 TOPICS
    Put data into 'Topic' folder
```
```
FE Stage 2:
    Feed 'audio_encoding_list' and 'audio_mfcc_list' into bilstm structure and name the extracted feature as 'self_feature_video_bilstm','self_feature_audio_bilstm'
    (model/Model.py/class self_feature_learn_BiLSTM(nn.Module))
```
```
FE Stage 3:
    Feed'self_feature_video_bilstm','self_feature_audio_bilstm' into self attention.And get 'train_self_feature_video_self_attention' and 'train_self_feature_audio_self_attention'
```
```
FE Stage 4:
    Feed 'self_feature_video_bilstm','self_feature_audio_bilstm' into 
    Data_Fusion_Model and get 'shared_feature'
    (model/Model.py/class Data_Fusion_model(nn.Module))
```
```
FE Stage 5:
    Concate 'shared_feature' and 'train_self_feature_video_self_attention','train_self_feature_audio_self_attention'
     together.

     ! This part has one problem, data must be cut for the same size after be concated together.
```


How to do meta adaptive:
    
 In our work, we divided the dataset into 10 topics according to interview text in each interview video. we use the 'bart-large-mnli' from facebook as text classification model (<https://huggingface.co/facebook/bart-large-mnli>)

 This is specified in the code -model/classification.py

 meta adaptive process:


   meta adaptive process aims to solve the Data scarcity in one specific domain.
   The data vplume in each topic is listed as follow.

   topic 1:3540
   topic 2:1800
   topic 3:
   topic 4:
   topic 5:
   topic 6:
   topic 7:
   topic 8:
   topic 9:
   topic 10:


# Experiment Setting

In this work,we take Topic1 and Topic2 as Source Topic. 
In this work,we take other Topic as Target Topic. 

Example: Source Topic1 -> Target Topic4 

First, we update prediction model on Topic1 for inner update(Support). At this update process, model parameter before update is recorded and model parameter after update is recorded as well. The two paramter is used to calculate task_gradient

Then update model on Target4 (Query). We use query loss to calculate grad on the model paramter. The is called 'meta grad'

Task_gradient and meta_grad is used to calculate domain_similarity

After one iteration, use task similarity to update model parameter.

Enter model 



# Experiment Result analysis

1.Result compare:
Target data into model directly
Target data for transfer update

R:show transfer efficiency

# How to run the code