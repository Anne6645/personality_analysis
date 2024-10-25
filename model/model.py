import torch.nn as nn
import torch
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import sys
import json
from collections import OrderedDict

class self_feature_learn_BiLSTM(nn.Module):
    def __init__(self,input_size):
        super(self_feature_learn_BiLSTM,self).__init__()
        self.BiLSTM=nn.Sequential(
            nn.Linear(in_features=input_size,out_features=128),
            nn.ReLU(),
            nn.LSTM(input_size=128,hidden_size=128,num_layers=2,bidirectional=True)
        )
        self.output_net = nn.Linear(in_features=256,out_features=256)
        

    def forward(self,x):
        
        output,(hn,cn) =self.BiLSTM(x)
        return output[-1,:,:]

class self_attention(nn.Module):
    def __init__(self,hidden_size):
        super(self_attention,self).__init__()
        self.q_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size,out_features=hidden_size)
        )
        self.k_layer = nn.Sequential(
            nn.Linear(in_features = hidden_size,out_features = hidden_size)
        )
        self.v_layer = nn.Sequential(
            nn.Linear(in_features =hidden_size,out_features = hidden_size)
        )

    def forward(self,x):

        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)
        scores = torch.matmul(q,torch.permute(k,dims=(1,0)))
        alpha = torch.softmax(scores,dim=1)
        v = torch.matmul(alpha,v) #[n,t,e]
        return v

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    @torch.no_grad()
    def forward(self, scores, target) -> (int, torch.Tensor):
        pred_value = scores.detach().numpy()
        ori_value =target.detach().numpy()
        y_label =['extraversion', 'neuroticism', 'agreeableness', 
              'conscientiousness', 'openness']
        i =0
        j =0
        acc_dict ={}
        acc_item_list=[]
        while j<=len(y_label)-1:
            pred_y = pred_value[:,j]
            ori_y = ori_value[:,j]
            acc_label =y_label[j]
            acc_item=0

            while i <= len(pred_y)-1:
                acc_y_pred = pred_y[i]
                acc_y = ori_y[i]
                acc_item += 1-abs(acc_y_pred-acc_y)
                i+=1
            acc_item =acc_item/len(pred_y)
            acc_dict.update({acc_label:acc_item})
            j+=1
            acc_item_list.append(acc_item)
            i=0
            
        average_acc_item=np.mean(acc_item_list)
        return acc_dict,average_acc_item

class Data_Fusion_Model(nn.Module):
    def __init__(self,x1_shape,x2_shape,x3_shape):
        super(Data_Fusion_Model_v2,self).__init__()
        self.model_1 = nn.Bilinear(x1_shape,x2_shape,128)
        self.model_2=  nn.Bilinear(x1_shape,x3_shape,128)
        self.MLP = nn.Linear(in_features=128,out_features=128)

    def forward(self,x,y,z):

        x_y = self.model_1(x,y)
        x_z = self.model_2(x,z)
        f = x_y +x_z
        f= self.MLP(f)
        return f

class Prediction_Model(nn.Module):
    def __init__(self,input_size):
        super(Prediction_Model,self).__init__()
        self.Predict_model = nn.Sequential(
            nn.Linear(in_features=input_size,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=5),
            # nn.ReLU(),
            # nn.Linear(in_features=256,out_features=5),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.Predict_model(x)
    
def feature_flatten(self_feature_path,shared_feature_path,
                    output_path):
    dir_name =os.listdir(self_feature_path)
    for dir in tqdm(dir_name,desc='file process',unit='dir'):
        topic_name =os.listdir(self_feature_path+'/'+dir)
        for topic in tqdm(topic_name,desc='file process',unit='topic'):
                file_name =os.listdir(self_feature_path+'/'+dir+'/'+topic)
                for file in tqdm(file_name,desc='file process',unit='file'):
                    try:
                        self_feature = read_pkl_file(self_feature_path+'/'
                                             +dir+'/'+topic+'/'+file)
                        if len(self_feature.shape)!=3:
                            self_feature =self_feature.unsqueeze(dim=1)
                        self_feature =self_feature.reshape(1,self_feature.shape[0]*self_feature.shape[2])
                        shared_feature =read_pkl_file(shared_feature_path+
                                              '/'+topic+'/'+file)
                        if len(shared_feature.shape)!=3:
                            shared_feature =shared_feature.unsqueeze(dim=1)
                        shared_feature=shared_feature.reshape(1,shared_feature.shape[0]*shared_feature.shape[2])
                        res =torch.cat((self_feature,shared_feature),1)
                    except:
                        print(dir,topic,file)
                        sys.exit
                    pa =output_path+topic+'/'
                    if not os.path.exists(output_path+topic+'/'):
                        os.makedirs(pa)
                    with open(pa+file,'wb') as file_res:
                        pkl.dump(res,file_res)

    
def data_to_tensor(json_path,path_for_feature):
    topics_list = ["topic1", "topic2", "topic3", "topic4", "topic5", 
                   "topic6", "topic7", "topic8", "topic9", "topic0","topic10",
                  "topic11","topic12","topic13","topic14","topic15","topic16","topic17","topic18","topic19"]
    y_label =['extraversion', 'neuroticism', 'agreeableness', 
              'conscientiousness', 'openness']
    for  topic in tqdm(topics_list,desc='data processing',unit ='topic'):
        path_for_feature_1 =path_for_feature+'/'+topic
        file_name =os.listdir(path_for_feature_1)
        i=0
        start_mask =0
        for file in tqdm(file_name,desc='data processing',unit='file'):
            path_for_feature_2 =path_for_feature_1 +'/'+file
            with open(path_for_feature_2,'rb') as file_x:
                x =pkl.load(file_x)
                # if len(x_tensor.shape)==1:
                # x_tensor =torch.randn(x.shape[0],x.shape[1])
                # x_tensor =torch.concat((x_tensor,x),dim=0)
                if start_mask==0:
                    x_tensor =x
                    start_mask +=1  
                else:
                    x_tensor = torch.concat((x_tensor,x),dim=0)
            with open(json_path,'rb') as file_y:
                data =json.load(file_y)
                file_name_in_json =file[:-3]+'mp4'
                if i ==0:
                    y_tensor = torch.tensor((data[file_name_in_json][y_label[0]],
                                            data[file_name_in_json][y_label[1]],
                                            data[file_name_in_json][y_label[2]],
                                            data[file_name_in_json][y_label[3]],
                                            data[file_name_in_json][y_label[4]]))
                    
                    y_tensor=torch.unsqueeze(y_tensor,dim=0)                
                    i+=1
                else:
                    y = torch.tensor((data[file_name_in_json][y_label[0]],
                                    data[file_name_in_json][y_label[1]],
                                    data[file_name_in_json][y_label[2]],
                                    data[file_name_in_json][y_label[3]],
                                    data[file_name_in_json][y_label[4]]))
                    y =torch.unsqueeze(y,dim=0)
                    y_tensor=torch.concat((y_tensor,y),dim=0)
        x_export_path ='/Users/ansixu/final/'+topic+'_x_tensor_all_wface.pkl'
        with open(x_export_path,'wb') as file:
            pkl.dump(x_tensor,file)
        y_export_path ='/Users/ansixu/final/'+topic+'_y_tensor_all_wface.pkl'
        with open(y_export_path,'wb') as file:
            pkl.dump(y_tensor,file)
    
