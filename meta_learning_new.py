import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader  
from learn2learn import algorithms  
from learn2learn.data import TaskDataset  
import pickle  
from Meta_Adapt import new_get_topic_dataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from config import*
import model.Model as Model
import numpy as np
import collections
import copy
from transformers.optimization import AdamW
from utils import utils
import sys
import tensorflow as tf
from subexperiments import transformer
import csv


# 1. 加载存储在.pkl文件中的数据集  
def load_pkl_data(file_path):  
    with open(file_path, 'rb') as f:  
        data = pickle.load(f)  
    return data  

# 数据读取dataloader
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
    
def functional_task_model(fast_weight,x,y):
    x=F.linear(x,
               fast_weight['Predict_model.0.weight'],
               fast_weight['Predict_model.0.bias'],)
    x =torch.relu(x)
    logits = F.linear(x,
               fast_weight['Predict_model.2.weight'],
               fast_weight['Predict_model.2.bias'],)
    logits=torch.nn.functional.sigmoid(logits)
    
    loss =None
    if y is not None:
        loss =F.mse_loss(logits.view(-1),y.view(-1))
    
    return loss,logits


def replace_weights(module, fw):  
    for name, param in module.named_parameters():  
        if name in fw:  
            param.data = fw[name]  
            
def functional_transformer(fast_weight,x,y):
    
    
    # input_dim =       
    # model_dim = 512
    # num_heads = 8  
    # num_layers = 2  
    # dropout = 0.1 
    # model = torch.jit.load('/Users/ansixu/final/trained_model_t5.pt')
    
    # model.apply(lambda module: replace_weights(module, fast_weight)) 
    
    
    
    # logits =model(x)
    
    # logits =nn.Sigmoid
    
    # fast_weights=collections.OrderedDict(model.named_parameters())
    
    
    x=F.linear(x,
               fast_weight['input_layer.weight'],
               fast_weight['input_layer.bias'],)
    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.0.self_attn.in_proj_weight'],
               fast_weight['transformer_encoder.layers.0.self_attn.in_proj_bias'],)
    
    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.0.self_attn.out_proj.weight'],
               fast_weight['transformer_encoder.layers.0.self_attn.out_proj.bias'],)
    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.0.linear1.weight'],
               fast_weight['transformer_encoder.layers.0.linear1.bias'],)
    
    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.0.norm1.weight'],
               fast_weight['transformer_encoder.layers.0.norm1.bias'],)
    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.0.linear2.weight'],
               fast_weight['transformer_encoder.layers.0.linear2.bias'],)
    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.0.norm2.weight'],
               fast_weight['transformer_encoder.layers.0.norm2.bias'],)

    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.1.self_attn.in_proj_weight'],
               fast_weight['transformer_encoder.layers.1.self_attn.in_proj_bias'],)
    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.1.self_attn.out_proj.weight'],
               fast_weight['transformer_encoder.layers.1.self_attn.out_proj.bias'],)
    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.1.linear1.weight'],
               fast_weight['transformer_encoder.layers.1.linear1.bias'],)
    
    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.1.norm1.weight'],
               fast_weight['transformer_encoder.layers.1.norm1.bias'],)
    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.1.linear2.weight'],
               fast_weight['transformer_encoder.layers.1.linear2.bias'],)
    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.1.norm2.weight'],
               fast_weight['transformer_encoder.layers.1.norm2.bias'],)
    
    x=F.linear(x,
               fast_weight['transformer_encoder.layers.1.norm2.weight'],
               fast_weight['transformer_encoder.layers.1.norm2.bias'],)
    
    x=F.linear(x,
               fast_weight['output_layer.weight'],
               fast_weight['output_layer.weight'],)
    logits =torch.sigmoid(x)
    
    loss =None
    if y is not None:
        loss =F.mse_loss(logits.view(-1),y.view(-1))
    
    return loss,logits

# 读取每一个pkl 文件中的将数据存储 
task0_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic0_x_tensor_all.pkl')  
task1_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic1_x_tensor_all.pkl')  
task2_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic2_x_tensor_all.pkl')  
task3_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic3_x_tensor_all.pkl')
task4_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic4_x_tensor_all.pkl')    
task5_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic5_x_tensor_all.pkl')
task6_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic6_x_tensor_all.pkl')  
task7_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic7_x_tensor_all.pkl')
task8_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic8_x_tensor_all.pkl')
task9_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic9_x_tensor_all.pkl')
task10_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic10_x_tensor_all.pkl')
task11_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic11_x_tensor_all.pkl')
task12_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic12_x_tensor_all.pkl')
task13_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic13_x_tensor_all.pkl')
task14_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic14_x_tensor_all.pkl')
task15_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic15_x_tensor_all.pkl')
task16_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic16_x_tensor_all.pkl')
task17_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic17_x_tensor_all.pkl')
task18_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic18_x_tensor_all.pkl')
task19_data = load_pkl_data('/Users/ansixu/final/x_tensor_all/topic19_x_tensor_all.pkl')

 
task0_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic0_y_tensor_all.pkl') 
task1_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic1_y_tensor_all.pkl') 
task2_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic2_y_tensor_all.pkl') 
task3_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic3_y_tensor_all.pkl') 
task4_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic4_y_tensor_all.pkl') 
task5_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic5_y_tensor_all.pkl') 
task6_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic6_y_tensor_all.pkl') 
task7_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic7_y_tensor_all.pkl') 
task8_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic8_y_tensor_all.pkl') 
task9_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic9_y_tensor_all.pkl') 
task10_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic10_y_tensor_all.pkl') 
task11_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic11_y_tensor_all.pkl') 
task12_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic12_y_tensor_all.pkl') 
task13_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic13_y_tensor_all.pkl') 
task14_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic14_y_tensor_all.pkl') 
task15_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic15_y_tensor_all.pkl') 
task16_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic16_y_tensor_all.pkl') 
task17_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic17_y_tensor_all.pkl') 
task18_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic18_y_tensor_all.pkl') 
task19_y = load_pkl_data('/Users/ansixu/final/y_tensor_all/topic19_y_tensor_all.pkl') 

def data_loader(x_path,y_path,topic):
    task_data=load_pkl_data(x_path+topic+'_x_tensor_all_wtext.pkl')
    task_y =load_pkl_data(y_path+topic+'_y_tensor_all_wtext.pkl')
    source_data =GetLoader(task_data,task_y)
    source_dataloader= DataLoader(source_data, batch_size=5)  
    return source_dataloader
    

# task0_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic0_x_tensor_all_wtext.pkl')  
# task1_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic1_x_tensor_all_wtext.pkl')  
# task2_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic2_x_tensor_all_wtext.pkl')  
# task3_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic3_x_tensor_all_wtext.pkl')
# task4_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic4_x_tensor_all_wtext.pkl')    
# task5_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic5_x_tensor_all_wtext.pkl')
# task6_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic6_x_tensor_all_wtext.pkl')  
# task7_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic7_x_tensor_all_wtext.pkl')
# task8_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic8_x_tensor_all_wtext.pkl')
# task9_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic9_x_tensor_all_wtext.pkl')
# task10_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic10_x_tensor_all_wtext.pkl')
# task11_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic11_x_tensor_all_wtext.pkl')
# task12_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic12_x_tensor_all_wtext.pkl')
# task13_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic13_x_tensor_all_wtext.pkl')
# task14_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic14_x_tensor_all_wtext.pkl')
# task15_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic15_x_tensor_all_wtext.pkl')
# task16_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic16_x_tensor_all_wtext.pkl')
# task17_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic17_x_tensor_all_wtext.pkl')
# task18_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic18_x_tensor_all_wtext.pkl')
# task19_data = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/x_final_feature_wtext/topic19_x_tensor_all_wtext.pkl')


# task0_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic0_y_tensor_all_wtext.pkl') 
# task1_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic1_y_tensor_all_wtext.pkl') 
# task2_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic2_y_tensor_all_wtext.pkl') 
# task3_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic3_y_tensor_all_wtext.pkl') 
# task4_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic4_y_tensor_all_wtext.pkl') 
# task5_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic5_y_tensor_all_wtext.pkl') 
# task6_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic6_y_tensor_all_wtext.pkl') 
# task7_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic7_y_tensor_all_wtext.pkl') 
# task8_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic8_y_tensor_all_wtext.pkl') 
# task9_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic9_y_tensor_all_wtext.pkl') 
# task10_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic10_y_tensor_all_wtext.pkl') 
# task11_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic11_y_tensor_all_wtext.pkl') 
# task12_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic12_y_tensor_all_wtext.pkl') 
# task13_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic13_y_tensor_all_wtext.pkl') 
# task14_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic14_y_tensor_all_wtext.pkl') 
# task15_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic15_y_tensor_all_wtext.pkl') 
# task16_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic16_y_tensor_all_wtext.pkl') 
# task17_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic17_y_tensor_all_wtext.pkl') 
# task18_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic18_y_tensor_all_wtext.pkl') 
# task19_y = load_pkl_data('/Users/ansixu/final/subexperiments/多模态有效性测试数据/y_final_subex1-1_wtext/topic19_y_tensor_all_wtext.pkl') 

# 2. 定义任务模型  
class Prediction_Model(nn.Module):
    """
    预测模型给出五个预测的数值在五个维度
    """
    def __init__(self,input_size):
        super(Prediction_Model,self).__init__()
        self.Predict_model = nn.Sequential(
            nn.Linear(in_features=input_size,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=5),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.Predict_model(x)
    
    def fastweight_update():
        pass
        

# 准备source 的数据集
# source_data1 =GetLoader(scaled_task1_data,task1_y)
# source_dataloader1 = DataLoader(source_data1, batch_size=5)  
# source_data2 =GetLoader(scaled_task2_data,task2_y)
# source_dataloader2 = DataLoader(source_data2, batch_size=5)  
# source_data3 =GetLoader(scaled_task3_data,task3_y)
# source_dataloader3 = DataLoader(source_data3, batch_size=5)  
# source_data4 =GetLoader(scaled_task4_data,task4_y)
# source_dataloader4 = DataLoader(source_data4, batch_size=5)  
# source_data5 =GetLoader(scaled_task5_data,task5_y)
# source_dataloader5 = DataLoader(source_data5, batch_size=5)  
# source_data6 =GetLoader(scaled_task6_data,task6_y)
# source_dataloader6 = DataLoader(source_data6, batch_size=5)  
# source_data7 =GetLoader(scaled_task7_data,task7_y)
# source_dataloader7 = DataLoader(source_data7, batch_size=5)  


source_data0 =GetLoader(task0_data,task0_y)
source_dataloader0= DataLoader(source_data0, batch_size=5)  
source_data1 =GetLoader(task1_data,task1_y)
source_dataloader1 = DataLoader(source_data1, batch_size=5)  
source_data2 =GetLoader(task2_data,task2_y)
source_dataloader2 = DataLoader(source_data2, batch_size=5)  
source_data3 =GetLoader(task3_data,task3_y)
source_dataloader3 = DataLoader(source_data3, batch_size=5)  
source_data4 =GetLoader(task4_data,task4_y)
source_dataloader4 = DataLoader(source_data4, batch_size=5)  
source_data5 =GetLoader(task5_data,task5_y)
source_dataloader5 = DataLoader(source_data5, batch_size=5)  
source_data6 =GetLoader(task6_data,task6_y)
source_dataloader6 = DataLoader(source_data6, batch_size=5)  
source_data7 =GetLoader(task7_data,task7_y)
source_dataloader7 = DataLoader(source_data7, batch_size=5) 
source_data8 =GetLoader(task8_data,task8_y)
source_dataloader8 = DataLoader(source_data8, batch_size=5) 
source_data9 =GetLoader(task8_data,task7_y)
source_dataloader9 = DataLoader(source_data7, batch_size=5) 
source_data10 =GetLoader(task7_data,task7_y)
source_dataloader10 = DataLoader(source_data7, batch_size=5) 
source_data11 =GetLoader(task7_data,task7_y)
source_dataloader11 = DataLoader(source_data7, batch_size=5) 
source_data12 =GetLoader(task7_data,task7_y)
source_dataloader12 = DataLoader(source_data7, batch_size=5) 
source_data13 =GetLoader(task7_data,task7_y)
source_dataloader13 = DataLoader(source_data7, batch_size=5) 
source_data14 =GetLoader(task7_data,task7_y)
source_dataloader14 = DataLoader(source_data7, batch_size=5)
source_data15 =GetLoader(task7_data,task7_y)
source_dataloader15 = DataLoader(source_data7, batch_size=5) 
source_data15 =GetLoader(task7_data,task7_y)
source_dataloader7 = DataLoader(source_data7, batch_size=5) 
source_data16 =GetLoader(task7_data,task7_y)
source_dataloader16 = DataLoader(source_data7, batch_size=5) 
source_data17 =GetLoader(task7_data,task7_y)
source_dataloader17 = DataLoader(source_data7, batch_size=5)
source_data18 =GetLoader(task7_data,task7_y)
source_dataloader18 = DataLoader(source_data7, batch_size=5) 
source_data19 =GetLoader(task7_data,task7_y)
source_dataloader19 = DataLoader(source_data7, batch_size=5) 
 

# 准备目标数据集

# source_data1 =GetLoader(scaled_task1_data,task1_y)
# source_dataloader1 = DataLoader(source_data1, batch_size=5)  
# 3. 定义元学习模型  
# meta_model = Prediction_Model(1536)  # 用于few-shot 数据训练的模型
# meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.0001)
# task_model = Prediction_Model(1536) # 用于各分任务的元模型
task_model = Prediction_Model(1536)  # 用于few-shot 数据训练的模型
# task_optimizer = optim.Adam(meta_model.parameters(), lr=0.0001)
# task_model =torch.jit.load('/Users/ansixu/final/models/trained_model_t6.pt')


# 4. 定义训练循环和测试循环   
def train_epoch(args,num_iterations): 
    trange = tqdm(range(num_iterations))
    # 准备target 的数据集
    query_dataset,val_dataset =new_get_topic_dataset(args,args.target_topic,args.path).load_topic_pkl_file()
    query_loader = DataLoader(
    query_dataset,
    sampler=RandomSampler(query_dataset),
    batch_size=args.train_batchsize
    )

# 准备val data 
    val_dataloader = DataLoader(
    val_dataset, 
    sampler=RandomSampler(val_dataset),
    batch_size=args.eval_batchsize) 
    
    # 准备元模型对应的数据集
    dataloaders = [source_dataloader0, source_dataloader1, source_dataloader2,source_dataloader3,  
                source_dataloader4,source_dataloader6,source_dataloader7,source_dataloader8,source_dataloader9,source_dataloader10,
                source_dataloader11,source_dataloader12,source_dataloader13,source_dataloader14,source_dataloader15,
                source_dataloader16,source_dataloader17,source_dataloader18,source_dataloader19]
    # dataloaders = [source_dataloader19]
    # dataloaders = [source_dataloader19, source_dataloader5,  
    #           source_dataloader6,source_dataloader8,source_dataloader10]
    # dataloaders = [source_dataloader3,source_dataloader8,source_dataloader19,source_dataloader10,source_dataloader13,
    #                source_dataloader14,source_dataloader15,source_dataloader16,source_dataloader17,source_dataloader18
    #             ]
    # dataloaders = [source_dataloader19,
    #              source_dataloader4, source_dataloader5,  
    #             source_dataloader6,source_dataloader7,source_dataloader8,source_dataloader3,source_dataloader10,
    #             source_dataloader1,source_dataloader2,source_dataloader14,source_dataloader15,
    #             source_dataloader16,source_dataloader17,source_dataloader18]
    
    print(args.target_topic)
# 设置优化器
    task_optimizer = optim.Adam(task_model.parameters(), lr=0.0001)
    huber_loss = nn.SmoothL1Loss()
    
# 参数更新初始化
    optimizer = torch.optim.AdamW(task_model.parameters(), lr=args.learning_rate_meta)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=args.num_iterations,
                                                           eta_min=args.learning_rate_meta*args.lr_decay_meta)
    inner_loop_optimizer =utils.LSLRGradientDescentLearningRule(
                                                           total_num_inner_loop_steps=args.num_updates,
                                                           init_learning_rate=args.learning_rate_learner,
                                                           use_learnable_lr=True,
                                                           lr_of_lr=args.learning_rate_lr)
    inner_loop_optimizer.initialize(names_weights_dict=task_model.named_parameters())   
    
    print('***** Running adaptation *****')
    
    losses_list=[]
    # scaler = torch.cuda.amp.GradScaler()

# 开始迭代训练
    for p in trange:
        task_model.train()
        mean_outer_loss=0
        meta_gradients = []
        domain_gradients_list=[]
        domain_similarity=[]
        losses=[]
        for i, dataloader in enumerate(dataloaders, start=1):
            # task_model = Prediction_Model(1536) 
            # print(f"遍历第 {i} 个 dataloader:") 
            # print(collections.OrderedDict(task_model.named_parameters()))
            fast_weights=collections.OrderedDict(task_model.named_parameters())
            t=0
            for task_batch in dataloader:
                x, y = task_batch
                t+=1
                for k  in range(args.num_updates):
                # 内循环：更新任务模型参数 
                    outputs = functional_task_model(fast_weight=fast_weights,x=x,y=y)  
                    # outputs = functional_transformer(fast_weight=fast_weights,x=x,y=y) 
                    loss = outputs[0]
                    # print('训练loss:',loss)
                    task_model.zero_grad()
                    # print("训练第{}loss:{:.4f}".format(k,loss))
                    # fast_weights=outputs[2]
                    scaled_grads = torch.autograd.grad(loss, fast_weights.values(),
                                                        create_graph=True, retain_graph=True,allow_unused=True)
                    # inv_scale = 1. / scaler.get_scale()
                    # grads = [p * inv_scale for p in scaled_grads]

                    fast_weights = inner_loop_optimizer.update_params(fast_weights, scaled_grads, k)
                
                
            # print(loss)   
            # print(f"遍历第 {i} 个 dataloader已经完成")
            weight_before = collections.OrderedDict(task_model.named_parameters())
            domain_gradients=tuple()
            for _,(params_before,params_after) in enumerate(zip(weight_before,fast_weights)):
                domain_gradients +=(fast_weights[params_before].detach()-weight_before[params_before].detach(),)
            domain_gradients_list.append(domain_gradients)
        # 外循环：更新元模型参数 
            # print(f"开始计算target domain在第 {i} 个 dataloader的元模型meta model 上的meta loss 和grad")
            # meta_optimizer.zero_grad()
        # 将target 中的train部分 带入计算 query loss 然后计算meta loss 将带入的数据的loss 全部加起来
            
            meta_loss =0
            for x,y in query_loader:
                outputs =functional_task_model(fast_weights,x,y)
                # outputs = functional_transformer(fast_weight=fast_weights,x=x,y=y) 
                
                query_loss =outputs[0]
                
                meta_loss +=query_loss
                
            meta_loss /=len(query_loader)
            meta_grads = torch.autograd.grad(meta_loss,
                                task_model.parameters(),create_graph=True, retain_graph=True,
                    allow_unused=True)
            meta_gradients.append(meta_grads)
            # print(f"target domain在第 {i} 个 dataloader的元模型meta model 上的meta loss 和grad 计算完成")
            cur_similarity = [F.cosine_similarity(a.view(-1), b.view(-1), dim=-1) for a, b in
                        zip(domain_gradients, meta_grads)]
            cur_similarity = torch.mean(torch.tensor(cur_similarity))
            domain_similarity.append(cur_similarity)
            
            if p ==20:
                print(domain_similarity)
            
        # 使用meta loss 更新可学习的lr
            inner_loop_optimizer.update_lrs(meta_loss)
            mean_outer_loss+=meta_loss
            
        domain_similarity = F.softmax(torch.tensor(domain_similarity) / args.softmax_temp, -1)
                # print("训练第{}loss:{:.4f}".format(k,loss))
        for weights in zip(task_model.parameters(), *meta_gradients):
            for k in range(len(meta_gradients)):
                if k == 0:
                    weights[0].grad = domain_similarity[k] * weights[k+1]
                else:
                    weights[0].grad += domain_similarity[k] * weights[k+1]
                    
        optimizer.step()

        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        # scaler.step(optimizer)
        # scaler.update()
            
        optimizer.zero_grad()
        scheduler.step()
            
        # sometimes scale becomes too small
        # if scaler.get_scale() < 1.:
        #     scaler.update(1.)
        
        mean_outer_loss /= len(dataloaders)
        losses.append(mean_outer_loss.item())
        losses_list.append(np.array(losses).mean())
        trange.set_description('Meta loss,: {:.4f},'.format(np.array(losses).mean())) 
    print(losses_list) 
    print(domain_similarity)
        # 模型评估阶段
    print('模型进入评估阶段')
    task_model.eval()
    test_losses = []
    test_true, test_total = 0, 0
    eval_acc=[]
    eval_average_acc=[]
    loss_fn = nn.MSELoss()
    acc_fn = Model.Accuracy()
    loss_list =[]
    with torch.no_grad():
        for x, y in val_dataloader:
                # 前向过程
            scores = task_model(x)
            loss = loss_fn(scores, y)
            loss_list.append(np.array(loss))
            final_acc,average_acc = acc_fn(scores,y)
            eval_acc.append(final_acc)
            eval_average_acc.append(average_acc)
            # loss = loss.item()
            # test_total += n
            # test_batch += 1
            test_losses.append(loss)

    eval_result = {
        'average_final_acc':eval_average_acc[-1],
        'final_acc': eval_acc[-1],
        'best_average_acc':max(eval_average_acc),
        'best_acc':eval_acc[eval_average_acc.index(max(eval_average_acc))],
        'full_averge_acc':np.mean(eval_average_acc)
        }
    print('Result', eval_result)
    # print(np.mean(eval_acc))
    
    acc_list =eval_acc


    extraversion_index =[]
    neuroticism_index =[]
    agreeableness_index =[]
    conscientiousness_index =[]
    openness_index =[]
    # average_acc_index=[]

    for i in acc_list:
        extraversion_index.append(i['extraversion'])
        neuroticism_index.append(i['neuroticism'])
        agreeableness_index.append(i['agreeableness'])
        conscientiousness_index.append(i['conscientiousness'])
        openness_index.append(i['openness'])
        # average_acc_index.append(i['average_acc'])
        
    average_extraversion=np.mean(np.array(extraversion_index))
    average_neuroticism=np.mean(np.array(neuroticism_index))
    average_agreeableness=np.mean(np.array(agreeableness_index))
    average_conscientiousness=np.mean(np.array(conscientiousness_index))
    average_openness=np.mean(np.array(openness_index))
    # average_average_acc=np.mean(np.array(average_acc_index))

    print(average_extraversion,average_neuroticism,average_agreeableness,average_conscientiousness,
        average_openness,np.mean(np.array(eval_average_acc)))
    
    print(np.mean(np.array(loss_list)))

def finetune(args,num_iterations):
# 准备finetune 的数据集
    query_dataset,val_dataset =new_get_topic_dataset(args,args.target_topic,args.path).load_topic_pkl_file()
    
    query_loader = DataLoader(
    query_dataset,
    sampler=RandomSampler(query_dataset),
    batch_size=args.train_batchsize
    )

# 准备val data 
    val_dataloader = DataLoader(
    val_dataset, 
    sampler=RandomSampler(val_dataset),
    batch_size=args.eval_batchsize)

# 加载模型
    task_model =torch.jit.load('/Users/ansixu/final/models/trained_model_s1-1.pt')
# 优化器设置
    task_optimizer = AdamW(task_model.parameters(), lr=1e-5) 
    trange = tqdm(range(num_iterations))

# 使用query 的数据对模型进行简单训练
    for i in trange:
        for task_batch in query_loader:  
            x, y = task_batch   
            task_optimizer.zero_grad()  
            predictions = task_model(x)  
            loss = nn.MSELoss()(predictions, y)
            loss.backward(retain_graph=True)  
            task_optimizer.step()

        # 模型评估
    print('模型进入评估阶段')
    task_model.eval()
    test_losses = []
    eval_acc=[]
    eval_average_acc=[]
    loss_fn = nn.MSELoss()
    acc_fn = Model.Accuracy()
    loss_list=[]
    with torch.no_grad():
        for x, y in val_dataloader:
            scores = task_model(x)
            loss = loss_fn(scores, y)
            loss_list.append(loss)
            final_acc,average_acc = acc_fn(scores,y)
            eval_acc.append(final_acc)
            eval_average_acc.append(average_acc)
            test_losses.append(loss)

    result  = {
    'average_final_acc':eval_average_acc[-1],
    'final_acc': eval_acc[-1],
    'best_average_acc':max(eval_average_acc),
    'best_acc':eval_acc[eval_average_acc.index(max(eval_average_acc))],
    'full_averge_acc':np.mean(eval_average_acc)
    }
    
    print('Result for evaluation', result)
    
    acc_list =eval_acc
    

    extraversion_index =[]
    neuroticism_index =[]
    agreeableness_index =[]
    conscientiousness_index =[]
    openness_index =[]
    # average_acc_index=[]

    for i in acc_list:
        extraversion_index.append(i['extraversion'])
        neuroticism_index.append(i['neuroticism'])
        agreeableness_index.append(i['agreeableness'])
        conscientiousness_index.append(i['conscientiousness'])
        openness_index.append(i['openness'])
        # average_acc_index.append(i['average_acc'])
        
    average_extraversion=np.mean(np.array(extraversion_index))
    average_neuroticism=np.mean(np.array(neuroticism_index))
    average_agreeableness=np.mean(np.array(agreeableness_index))
    average_conscientiousness=np.mean(np.array(conscientiousness_index))
    average_openness=np.mean(np.array(openness_index))
    # average_average_acc=np.mean(np.array(average_acc_index))

    print(average_extraversion,average_neuroticism,average_agreeableness,average_conscientiousness,
        average_openness,np.mean(np.array(eval_average_acc)))
    print(np.mean(np.array(loss_list)))
    
if __name__=='__main__':

    train_epoch(args,20)  
    # finetune(args,10)
  
