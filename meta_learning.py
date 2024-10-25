import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader  
from learn2learn import algorithms  
from learn2learn.data import TaskDataset  
import pickle  
from utils import new_get_topic_dataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from config import*
import utils as Model
import numpy as np
import collections
import copy
from transformers.optimization import AdamW
import sys
import tensorflow as tf
from subexperiments import transformer
import csv

topic_list=['topic0','topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9','topic10','topic11',
'topic12','topic13','topic14','topic15','topic16','topic17','topic18','topic19']
	    

task0_data = load_pkl_data('data/x_tensor/topic0_x_tensor_all.pkl')  
task1_data = load_pkl_data('data/x_tensor/topic1_x_tensor_all.pkl')  
 
task0_y = load_pkl_data('data/y_tensor/topic0_y.pkl') 
task1_y = load_pkl_data('data/y_tensor/topic1_y.pkl') 

def load_pkl_data(file_path):  
    with open(file_path, 'rb') as f:  
        data = pickle.load(f)  
    return data  

# dataloader
class GetLoader(torch.utils.data.Dataset):
	
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
	    
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

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
            )
		
def data_loader(x_path,y_path,topic):
    task_data=load_pkl_data(x_path+topic+'_x_tensor_all_wtext.pkl')
    task_y =load_pkl_data(y_path+topic+'_y_tensor_all_wtext.pkl')
    source_data =GetLoader(task_data,task_y)
    source_dataloader= DataLoader(source_data, batch_size=5)  
    return source_dataloader

class Prediction_Model(nn.Module):
    """
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

#  
def train_epoch(args,num_iterations): 
    trange = tqdm(range(num_iterations))
    # 准备target 的数据集
    query_dataset,val_dataset =new_get_topic_dataset(args,args.target_topic,args.path).load_topic_pkl_file()
    query_loader = DataLoader(
    query_dataset,
    sampler=RandomSampler(query_dataset),
    batch_size=args.train_batchsize
    )

    val_dataloader = DataLoader(
    val_dataset, 
    sampler=RandomSampler(val_dataset),
    batch_size=args.eval_batchsize) 
    
    
    print(args.target_topic)

    task_optimizer = optim.Adam(task_model.parameters(), lr=0.0001)
    huber_loss = nn.SmoothL1Loss()
    
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
                #  
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
        # 
            # print(f"开始计算target domain在第 {i} 个 dataloader的元模型meta model 上的meta loss 和grad")
            # meta_optimizer.zero_grad()
            
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

        optimizer.zero_grad()
        scheduler.step()
	    
        mean_outer_loss /= len(dataloaders)
        losses.append(mean_outer_loss.item())
        losses_list.append(np.array(losses).mean())
        trange.set_description('Meta loss,: {:.4f},'.format(np.array(losses).mean())) 
    print(losses_list) 
    print(domain_similarity)
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
    
    acc_list =eval_acc


    extraversion_index =[]
    neuroticism_index =[]
    agreeableness_index =[]
    conscientiousness_index =[]
    openness_index =[]

    for i in acc_list:
        extraversion_index.append(i['extraversion'])
        neuroticism_index.append(i['neuroticism'])
        agreeableness_index.append(i['agreeableness'])
        conscientiousness_index.append(i['conscientiousness'])
        openness_index.append(i['openness'])
        
    average_extraversion=np.mean(np.array(extraversion_index))
    average_neuroticism=np.mean(np.array(neuroticism_index))
    average_agreeableness=np.mean(np.array(agreeableness_index))
    average_conscientiousness=np.mean(np.array(conscientiousness_index))
    average_openness=np.mean(np.array(openness_index))

    print(average_extraversion,average_neuroticism,average_agreeableness,average_conscientiousness,
        average_openness,np.mean(np.array(eval_average_acc)))
    
    print(np.mean(np.array(loss_list)))
    
if __name__=='__main__':

    train_epoch(args,20)  
  
