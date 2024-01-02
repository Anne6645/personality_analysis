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
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,recall_score


# 第一次特征提取使用BiLSTM
# 每个数据自己学习自己的特征输出
class self_feature_learn_BiLSTM(nn.Module):
    def __init__(self,input_size):
        super(self_feature_learn_BiLSTM,self).__init__()
        self.BiLSTM=nn.Sequential(
            nn.Linear(in_features=input_size,out_features=128),
            nn.LSTM(input_size=128,hidden_size=128,num_layers=2,bidirectional=True)
        )
        self.output_net = nn.Linear(in_features=256,out_features=input_size)

    def forward(self,x):
        #执行BiLSTM 并输出融合结果
        output,(hn,cn) =self.BiLSTM(x)
        return self.output_net(output)

# 利用self_attention 学习模型的自己的特征
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
        """
        前向过程
        ：param x:[n,t,e] n个视频文件 t个抽取的片段  每个片段编码为e 维
        ：return：[n,t,e]
        """
        # 1.获取q,k,v
        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)
        # 2.计算q 和 k 之间 的相关性-> F函数
        scores = torch.matmul(q,torch.permute(k,dims=(0,2,1))) #[n,t,t] 每个视频和每个视频之间的相关性

        # 3.转换为权重
        alpha = torch.softmax(scores,dim=2)

        # 4. 值的合并
        v = torch.matmul(alpha,v) #[n,t,e]
        return v

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    @torch.no_grad()
    def forward(self, scores, target,batch_size) -> (int, torch.Tensor):
        """
        准确率计算方法
          N: 表示样本批次大小；
          C: 表示类别数目
        :param scores: 模型预测对象 [N,C] float类型
        :param target: 样本实际标签类别对象 [N] long类型，内部就是[0,C)的索引id
        :return: (N,准确率值)
        """
        # 获取预测的标签值
        pred_value = scores.detach().numpy()
        ori_value =target.detach().numpy()
        y_label =['extraversion', 'neuroticism', 'agreeableness', 
              'conscientiousness', 'openness']
        i =0
        j =0
        acc_dict ={}
        # print(pred_value[:,i])
        # print(ori_value[:,i])
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
            i=0
        return acc_dict
# 使用bilinear 学习模态间的特征
class Data_Fusion_Model(nn.Module):
    def __init__(self,x1_shape,x2_shape,x3_shape,x4_shape):
        super(Data_Fusion_Model,self).__init__()
        self.model_1 = nn.Bilinear(x1_shape,x2_shape,128)
        self.model_2=  nn.Bilinear(x1_shape,x3_shape,128)
        self.model_3 = nn.Bilinear(x1_shape,x4_shape,128)
        self.MLP = nn.Linear(in_features=128,out_features=128)

    def forward(self,x,y):
        """
        利用transformer 机制计算不同模态之间的关系
        :param X_v: 视频（图片）面部信息
        :param X_vo: 语音信息
        :param X_t: 文本（text）信息
        :return: 利用MLP 将第一个模态的信息与另外两个模态计算的相关性的参数，flat
        """
        x_y = self.model_1(x,y)
        # x_z = self.model_2(x,z)
        # x_k = self.model_3(x,k)
        # f = x_y +x_z+x_k
        f =x_y
        f= self.MLP(f)
        return f

# 最终的预测使用的MLP 模块
    
class Prediction_Model(nn.Module):
    """
    预测模型给出五个预测的数值在五个维度
    """
    def __init__(self,input_size):
        super(Prediction_Model,self).__init__()
        self.Predict_model = nn.Sequential(
            nn.Linear(in_features=input_size,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=5)
        )

    def forward(self,x):
        return self.Predict_model(x)
    
# 将所有的得到的数据进行拼接，拉平成同一维度。
def feature_flatten(self_feature_path,shared_feature_path,
                    output_path):
    dir_name =os.listdir(self_feature_path)
    for dir in tqdm(dir_name,desc='file process',unit='dir'):
        # topic_name =['topic6']
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
    # topics_list=['topic4']
    topics_list = ["topic1","topic2","topic3","topic4", "topic5", 
                 "topic6", "topic7", "topic8", "topic9", "topic10"]
    y_label =['extraversion', 'neuroticism', 'agreeableness', 
              'conscientiousness', 'openness']
    # x_tensor=torch.tensor([0,0,0])
    for  topic in tqdm(topics_list,desc='data processing',unit ='topic'):
        path_for_feature_1 =path_for_feature+'/'+topic
        file_name =os.listdir(path_for_feature_1)
        i=0
        start_mask =0
        for file in tqdm(file_name,desc='data processing',unit='file'):
            path_for_feature_2 =path_for_feature_1 +'/'+file
            with open(path_for_feature_2,'rb') as file_x:
                x =pkl.load(file_x)[:,:5000]
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
        x_export_path ='/Users/ansixu/final/'+topic+'_x_tensor_a_t.pkl'
        with open(x_export_path,'wb') as file:
            pkl.dump(x_tensor,file)
        y_export_path ='/Users/ansixu/final/'+topic+'_y_tensor_a_t.pkl'
        with open(y_export_path,'wb') as file:
            pkl.dump(y_tensor,file) 

         
           
# meta 模型参数更新的规则函数
class LSLRGradientDescentLearningRule(nn.Module):
    def __init__(self,total_num_inner_loop_steps,init_learning_rate =1e-5,use_learnable_lr=True, lr_of_lr=1e-3):
        super(LSLRGradientDescentLearningRule, self).__init__()
        assert init_learning_rate > 0., 'learning_rate should be positive.'

        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.total_num_inner_loop_steps = total_num_inner_loop_steps
        self.use_learnable_lr = use_learnable_lr
        self.lr_of_lr = lr_of_lr

    def initialize(self,names_weight_dict):
        self.names_learning_rates_dict = nn.ParameterDict()
        for key,param in names_weight_dict:
            self.names_learning_rates_dict[key.replace('.','-')] =nn.Parameter(
                data=torch.ones(self.total_num_inner_loop_steps)*self.init_learning_rate,
                requires_grad=self.use_learnable_lr
        )

    def updates_lr(self,loss,scaler =None,names_weight_dict=None):
        loss.requires_grad_(True)
        if self.use_learnable_lr:
            if scaler is not None:
                scaled_grads = torch.autograd.grad(scaler.scale(loss), self.names_learning_rates_dict.values())
                inv_scale = 1. / scaler.get_scale()
                grads = [p * inv_scale for p in scaled_grads]
                if any([False in torch.isfinite(g) for g in grads]):
                    print('Invalid LR gradients, adjust scale and zero out gradients')
                    if scaler.get_scale() * scaler.get_backoff_factor() >= 1.:
                        scaler.update(scaler.get_scale() * scaler.get_backoff_factor())
                    for g in grads: g.zero_()

            else:
                grads = torch.autograd.grad(loss,self.names_learning_rates_dict().values,
                                            allow_unused=True
                                            )
                if any([False in torch.isfinite(g) for g in grads]):
                    print('Invalid LR gradients, zero out gradients')
                    for g in grads: g.zero_()
            for idx, key in enumerate(self.names_learning_rates_dict.keys()):
                self.names_learning_rates_dict[key] = nn.Parameter(
                    self.names_learning_rates_dict[key] - self.lr_of_lr * grads[idx])

    def update_params(self, names_weights_dict, grads, num_step):
        return OrderedDict(
            (
            key, names_weights_dict[key] - self.names_learning_rates_dict[key.replace(".", "-")][num_step] * grads[idx])
            for idx, key in enumerate(names_weights_dict.keys()))
        # for idx, key in enumerate(names_weights_dict.keys()):

        # return (
        #     (
        #     key, names_weights_dict[key] - self.names_learning_rates_dict[key.replace(".", "-")][num_step] * grads[idx])
        #     for idx, key in enumerate(names_weights_dict.keys()))


def functional_model_parameters_updates(model,weight):
      model_dict = model.state_dict()
      model_dict.update(weight)
      model.load_state_dict(model_dict)
      return model


def evaluation(args, model, eval_dataloader):
    eval_preds, eval_labels, eval_losses,eval_acc = [], [], [],[]
    tqdm_dataloader = tqdm(eval_dataloader)
    acc_fn =Accuracy()
    model.eval()
    with torch.no_grad():
        for x,y in tqdm_dataloader:
            outputs = model(x)
            loss = F.mse_loss(outputs,y)
            eval_preds.append(outputs)
            eval_labels.append(y)
            eval_losses.append(loss.item())
            acc = acc_fn(outputs,y,args.val_size)
            eval_acc.append(acc)
            tqdm_dataloader.set_description(" eval acc: %s" %(acc) )

    final_acc = eval_acc[-1]
    final_loss =np.mean(eval_losses)
    return final_acc,final_loss
 
def read_pkl_file(path):
    with open(path,'rb') as file:
        data =pkl.load(file)
    return data

if __name__ =='__main__':
    pass
# #第一阶段的特征提取
#     pa = '/Users/ansixu/final/Dataset/train_Dataset/train_text_feat/topic1'
#     model_face_encoding= self_feature_learn_BiLSTM(1024)
#     self_feature_learn_face=[]
#     file =os.listdir(pa)
#     # Load data from pickle file
#     for i in tqdm(file,desc='file_processing',unit='file'):
#         file_path = pa +r'/'+i
#         with open(file_path, 'rb') as file:
#             reshape_array_face_data =[]
#             data = pkl.load(file,encoding='latin1')
#             data_face = data['text_encoding_list']
#             for e in data_face:
#                 if type(e) is np.ndarray:
#                     e = e.reshape((1,1024))
#                     reshape_array_face_data.append(e)
#                 else:
#                     # e = e.reshape((1,24))
#                     reshape_array_face_data.append(e)
#         reshape_array_face_data = np.array(reshape_array_face_data)
#         data_face= torch.tensor(reshape_array_face_data, dtype=torch.float)

#         y =model_face_encoding(data_face)
#         # except:
#         #     print(i)
#         # 将该视频的特征存为pkl 文件
#         output_path ='/Users/ansixu/final/train_text_self_feature_bilstm/topic1/'
#         if not os.path.exists(output_path):
#             os.makedirs(output_path)
#         with open(output_path+i,'wb') as file:
#             pkl.dump(y,file)
# # 第二阶段中的单独模态的特征提取。
#     topics_list = ["topic1", "topic2", "topic3", "topic4", "topic5", 
#                    "topic6", "topic7", "topic8", "topic9", "topic10"]
#     model_self =self_attention(1024)
#     pa ='/Users/ansixu/final/train_text_self_feature_bilstm/'
#     for t in topics_list:
#         pa_2 =pa+t
#         file = os.listdir(pa_2)
#         for i in tqdm(file,desc='file_processing',unit='file'):
#             file_path = pa_2 +r'/'+i
#             with open(file_path, 'rb') as file:
#                 data = pkl.load(file,encoding='latin1')
#                 y =model_self(data)
#                 output_path ='/Users/ansixu/final/train_self_feature'+pa[-26:-21]+'_self_attention/'+t+'/'
#             if not os.path.exists(output_path):
#                 os.makedirs(output_path)
#             with open(output_path+i,'wb') as file:
#                 pkl.dump(y,file)
# 第三阶段中的单独模态的特征提取。
    # topics_list = ["topic1", "topic2", "topic3", "topic4", "topic5", 
    #                "topic6", "topic7", "topic8", "topic9", "topic10"]
    # model =Data_Fusion_Model(128,1024,24,1024)
    # pa_1 = '/Users/ansixu/final/train_self_feature_bilstm/train_face_self_feature_bilstm'
    # pa_2 = '/Users/ansixu/final/train_self_feature_bilstm/train_audio_self_feature_bilstm'
    # pa_3 = '/Users/ansixu/final/train_self_feature_bilstm/train_video_self_feature_bilstm'
    # pa_4 ='/Users/ansixu/final/train_self_feature_bilstm/train_text_self_feature_bilstm'

    # for topic in tqdm(topics_list,desc='topic process',unit='topic'):
    #     file_name=os.listdir(pa_1+'/'+topic)
    #     for file in tqdm(file_name,desc='file processing',unit='file'):
    #         # data_f =read_pkl_file(pa_1+'/'+topic+'/'+file)

    #         data_a =read_pkl_file(pa_2+'/'+topic+'/'+file)
    #         padding =(0,0,0,0,0,data_f.shape[0]-data_a.shape[0])
    #         data_a =F.pad(data_a, padding, 'constant', 0)

    #         data_v =read_pkl_file(pa_3+'/'+topic+'/'+file)
    #         data_v = torch.unsqueeze(data_v, dim=1)
    #         padding =(0,0,0,0,0,data_f.shape[0]-data_v.shape[0])
    #         data_v =F.pad(data_v, padding, 'constant', 0)

    #         data_t =read_pkl_file(pa_4+'/'+topic+'/'+file)
    #         padding =(0,0,0,0,0,data_f.shape[0]-data_t.shape[0])
    #         data_t =F.pad(data_t, padding, 'constant', 0)

    #         y =model(data_f,data_a,data_v,data_t)
    #         output_path ='/Users/ansixu/final/train_shared_feature/'+topic+'/'
    #         if not os.path.exists(output_path):
    #             os.makedirs(output_path)
    #         with open(output_path+file,'wb') as file:
    #             pkl.dump(y,file)
    # topics_list = ["topic1", "topic2", "topic3", "topic4", "topic5", 
    #                "topic6", "topic7", "topic8", "topic9", "topic10"]
    # model =Data_Fusion_Model(1024,1024,24,1024)
    # pa_1 = '/Users/ansixu/final/train_self_feature_bilstm/train_face_self_feature_bilstm'
    # pa_2 = '/Users/ansixu/final/train_self_feature_bilstm/train_audio_self_feature_bilstm'
    # pa_3 = '/Users/ansixu/final/train_self_feature_bilstm/train_video_self_feature_bilstm'
    # pa_4 ='/Users/ansixu/final/train_self_feature_bilstm/train_text_self_feature_bilstm'

    # for topic in tqdm(topics_list,desc='topic process',unit='topic'):
    #     file_name=os.listdir(pa_1+'/'+topic)
    #     for file in tqdm(file_name,desc='file processing',unit='file'):
    #         # data_f =read_pkl_file(pa_1+'/'+topic+'/'+file)

    #         data_a =read_pkl_file(pa_2+'/'+topic+'/'+file)

    #         data_t =read_pkl_file(pa_4+'/'+topic+'/'+file)
    #         padding =(0,0,0,0,0,data_a.shape[0]-data_t.shape[0])
    #         data_t =F.pad(data_t, padding, 'constant', 0)

    #         y =model(data_a,data_t)
    #         output_path ='/Users/ansixu/final/train_shared_feature_a_t/'+topic+'/'
    #         if not os.path.exists(output_path):
    #             os.makedirs(output_path)
    #         with open(output_path+file,'wb') as file:
    #             pkl.dump(y,file)

# 将获得的self feature 与shared feature 拉平融合
    # topics_list = ["topic1", "topic2", "topic3", "topic4", "topic5", 
    #                 "topic6", "topic7", "topic8", "topic9", "topic10"]
    # # topics_list =['topic6']
    # self_feature_path='/Users/ansixu/final/train_self_feature_a_t'
    # shared_feature_path='/Users/ansixu/final/train_shared_feature_a_t'
    
    # output_path='/Users/ansixu/final/new_extracted_feature_a_t/'
    # feature_flatten(self_feature_path,shared_feature_path,
    #                 output_path)
    

# 特征读取输入MLP 模型
    # 1.读取模型特征
    path ='/Users/ansixu/final/y_train_data.json'
    path_for_feature ='/Users/ansixu/final/new_extracted_feature_a_t'
    data_to_tensor(path,path_for_feature)
    y_label =['extraversion', 'neuroticism', 'agreeableness',
              'conscientiousness', 'interview', 'openness']
    #
    # acc_fn =Accuracy()
    # scores = torch.rand(6,5)
    # target =torch.rand(6,5)
    # acc =acc_fn(scores,target)
    # print(acc)