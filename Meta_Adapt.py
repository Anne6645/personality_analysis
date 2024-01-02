from tqdm import tqdm
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing
from transformers.optimization import AdamW
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

import random
import onnxruntime
from config import*
import model.Model as Model
import pickle as pkl
from sklearn.preprocessing import StandardScaler
import copy
from model.Model import evaluation
from datetime import datetime
from abc import ABC, abstractmethod
from torch.onnx import TrainingMode
from pathlib import Path

class GetLoader(torch.utils.data.Dataset):
# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
        # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index-1]
        labels = self.label[index-1]
        return data, labels
# 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
    
def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AbstractDataset():
    def __init__(self, args, mode, dataset,path):
        self.args = args
        self.mode = mode
        self.path = path
        self.dataset =dataset

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def load_dataset(self):
        pass

    def get_train_val_test_split_path(self):
        return os.path.join(self.path,
        self.code()+'_'+str(self.mode)+'_'+str(self.args.val_size)+'_'+str(self.args.test_size)+'_split.json')

    def get_train_val_test_split(self):
        split_path = self.get_train_val_test_split_path()
        return json.loads(open(split_path).read())

    def make_train_val_test_split(self, length, val=None, test=None):
        if val is None:
            val = self.args.val_size
        if test is None:
            test = self.args.test_size
        split_path = self.get_train_val_test_split_path()
        indices = list(range(length-1))
        random.shuffle(indices)
        split = {
            'train': indices[:int(length*(1-val-test))],
            'val': indices[int(length*(1-val-test)):int(length*(1-test))],
            'test': indices[int(length*(1-test)):]
            }
        with open(split_path, 'w') as f:
            json.dump(split, f)
        return split


class load_topic_data(AbstractDataset):

    @classmethod
    def code(cls):
        return 'data'

    def load_topic_pkl_file(self):
        x_path =self.path +'/'+'x_tensor_a_t'+'/'+self.dataset+'_x_tensor_a_t.pkl'
        y_path =self.path +'/'+'y_tensor_a_t'+'/'+self.dataset+'_y_tensor_a_t.pkl'
        with open(x_path,'rb') as file_x:
            x_data =pkl.load(file_x)
        with open(y_path,'rb') as file_y:
            y_data =pkl.load(file_y)
        # split_file = self.get_train_val_test_split_path()

        split = self.make_train_val_test_split(len(x_data)-1)
        indices = split[self.mode]
        x_data =x_data[indices]
        y_data =y_data[indices]
        torch_data =GetLoader(x_data,y_data)
        return torch_data

def get_topic_dataset(args, mode, dataset,path):
        assert dataset.lower() in ["topic1", "topic2", "topic3", "topic4", "topic5", 
            "topic6", "topic7", "topic8", "topic9", "topic10"]
        assert mode in ['train', 'val', 'test']
        return load_topic_data(args,mode,dataset,path)

def adapt(args):
    """
:param num_iterations: adaption iteration 的次数
:param num_pi:number of subtasks:
Number of update steps for each subtask (inner update)
:return:
    """
    fix_random_seed_as(args.seed)

# 输出文件夹定义
    if not args.output_dir:
        args.output_dir = datetime.now().strftime("%Y%m%d%H%M%S")
        export_root = os.path.join(EXPERIMENT_ROOT_FOLDER, args.output_dir)
    if not os.path.exists(export_root):
        os.makedirs(export_root)

    # 数据准备阶段：
    #1. 将提取的特征和标签数据存成pkl 文件，并读取
    #2. 将x,y 数据输入到Getloader 形成可以给dataloader 识别的dataset
    train_dataset =get_topic_dataset(args,'train',args.source_topic,args.path).load_topic_pkl_file()
    # val_dataset =get_topic_dataset(args,'val',args.target_topic)

    # 3. 将形成的dataset 输入到DataLoader 中
    train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=args.train_batchsize
    )
# 4. 创建迭代器遍历训练数据
    train_iter = iter(train_dataloader)

# 准备query data
    query_dataset = get_topic_dataset(args,'train',args.target_topic,args.path).load_topic_pkl_file()
    val_dataset =get_topic_dataset(args,'val',args.target_topic,args.path).load_topic_pkl_file()
    test_dataset =get_topic_dataset(args,'test',args.target_topic,args.path).load_topic_pkl_file()
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

# 准备test data
    test_dataloader = DataLoader(
    test_dataset, 
    sampler=RandomSampler(test_dataset),
    batch_size=args.eval_batchsize
)

    # 创建模型阶段：
# 1. 读取之前训练好的MLP 模型文件
    model =torch.jit.load('/Users/ansixu/final/trained_model.pt')
# 将读取的数据输入进模型训练获取结果

# 优化器配置：
# 初始化模型迭代方法为第一层迭代，内部根据之前的存取的模型参数进行迭代更新
    inner_loop_optimizer = Model.LSLRGradientDescentLearningRule(
    total_num_inner_loop_steps=args.num_updates,
    init_learning_rate=args.learning_rate_learner,
    use_learnable_lr=True,
    lr_of_lr=args.learning_rate_lr)
    inner_loop_optimizer.initialize(names_weight_dict=model.named_parameters())
# 模型优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate_meta)
# 自动调整学习率：
# 将每个参数组的学习率设置为初始lr乘以给定函数。
# 当last_epoch=-1时，将初始lr设置为lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
    T_max=args.num_iterations,
    eta_min=args.learning_rate_meta * args.lr_decay_meta)
    print('***** Running adaptation *****')
    print('***** No adaption eval on target result *****')
# 模型迁移：计算不同任务之间的similarity
# GradScaler 可以将梯度缩放到较小的范围，
#以避免数值下溢或溢出的问题，同时保持足够的精度以避免模型的性能下降
# scaler = torch.cuda.amp.GradScaler()
    trange = tqdm(range(args.num_iterations))
    losses =[]
# 输出的是当模型没有做Meta adaptive 时候的准确率用于与Meta adaptive 做对比
    best_acc, _ = Model.evaluation(args, model, val_dataloader)
# 模型参数备份用于模型更新
    model_i =copy.deepcopy(model)
    for i in trange: # total number of adaption iteration
        model.train()
        mean_outer_loss = 0
        meta_gradients = []
        domain_similarity = []
        for j in range(args.num_pi):
# num_pi 是domain的 数量
            support=tuple(t for t in next(train_iter))
# 从原始参数开始
            fast_weights = collections.OrderedDict(model_i.named_parameters())
            weight_before =fast_weights
# fast_weights =model_i.named_parameters()
            for k in range(args.num_updates):
                x,y =support
    # 将第一个domain 的训练集融合数据输入到MLP模型中，
    # 模型返回的outputs是预测的五大性格的值
                # for x,y in train_dataloader:
                #print(y)
                outputs = model_i(x)
                #print(outputs)
    # 根据模型预测的y 计算loss
                loss = F.mse_loss(y,outputs)
                loss.requires_grad_=True
    #print(loss)
    # 模型梯度归零
                model.zero_grad()
                fast_weights = collections.OrderedDict(model_i.named_parameters())
# print(fast_weights.values())
# 将计算的loss 根据模型参数scale，并计算梯度
                scaled_grads = torch.autograd.grad(loss, fast_weights.values(),
                create_graph=True, retain_graph=True,
                allow_unused=True)
# inv_scale = 1. / scaler.get_scale()
# print(scaled_grads)
                grads = [p for p in scaled_grads]
# model.step()
# print(len(grads))

# if any([False in torch.isfinite(g) for g in grads]):
                fast_weights = inner_loop_optimizer.update_params(fast_weights,grads,k)
                model_i =Model.functional_model_parameters_updates(model_i,weight=fast_weights)
    # 计算domain 梯度更新,计算更新后的weight 与原来的梯度
        domain_gradients=tuple()
# weight_before =collections.OrderedDict(model.named_parameters())
        for _,(params_before,params_after) in enumerate(zip(weight_before,fast_weights)):
            domain_gradients +=(fast_weights[params_before].detach()-weight_before[params_before].detach(),)

# 在target domain 中选择合适query
        query_loss =0.
        for x,y in query_loader:
# 在target domain中 作为query，
# 然后将query 输入到之前训练好的模型中
            model_i = Model.functional_model_parameters_updates(model_i,fast_weights)
# 模型对query进行预测
            outputs = model_i(x)
# 根据输出计算query loss
            query_loss = F.mse_loss(outputs,y)
# query_loss.autograd.set_detect_anomaly(True)

# query loss 做累加除以数据长度做平均
#         query_loss/=len(query_loader)
# 用query loss 在模型参数上求梯度作为meta grad
        scaled_meta_grads = torch.autograd.grad(query_loss,
                            model_i.parameters(), retain_graph=True)
        meta_gradients.append(scaled_meta_grads)

# 计算 Domain similarity: normalize domain and meta gradients before computing Domain similarity
    cur_similarity = [F.cosine_similarity(a.view(-1), b.view(-1), dim=-1) for a, b in
                    zip(domain_gradients, scaled_meta_grads)]
    cur_similarity = torch.mean(torch.tensor(cur_similarity))
    domain_similarity.append(cur_similarity)
# 更新学习率：use meta loss to update learnable lr
# 计算梯度
    

    domain_similarity = F.softmax(torch.tensor(domain_similarity) / args.softmax_temp, -1)
    for weights in zip(model_i.parameters(), *meta_gradients):
        for k in range(len(meta_gradients)):
            if k == 0:
                weights[0].grad = domain_similarity[k] * weights[k+1]
            else:
                weights[0].grad += domain_similarity[k] * weights[k+1]
# meta gradient 是scale的，做恢复
# scaler.unscale_(optimizer)
# torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
# scaler.step(optimizer)
# scaler.update()

    optimizer.zero_grad()
    scheduler.step()

# sometimes scale becomes too small
# if scaler.get_scale() < 1.:
# scaler.update(1.)

    mean_outer_loss /= args.num_pi
    losses.append(mean_outer_loss)
    trange.set_description('Meta loss, lr: {:.4f}, {:.8f}'.format(np.array(losses).mean(),
    scheduler.get_last_lr()[-1]))
# 获取最好的模型并存储
    if (i+1) % args.eval_interval == 0:
        acc,_ = Model.evaluation(args, model_i, val_dataloader)
        if acc > best_acc:
            best_acc = acc
    print()
    print('***** Saving best model *****')
    example = torch.rand(256, 5000)
    traced_script_module = torch.jit.trace(model_i, example)
    export_root=Path(args.export_root)
    traced_script_module.save(export_root/'best.pt')

    torch.onnx.export(
        model=model_i,  # 给定模型对象
        args=example,  # 给定模型forward的输出参数
        f=args.export_root+'/'+'best.onnx',  # 输出文件名称
        training=TrainingMode.EVAL,  # 训练还是eval阶段
        input_names=['features'],  # 给定输入的tensor名称列表
        output_names=['label'],  # 给定输出的tensor名称列表
        opset_version=12,
        dynamic_axes=None  # 给定是否是动态结构
    )
    torch.onnx.export(
        model=model_i,  # 给定模型对象
        args=example,  # 给定模型forward的输出参数
        f=args.export_root+'/'+'best_dynamic.onnx',  # 输出文件名称
        training=TrainingMode.EVAL,  # 训练还是eval阶段
        input_names=['features'],  # 给定输入的tensor名称列表
        output_names=['label'],  # 给定输出的tensor名称列表
        opset_version=12,
        dynamic_axes={
            'features': {
                0: 'batch'
            },
            'label': {
                0: 'batch'
            }
        }  # 给定是否是动态结构
    )
    print('***** Running evaluation *****')
    print('***** adaption training model eval on target result *****')
# 将保存的模型做恢复
    net =torch.jit.load(export_root/'best.pt')
    test_acc,_ =evaluation(args, net, test_dataloader)
    result = {
    'acc': test_acc
    }
    print('Result', result)
    with open(os.path.join(export_root, 'test_metrics.json'), 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    adapt(args)
