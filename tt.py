import config
import model.Model as Model
import torch
import pickle as pkl
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.onnx import TrainingMode
import numpy as np
from pathlib import Path
import tensorflow as tf
from model.Model import Accuracy
 from datetime import datetime

def export(model_dir):
    """
    NOTE: 可以通过netron（https://netron.app/）来查看网络结构
    将训练好的模型转换成可以支持多平台部署的结构，常用的结构：
    pt: Torch框架跨语言部署的结构
    onnx: 一种比较通用的深度学习模型框架结构
    tensorRT: 先转换成onnx，然后再进行转换使用TensorRT进行GPU加速
    openvino: 先转换为onnx，然后再进行转换使用OpenVINO进行GPU加速
    :param model_path:
    :return:
    """
    model_dir = Path(model_dir)
    # 模型恢复
    net = torch.load(model_dir / 'best.pkl', map_location='cpu')['net']
    net.eval().cpu()

    # 模型转换为pt结构
    example = torch.rand(1, 4)
    traced_script_module = torch.jit.trace(net, example)
    traced_script_module.save(model_dir / 'best.pt')


    # 转换为onnx结构
    torch.onnx.export(
        model=net,  # 给定模型对象
        args=example,  # 给定模型forward的输出参数
        f=model_dir / 'best.onnx',  # 输出文件名称
        training=TrainingMode.EVAL,  # 训练还是eval阶段
        input_names=['features'],  # 给定输入的tensor名称列表
        output_names=['label'],  # 给定输出的tensor名称列表
        opset_version=12,
        dynamic_axes=None  # 给定是否是动态结构
    )
    torch.onnx.export(
        model=net,  # 给定模型对象
        args=example,  # 给定模型forward的输出参数
        f=model_dir / 'best_dynamic.onnx',  # 输出文件名称
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

def load(path, net):
    print(f"模型恢复:{path}")
    ss_model = torch.load(path, map_location='cpu')
    net.load_state_dict(state_dict=ss_model['net'].state_dict(), strict=True)
    start_epoch = ss_model['epoch'] + 1
    best_acc = ss_model['acc']
    train_batch = ss_model['train_batch']
    test_batch = ss_model['test_batch']
    return start_epoch, best_acc, train_batch, test_batch
 
def training(x,y,summary_dir):
    # 读取训练时间
    now = datetime.now().strftime("%y%m%d%H%M%S")

    # 设置输出路径
    root_dir = Path(f'./output/01/{train_start_time}')

    # 设置模型总结文件夹
    summary_dir = root_dir / 'summary'
    if not summary_dir.exists():
        summary_dir.mkdir(parents=True)
    checkout_dir = root_dir / 'model'
    if not checkout_dir.exists():
        checkout_dir.mkdir(parents=True)
    last_path = checkout_dir / 'last.pkl'
    best_path = checkout_dir / 'best.pkl'

    # 总的训练次数 参数设置
    total_epoch = 100
    # 训练总结的训练间隔
    summary_interval_batch = 20
    # 存储间隔
    save_interval_epoch = 2

    start_epoch = 0
    best_acc = -1.0
    train_batch = 0
    test_batch = 0

    # 模型可视化
    writer = SummaryWriter(log_dir=summary_dir)
    # writer.add_graph(net)
    # 模型前向过程
    loss_fn = nn.MSELoss()
    acc_fn = Accuracy()
    opt = optim.SGD(params=net.parameters(), lr=0.005)
   
    # 1.读取数据
    torch_data =GetLoader(x,y)
    datas =DataLoader(torch_data, batch_size=6, shuffle=True, drop_last=False)

    #2.定义模型
    input_size =5000
    net =Model.Prediction_Model(input_size)

    #3. 模型恢复
    if best_path.exists():
        start_epoch, best_acc, train_batch, test_batch = load(best_path, net)
    elif last_path.exists():
        start_epoch, best_acc, train_batch, test_batch = load(last_path, net)

    # 遍历训练模型
    for epoch in range(start_epoch, total_epoch + start_epoch):
        # 5.1 训练
        net.train()
        train_losses = []
        train_true, train_total = 0, 0
        i=0
        n=5
        for x,y in datas:
            # 前向过程
            scores = net(x)  # [5] 得到的是每个样本属于5个类别的置信度
            loss = loss_fn(scores, y)
            acc = acc_fn(scores,y)
            # n, acc = acc_fn(scores, y_s)
            # print(acc)
            
            # 反向过程
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss = loss.item()
            # acc = acc.item()
            train_total += n
            # train_true += n * acc
            if train_batch % summary_interval_batch == 0:
                print(f"epoch:{epoch}, train batch:{train_batch}, loss:{loss:.3f}", acc)
                # writer.add_scalar('train_loss', loss, global_step=train_batch)
                # writer.add_scalar('train_acc', acc, global_step=train_batch)
            train_batch += 1
            train_losses.append(loss)
        # 模型存储
    example = torch.rand(256, 5000)
    traced_script_module = torch.jit.trace(net, example)
    traced_script_module.save('/Users/ansixu/final/trained_model.pt')

    torch.onnx.export(
        model=net,  # 给定模型对象
        args=example,  # 给定模型forward的输出参数
        f='/Users/ansixu/final/best.onnx',  # 输出文件名称
        training=TrainingMode.EVAL,  # 训练还是eval阶段
        input_names=['features'],  # 给定输入的tensor名称列表
        output_names=['label'],  # 给定输出的tensor名称列表
        opset_version=12,
        dynamic_axes=None  # 给定是否是动态结构
    )
    torch.onnx.export(
        model=net,  # 给定模型对象
        args=example,  # 给定模型forward的输出参数
        f='/Users/ansixu/final/best_dynamic.onnx',  # 输出文件名称
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
            

if __name__ =='__main__':
    x_path ='/Users/ansixu/final/topic1_x_tensor_a_t.pkl'
    y_path ='/Users/ansixu/final/topic1_y_tensor_a_t.pkl'
    summary_dir ='/Users/ansixu/final'
    with open(x_path,'rb') as file_x:
        x_data =pkl.load(file_x)
    with open(y_path,'rb') as file_y:
        y_data =pkl.load(file_y)

    training(x_data,y_data,summary_dir)