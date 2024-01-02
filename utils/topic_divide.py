from config import*
import torch

# 该函数用于获取annotation.json 中train，val,test 的数据集中的text，
# 并获取对应不同视频文件的y
json_name ='/Users/ansixu/Personality_Project/annotation.json'
y_output_json_file ='/Users/ansixu/Personality_Project/y_train_data.json'
text_output_json_file ='/Users/ansixu/Personality_Project/text_train_data.json'
f = open(json_name, 'r')
content = f.read()
a = json.loads(content)

k = a['train']
text_train_data ={}
y_train_data ={}
for i in k.keys():
    d ={}
    for j in k[i].keys():
        if j =='text':
            text_train_data.update({i:k[i][j]})
        if j == 'openness':
            o = {j:k[i][j]}
            d.update(o)
        if j =='conscientiousness':
            c ={j:k[i][j]}
            d.update(c)
        if j =='extraversion':
            e ={j:k[i][j]}
            d.update(e)
        if j =='agreeableness':
            a ={j:k[i][j]}
            d.update(a)
        if j =='neuroticism':
            n ={j:k[i][j]}
            d.update(n)
    y_train_data.update({i:d})


with open(y_output_json_file,'w') as json_file:
    json.dump(y_train_data,json_file)

with open(text_output_json_file,'w') as json_file_1:
    json.dump(text_train_data,json_file_1)

# 该函数用于将数据集按照划分的topic，存储在一个文件夹内

def organize_files(json_file_path, destination_folder,file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Create folders for each class
    for k,v in data.items():
        class_label = v['class']  # Adjust this based on the actual structure of your JSON file
        class_folder = os.path.join(destination_folder, class_label)
        filename =k

        # Create the folder if it doesn't exist
        os.makedirs(class_folder, exist_ok=True)

        # Move or copy the file to the class folder
        file_path_copy = file_path+'/'+filename[:-3]+'pkl' # Adjust this based on the actual structure of your JSON file
        shutil.copy(file_path_copy, class_folder)  # Use shutil.copy if you want to copy instead of move

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
    


            