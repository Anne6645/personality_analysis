import pickle as pkl


def get_topic_dataset(args, dataset,path):
        assert dataset.lower() in ["topic0","topic1", "topic2", "topic3", "topic4", "topic5", 
            "topic6", "topic7", "topic8", "topic9", "topic10","topic11","topic12","topic13","topic14","topic15","topic16","topic17","topic18","topic19"]
        return load_data(args,'train',dataset,path)
  
class GetLoader(torch.utils.data.Dataset):

    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index-1]
        labels = self.label[index-1]
        return data, labels

    def __len__(self):
        return len(self.data)

  class load_data(AbstractDataset):
    @classmethod
    def code(cls):
        return 'data'
    def new_make_train_val_test_split(self,length,val=None, test=None):
        split_path = self.get_train_val_test_split_path()
        indices = list(range(length-1))
        random.shuffle(indices)
        split = {
            'train': indices[:1],
            'val': indices[1:],
            }
        with open(split_path, 'w') as f:
            json.dump(split, f)
        return split
    def load_topic_pkl_file(self):
        x_path =self.path +'/'+'x_tensor_all'+'/'+self.dataset+'_x_tensor_all.pkl'
        y_path =self.path +'/'+'y_tensor_all'+'/'+self.dataset+'_y_tensor_all.pkl'
        with open(x_path,'rb') as file_x:
            x_data =pkl.load(file_x)
        with open(y_path,'rb') as file_y:
            y_data =pkl.load(file_y)

        split = self.new_make_train_val_test_split(len(x_data)-1)
        #1.train
        train_indices = split['train']
        train_x_data =x_data[train_indices]
        train_y_data =y_data[train_indices]
        train_torch_data =GetLoader(train_x_data,train_y_data)
        # 1.val
        val_indices = split['val']
        val_x_data =x_data[val_indices]
        val_y_data =y_data[val_indices]
        val_torch_data =GetLoader(val_x_data,val_y_data)
        return train_torch_data,val_torch_data
