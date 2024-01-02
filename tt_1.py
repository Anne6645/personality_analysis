import pickle as pkl
x_path ='/Users/ansixu/final/topic5_y_tensor_a_t.pkl'
y_path ='/Users/ansixu/final/topic5_x_tensor_a_t.pkl'

with open(x_path,'rb') as file_x:
    x_data =pkl.load(file_x)


with open(y_path,'rb') as file_y:
    y_data =pkl.load(file_y)

print(x_data.shape,y_data.shape)