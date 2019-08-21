# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn # torch的神经网络库
import torch.optim as optim # 是一个实现各种优化算法的包
from torch.autograd import Variable #提供实现任意标量值函数的自动区分的类和函数。

dtype = torch.FloatTensor # 创建一个浮点类型的CPUtensor，Torch.Tensor是一个包含单一数据类型元素的多维矩阵

sentences = [ "i like dog", "i love coffee", "i hate milk"] # 初始化三个句子

word_list = " ".join(sentences).split() # split() 方法用于把一个字符串分割成字符串数组。
# join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。
# ['i', 'like', 'dog', 'i', 'love', 'coffee', 'i', 'hate', 'milk']

word_list = list(set(word_list)) #先让list变成set，然后再变回去，去重
# ['dog', 'milk', 'i', 'like', 'hate', 'love', 'coffee']

# 使用了enumerate函数得到字典
word_dict = {w: i for i, w in enumerate(word_list)} # {'dog': 0, 'milk': 1, 'i': 2, 'like': 3, 'hate': 4, 'love': 5, 'coffee': 6}
number_dict = {i: w for i, w in enumerate(word_list)} # {0: 'dog', 1: 'milk', 2: 'i', 3: 'like', 4: 'hate', 5: 'love', 6: 'coffee'}

n_class = len(word_dict) # number of Vocabulary

# NNLM Parameter
n_step = 2 # n-1 in paper 根据前两个单词预测第三个单词
n_hidden = 2 # h in paper 隐藏层神经元个数
m = 2 # m in paper 词向量维数

def make_batch(sentences):
    input_batch = [] # 输入batch
    target_batch = [] # 目标batch

    for sen in sentences: # 对sentences中每个句子
        word = sen.split() # 按空格分割
        input = [word_dict[n] for n in word[:-1]] # 直到倒数第二个
        target = word_dict[word[-1]] # 最后一个

        # 添加分割好的词
        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch

# Model
class NNLM(nn.Module): # 定义NNLM网络，继承nn.Module
    def __init__(self): # 构造函数
        super(NNLM, self).__init__() # 父类构造函数
        # 参数都是论文中的数学表示
        # 以下是设置神经网络中的各项参数
        # 一个嵌入字典，第一个参数是嵌入字典的大小，第二个参数是每个嵌入向量的大小
        # C词向量C(w)存在于矩阵C(|V|*m)中，矩阵C的行数表示词汇表的大小；列数表示词向量C(w)的维度。矩阵C的某一行对应一个单词的词向量表示
        self.C = nn.Embedding(n_class, m)
        # Parameter类是Variable的子类，常用于模块参数，作为属性时会被自动加入到参数列表中
        # 隐藏层的权重(h*(n-1)m)
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype)) # torch.randn正态分布 4*2
        # 输入层到输出层权重(|V|*(n-1)m)
        self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype)) # 4*5
        # 隐藏层偏置bias(h)
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype)) # 2
        # 隐藏层到输出层的权重(|V|*h)
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype)) # 2*5
        # 输出层的偏置bias(|V|)
        self.b = nn.Parameter(torch.randn(n_class).type(dtype)) #5

    # 前向传播
    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, n_step * m) # [batch_size, n_step * n_class]  转换成 -1*4
        tanh = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden] torch.mm:Performs a matrix multiplication of the matrices input and mat2.
        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U) # [batch_size, n_class]
        #          5             -1*4, 4*5               -1*2, 2*5
        return output

model = NNLM() # 初始化模型

# 损失函数定义为交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 采用Adam优化算法，学习率0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 以下三行将输入进行torch包装，用Variable可以实现自动求导
input_batch, target_batch = make_batch(sentences)
input_batch = Variable(torch.LongTensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))

# Training训练过程，5000轮
for epoch in range(5000):

    optimizer.zero_grad() #清空梯度
    output = model(input_batch)

    # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1)%1000 == 0:  # 每1000轮查看一次损失函数变化
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    # 自动求导反向传播，使用step()来更新参数
    loss.backward()
    optimizer.step()

# Predict 预测值
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test 测试
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
#
#Epoch: 1000 cost = 0.133690
#Epoch: 2000 cost = 0.021938
#Epoch: 3000 cost = 0.007234
#Epoch: 4000 cost = 0.003100
#Epoch: 5000 cost = 0.001504
#[['i', 'like'], ['i', 'love'], ['i', 'hate']] -> ['dog', 'coffee', 'milk']
