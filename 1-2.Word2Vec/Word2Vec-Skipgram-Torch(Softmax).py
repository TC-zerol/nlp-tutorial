'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn # torch的神经网络库
import torch.optim as optim # 是一个实现各种优化算法的包
from torch.autograd import Variable # 提供实现任意标量值函数的自动区分的类和函数
import matplotlib.pyplot as plt # Matplotlib 是 Python 的绘图库。

dtype = torch.FloatTensor # 创建一个浮点类型的CPUtensor，Torch.Tensor是一个包含单一数据类型元素的多维矩阵

# 3 Words Sentence
sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()
# ['i', 'like', 'dog', 'i', 'like', 'cat', 'i', 'like', 'animal', 'dog', 'cat',
# 'animal', 'apple', 'cat', 'dog', 'like', 'dog', 'fish', 'milk', 'like', 'dog',
# 'cat', 'eyes', 'like', 'i', 'like', 'apple', 'apple', 'i', 'hate', 'apple', 'i',
# 'movie', 'book', 'music', 'like', 'cat', 'dog', 'hate', 'cat', 'dog', 'like']
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
# {'music': 0, 'cat': 1, 'fish': 2, 'animal': 3, 'like': 4, 'book': 5, 'dog': 6,
# 'movie': 7, 'eyes': 8, 'i': 9, 'hate': 10, 'milk': 11, 'apple': 12}

# Word2Vec Parameter
batch_size = 20 # batch大小
embedding_size = 2  # To show 2 dim embedding graph  显示2维词嵌入图
voc_size = len(word_list)

def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False) # 从range(len(data))选取size个，不放回

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[data[i][0]])  # target   np.eye(n)[m]转换成列one-hot第m-1列为1
        random_labels.append(data[i][1])  # context word

    return random_inputs, random_labels

# Make skip gram of one size window  创建一个左右大小为1的滑动窗口，将一个词和它的左右组成对
skip_grams = []
for i in range(1, len(word_sequence) - 1):
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

    for w in context:
        skip_grams.append([target, w])

# Model
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()

        # W and WT is not Traspose relationship
        self.W = nn.Parameter(-2 * torch.rand(voc_size, embedding_size) + 1).type(dtype) # voc_size > embedding_size Weight
        #                                 13*2
        self.WT = nn.Parameter(-2 * torch.rand(embedding_size, voc_size) + 1).type(dtype) # embedding_size > voc_size Weight
        #                                 2*13

    def forward(self, X):
        # X : [batch_size, voc_size]  20*13
        hidden_layer = torch.matmul(X, self.W) # hidden_layer : [batch_size, embedding_size]
        output_layer = torch.matmul(hidden_layer, self.WT) # output_layer : [batch_size, voc_size]
        return output_layer # 20*13

model = Word2Vec() # 初始化模型

criterion = nn.CrossEntropyLoss() # 定义损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001) # 定义优化函数

# Training
for epoch in range(5000):

    input_batch, target_batch = random_batch(skip_grams, batch_size)

    input_batch = Variable(torch.Tensor(input_batch))
    target_batch = Variable(torch.LongTensor(target_batch))

    optimizer.zero_grad()
    output = model(input_batch)

    # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    # 自动求导反向传播，使用step()来更新参数
    loss.backward()
    optimizer.step()

W, WT = model.parameters() # 返回模块参数的迭代器。 这通常被传递给优化器。
# Parameter containing:
# tensor([[-1.0329,  0.2841],
#         [ 0.6558, -1.4412],
#         [ 1.0912,  0.0807],
#         [ 0.1451, -2.4376],
#         [ 1.1132, -0.7544],
#         [ 3.3368,  1.7286],
#         [-0.9126, -1.0992],
#         [ 0.8366,  0.5566],
#         [-1.6334, -2.2764],
#         [-0.2392, -1.2737],
#         [ 0.8066, -1.9696],
#         [-2.4500, -0.2954],
#         [ 0.3445, -1.4208]], requires_grad=True)

# Parameter containing:
# tensor([[ 1.1723, -0.6896, -1.4554,  0.2856, -1.1932, -0.2595,  1.2844,  1.0478,
#           0.7575,  1.4921, -0.2393,  0.8439,  0.3777],
#         [ 2.0184, -0.6579,  1.4127,  0.2516, -0.6905,  3.3368, -0.3101,  1.7744,
#           0.8838,  0.0417,  0.0671,  1.6964, -0.4147]], requires_grad=True)
for i, label in enumerate(word_list):
    x,y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y) # scatter散点图
    # plt.annotate()函数用于标注文字。s 为注释文本内容，xy 为被注释的坐标点，xytext 为注释文字的坐标位置
    #参考https://blog.csdn.net/TeFuirnever/article/details/88946088
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()
