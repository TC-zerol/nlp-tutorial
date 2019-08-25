'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

dtype = torch.FloatTensor

# Text-CNN Parameter
embedding_size = 2 # n-gram
sequence_length = 3
num_classes = 2  # 0 or 1
filter_sizes = [2, 2, 2] # n-gram window
num_filters = 3

# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
#['is', 'hate', 'she', 'this', 'love', 'he', 'likes', 'you', 'i',
# 'baseball', 'for', 'that', 'awful', 'sorry', 'loves', 'me']
word_dict = {w: i for i, w in enumerate(word_list)}
# {'is': 0, 'hate': 1, 'she': 2, 'this': 3, 'love': 4, 'he': 5, 'likes': 6,
# 'you': 7, 'i': 8, 'baseball': 9, 'for': 10, 'that': 11,
# 'awful': 12, 'sorry': 13, 'loves': 14, 'me': 15}
vocab_size = len(word_dict)

inputs = []
for sen in sentences:
    inputs.append(np.asarray([word_dict[n] for n in sen.split()]))
# [array([8, 4, 7]), array([ 5, 14, 15]), array([2, 6, 9]), array([8, 1, 7]),
# array([13, 10, 11]), array([ 3,  0, 12])]

targets = []
for out in labels:
    targets.append(out) # To using Torch Softmax Loss function
# [1, 1, 1, 0, 0, 0]

input_batch = Variable(torch.LongTensor(inputs))
target_batch = Variable(torch.LongTensor(targets))


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()

        self.num_filters_total = num_filters * len(filter_sizes) # 3*3
        # torch.empty使用均匀分布U(a,b)初始化Tensor，即Tensor的填充值是等概率的范围为 [a，b) 的值。均值为 （a + b）/ 2.
        self.W = nn.Parameter(torch.empty(vocab_size, embedding_size).uniform_(-1, 1)).type(dtype) # 16*2
        self.Weight = nn.Parameter(torch.empty(self.num_filters_total, num_classes).uniform_(-1, 1)).type(dtype)
        self.Bias = nn.Parameter(0.1 * torch.ones([num_classes])).type(dtype)

    def forward(self, X):
        embedded_chars = self.W[X] # [batch_size, sequence_length, embedding_size]  torch.Size([6, 3, 2])  [X]为列表截取
        embedded_chars = embedded_chars.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]

        pooled_outputs = []
        for filter_size in filter_sizes:
            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            # nn.Conv2d二维卷积层
            conv = nn.Conv2d(1, num_filters, (filter_size, embedding_size), bias=True)(embedded_chars) # torch.Size([6, 3, 2, 1]) (embedded_chars)是输入
            h = F.relu(conv)
            # mp : ((filter_height, filter_width))
            # nn.MaxPool2d2维最大池化
            mp = nn.MaxPool2d((sequence_length - filter_size + 1, 1)) #
            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 3, 2, 1) #将tensor的维度换位
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(filter_sizes)) # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        # torch.cat按第二个参数维数拼接
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total]) # [batch_size(=6), output_height * output_width * (output_channel * 3)]

        model = torch.mm(h_pool_flat, self.Weight) + self.Bias # [batch_size, num_classes] orch.mm(a, b)是矩阵a和b矩阵相乘
        return model

model = TextCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5000):
    optimizer.zero_grad()
    output = model(input_batch)

    # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Test
test_text = 'sorry hate you'
tests = [np.asarray([word_dict[n] for n in test_text.split()])]
test_batch = Variable(torch.LongTensor(tests))

# Predict
predict = model(test_batch).data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")