import re
import jieba
import torch
from torchtext.legacy import data
from torchtext.vocab import Vectors
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Jieba分词器
def Tokenizer(text):
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    text = regex.sub(' ', text)  # 过滤掉所有的标点符号(将标点符号替换为空格)
    return [word for word in jieba.cut(text) if word.strip()]


# 将标签转换成One-hot向量
def OneHotLabel(label):
    label = list(map(int, label.split('-')))  # string -> list
    label = torch.Tensor(label)  # list -> Tensor
    return label


# 停用词表
def GetStopWords():
    stop_words = []
    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stop_words.append(line.strip())
    return stop_words


class MultiLabelDataSet():

    def __init__(self, mode):
        self.mode = mode  # static:使用预训练词向量  rand:随机初始化
        stop_words = GetStopWords()  # 停用词表

        # 文本预处理配置
        self.text = data.Field(sequential=True, tokenize=Tokenizer, stop_words=stop_words)
        self.label = data.Field(sequential=True, tokenize=OneHotLabel, use_vocab=False)

        self.train_set, self.valid_set = data.TabularDataset.splits(
            path='./data/',
            skip_header=True,
            train='multi_label_train.csv',
            validation='multi_label_valid.csv',
            format='csv',
            fields=[('label', self.label), ('text', self.text)]
        )

        if self.mode == 'static' or self.mode == 'not-static':  # 使用预训练词向量
            cache = './data/.vector_cache'  # 指定缓存文件的目录位置
            vectors = Vectors(name='./pretrain_models/sgns.zhihu.word', cache=cache)  # 知乎问答预训练词向量
            self.text.build_vocab(self.train_set, self.valid_set, vectors=vectors)  # 利用预训练模型建立词典
            self.embedding_dim = self.text.vocab.vectors.size()[-1]  # 单个词向量维度
            self.vectors = self.text.vocab.vectors  # 预训练的词向量(权重矩阵)
        else:
            self.text.build_vocab(self.train_set, self.valid_set)  # 建立词典
            self.embedding_dim = 300  # 词向量维度为300
            self.vectors = None

        # self.label.build_vocab(self.train_set, self.valid_set)
        self.vocab_num = len(self.text.vocab)  # 单词个数
        self.label_num = 18  # 标签个数

    # 建立迭代器(batch化)
    def GetIter(self):
        train_iter, val_iter = data.Iterator.splits(
            (self.train_set, self.valid_set),
            sort_key=lambda x: len(x.text),
            # sort=False,
            batch_sizes=(BATCH_SIZE, len(self.valid_set)),
            device=0 if torch.cuda.is_available() else -1
        )
        return train_iter, val_iter

    # 获取参数
    def GetArgs(self):
        args = (self.mode, self.vocab_num, self.label_num, self.embedding_dim, self.vectors)
        return args


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()

        # 模型参数
        self.mode = args[0]
        self.vocab_num = args[1]  # 词汇个数
        self.label_num = args[2]  # 标签个数
        self.embedding_dim = args[3]  # 词向量维度
        self.vectors = args[4]  # 预训练词向量
        self.kernel_size = [2, 3, 4]  # 卷积核大小
        self.kernel_num = 100  # 卷积核个数(输出通道数)

        # 网络结构
        self.embedding = nn.Embedding(self.vocab_num, self.embedding_dim)
        if self.mode == 'static':  # 使用预训练词向量
            self.embedding = self.embedding.from_pretrained(self.vectors, freeze=True)  # 训练中无需微调
        elif self.mode == 'not-static':
            self.embedding = self.embedding.from_pretrained(self.vectors, freeze=False)  # 训练中微调

        self.convs = nn.ModuleList()  # 多个一维卷积层
        for _size in self.kernel_size:  # 每种大小的卷积核(3,4,5)各kernel_num(100)个
            self.convs.append(
                nn.Conv1d(
                    in_channels=self.embedding_dim,
                    out_channels=self.kernel_num,
                    kernel_size=_size)
            )

        self.dropout = nn.Dropout(p=0.5)
        # feature vec个数 = 卷积核种类*该类卷积核个数
        self.linear = nn.Linear(len(self.kernel_size) * self.kernel_num, self.label_num)

    def forward(self, sentence):
        # sentence: (batch_size, max_len)
        # 词嵌入
        embedding = self.embedding(sentence)  # embedding: (batch_size, max_len, embedding_dim)

        # 卷积核的尺寸为: kernel_size * in_channels, 在最后一个维度上进行卷积运算, 所以需要调整维度
        embedding = embedding.permute(0, 2, 1)  # 维度调整: (batch_size, embedding_dim, max_len)

        # 卷积得到feature_maps: (batch_size, output_channel, max_len-kernel_size+1)
        feature_maps = [F.relu(conv(embedding)) for conv in self.convs]

        # 各feature map分别进行最大池化得到univariate_vec: (batch_size, output_channel, 1)
        univariate_vecs = [F.max_pool1d(input=feature_map, kernel_size=feature_map.shape[2]) for feature_map in
                           feature_maps]

        # 在第二个维度上进行拼接: (batch_size, 3*output_channel, 1)
        univariate_vecs = torch.cat(univariate_vecs, dim=1)

        # 拼接成特征向量: (batch_size, 3*output_channel)
        feature_vec = univariate_vecs.view(-1, univariate_vecs.shape[1])

        # dropout
        feature_vec = self.dropout(feature_vec)

        # 全连接输出 (batch_size, 3*output_channel) -> (3*output_channel, label_num)
        output = self.linear(feature_vec)
        return output


class TextRCNN(nn.Module):
    def __init__(self, args):
        super(TextRCNN, self).__init__()

        # 模型参数
        self.mode = args[0]
        self.vocab_num = args[1]  # 词汇个数
        self.label_num = args[2]  # 标签个数
        self.embedding_dim = args[3]  # 词向量维度
        self.vectors = args[4]  # 预训练词向量
        self.hidden_size = 256  # lstm隐藏层个数
        self.num_layers = 1  # lstm层数

        # 网络结构
        self.embedding = nn.Embedding(self.vocab_num, self.embedding_dim)
        if self.mode == 'static':  # 使用预训练词向量
            self.embedding = self.embedding.from_pretrained(self.vectors, freeze=True)  # 训练中无需微调
        elif self.mode == 'not-static':
            self.embedding = self.embedding.from_pretrained(self.vectors, freeze=False)  # 训练中微调
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )

        self.linear = nn.Linear(self.embedding_dim + self.hidden_size * 2, self.label_num)

    def forward(self, sentence):
        # sentence: (batch_size, max_len)
        # 词嵌入
        embedding = self.embedding(sentence)  # embedding: (batch_size, max_len, embedding_dim)

        # bi-lstm
        context_info, (c, h) = self.lstm(embedding)  # context_info: (batch_size, max_len, hidden_size*2)

        # 拼接词嵌入和上下文信息得到词表征: (batch_size, max_len, hidden_size*2+embedding_dim)
        context_info_chunks = torch.chunk(context_info, chunks=2, dim=2)  # 切割得到左右文本信息
        context_left = context_info_chunks[0]  # 左文本信息
        context_right = context_info_chunks[1]  # 右文本信息
        representation = torch.cat((context_left, embedding), 2)
        representation = torch.cat((representation, context_right), 2)  # [cl,w,cr]
        representation = F.tanh(representation)  # 非线性激活

        # 维度变换, 以进行池化操作: (batch_size, hidden_size*2+embedding_dim, max_len)
        representation = representation.permute(0, 2, 1)

        # max_pooling得到特征向量
        # feature_vec: (batch_size, hidden_size*2+embedding_dim, 1)
        feature_vec = F.max_pool1d(input=representation, kernel_size=representation.shape[-1])
        feature_vec = feature_vec.squeeze(-1)  # ... -> (batch_size, hidden_size*2+embedding_dim)

        # 全连接输出
        output = self.linear(feature_vec)  # (batch_size, hidden_size*2+embedding_dim) -> (batch_size, label_num)

        return output


# 输入句子进行测试
def Classify(sentence):
    # 加载数据集
    print('load data...')
    data_set = MultiLabelDataSet('static')
    print('load model...')
    # textcnn = torch.load('./pretrain_models/textcnn_multi_label_80.pth')
    # textrcnn = torch.load('./pretrain_models/textrcnn_multi_label_80.pth')

    args = data_set.GetArgs()  # 获取模型所需参数
    textcnn = TextCNN(args)
    textcnn.load_state_dict(torch.load('./pretrain_models/textcnn_params_multi_label_80.pth', map_location='cpu'))
    textrcnn = TextRCNN(args)
    textrcnn.load_state_dict(torch.load('./pretrain_models/textrcnn_params_multi_label_80.pth', map_location='cpu'))

    token = Tokenizer(sentence)  # 分词
    indices = data_set.text.vocab.lookup_indices(token)  # str -> index
    for i in range(5):  # padding, 默认5个0
        indices.append(0)
    x = torch.Tensor(indices).to(torch.int64)  # !!!转换成int64类型
    x = x.unsqueeze(0).to(DEVICE)  # (len) -> (1, len)增加一个batch维度

    output_cnn = textcnn(x)  # 预测
    output_cnn[output_cnn >= 0.5] = 1
    output_cnn[output_cnn < 0.5] = 0

    output_rcnn = textrcnn(x)
    output_rcnn[output_rcnn >= 0.5] = 1
    output_rcnn[output_rcnn < 0.5] = 0

    max_value_index = torch.max(output_cnn + output_rcnn, 1)[1]  # 获取output中每一行最大值的下标
    output = output_cnn + output_rcnn  # 两个模型预测的结果取并集
    output[output >= 1] = 1
    output[output < 1] = 0
    for i in range(output.shape[0]):
        output[i][max_value_index[i]] = 1  # 防止全0的情况
    result_list = output.int().cpu().detach().numpy().tolist()[0]
    label_list = ['空间', '动力', '操控', '能耗', '舒适性', '外观', '内饰', '性价比', '配置', '续航', '安全性',
                  '环保', '质量与可靠性', '充电', '服务', '品牌', '智能驾驶', '其它']
    labels_predicted = []
    for index, value in enumerate(result_list):
        if value == 1:
            labels_predicted.append(label_list[index])
    print(labels_predicted)
    return ' '.join(labels_predicted)


if __name__ == '__main__':
    Classify('空间很大，比较舒服。')
