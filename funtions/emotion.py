from string import punctuation
import torch
from torch import nn
import dill
import re
import os
import jieba


class lsRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(lsRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)  # 映射长度
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)
        self.decoder1 = nn.Linear(4 * num_hiddens, 100)
        self.decoder2 = nn.Linear(100, 3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        # 将inputs转置，seq_len作为第一维
        embeddings = self.embedding(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs1 = self.decoder1(encoding)
        outs2 = self.decoder2(outs1)
        outs = self.softmax(outs2)
        return outs


def get_tokenized(data):
    def tokenized(text):
        punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\\s:：'
        text = str(text)
        text = re.sub(r"[{}]+".format(punc), " ", text)
        word_list = list(jieba.cut(text))
        return list(filter(lambda x: x != ' ', word_list))
    return [tokenized(review) for review, _ in data]


def preprocess(data, vocab):
    max_len = 200

    def pad(x):
        return x[:max_len] if len(x) > max_len else x + [0] * (max_len - len(x))

    def to_label(score):
        # return [1, 0, 0] if score==-1 else [0, 1, 0] if score==0 else [0, 0, 1]
        return 0 if score == -1 else 1 if score == 0 else 2

    tokenized_data = get_tokenized(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in view]) for view in tokenized_data])

    labels = torch.tensor([0])
    return features, labels


def predict_zfz(text):
    file_os = os.path.dirname(__file__)
    with open('./pretrain_models/zfz_vocab_new', 'rb')as f:
        vocab = dill.load(f)
        RNNnet = lsRNN(vocab, 50, 100, 2)
        RNNnet.load_state_dict(torch.load('./pretrain_models/Emotion_classification_zfz_new.pth', map_location='cpu'))
        RNNnet.eval()

        data = [[text, 1]]
        x, _ = preprocess(data, vocab)

        with torch.no_grad():
            label = torch.argmax(RNNnet(x), dim=1)

        if label.item() == 0:
            return '负面'
        if label.item() == 1:
            return '中性'
        if label.item() == 2:
            return '正面'
        print(label.item)


if __name__ == '__main__':
    print(predict_zfz('经济模式下开没有想象中费油，大概9～10个左右城市路段表现：走烂路比较颠，过减速带难受，市区车多的时候开经济模式很舒服高速路段表现：重心低，四驱，跑高速比较稳，100公里以上动力也还算充沛'))
