import jieba.analyse as analyse


def ExtractKeyword(sentence):
    length = len(sentence)
    k = 0  # 关键词个数
    keywords = []
    analyse.set_stop_words('./data/stopwords.txt')
    if length <= 10:  # 短文档用textrank,长文档用tfidf
        k = 1
        keywords.extend(analyse.textrank(sentence, topK=k))  # TextRank
    elif length <= 20:
        k = 2
        keywords.extend(analyse.textrank(sentence, topK=k))  # TextRank
    else:
        k = 3
        keywords.extend(analyse.extract_tags(sentence, topK=k))  # tf-idf
    print(keywords)
    return ' '.join(keywords)


if __name__ == '__main__':
    sentence = '后备箱的空间也要赞，几乎是同级别中变态的大啊，过年过节拉点东西，不用担心空间不够啊有没有，说到空间，储物空间也要说一下，车门下放的开关式储物格真心实用还不影响美观'
    ExtractKeyword(sentence)
