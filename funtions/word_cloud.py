from matplotlib import pyplot as plt  # 绘图，数据可视化
from wordcloud import WordCloud  # 词云
from PIL import Image  # 图片处理
import numpy as np  # 矩阵运算
import pymysql  # 数据库


def GetWordCloud(_class, value):
    # 读取关键词
    keywords = []
    db = pymysql.connect(host='localhost', user='root', password='123456', database='nlp')
    cursor = db.cursor()
    if _class == 'sentiment':
        sql = 'select keyword1, keyword2, keyword3 from bmw where {}={}'.format(_class, value)
    else:
        sql = 'select keyword1, keyword2, keyword3 from bmw where {}="{}"'.format(_class, value)
    print(sql)
    cursor.execute(sql)
    for item in cursor.fetchall():
        keywords.extend(list(item))
    cursor.close()
    db.close()

    # 选出词频最高的前20个
    while None in keywords:
        keywords.remove(None)
    keywords_set = set(keywords)
    count = {}
    for keyword in keywords_set:
        count.update({keyword: keywords.count(keyword)})
    count_sort = sorted(count.items(), key=lambda item: item[1], reverse=True)
    keywords = [count_sort[i][0] for i in range(20)]
    string = ' '.join(keywords)

    name_dict = {'空间': 'space', '动力': 'power', '操控': 'control', '能耗': 'consumption',
                 '舒适性': 'comfort', '外观': 'appearance', '内饰': 'interior', '性价比': 'cost_performance'}
    if _class == 'sentiment' and value == 1:
        pic_name = 'positive'
    elif _class == 'sentiment' and value == -1:
        pic_name = 'negative'
    elif _class == 'sentiment' and value == 0:
        pic_name = 'neutral'
    else:
        pic_name = name_dict[value]
    img = Image.open('./System_Web/static/assets/img/{}.jpg'.format(pic_name))  # 打开遮罩图片
    img_array = np.array(img)  # 将图片转换为数组
    wc = WordCloud(
        background_color='white',
        mask=img_array,
        font_path='msyh.ttc'  # 字体所在位置：C:\Windows\Fonts
    )
    wc.generate_from_text(string)

    # 绘制图片
    plt.figure(1)
    plt.imshow(wc)
    plt.axis('off')  # 是否显示坐标轴

    # 输出词云图片到文件
    plt.savefig('./System_Web/static/assets/img/wc_{}.png'.format(pic_name), dpi=500)

    # plt.show()  # 显示生成的词云图片


if __name__ == '__main__':
    _class = 'sentiment'  # sentiment / car_class
    for value in [1,-1]:
        GetWordCloud(_class, value)
    # _class = 'car_class'
    # for value in ['空间', '动力', '操控', '能耗', '舒适性', '外观', '内饰', '性价比']:
    #     GetWordCloud(_class, value)
