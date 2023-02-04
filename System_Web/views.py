from os import environ
from django.shortcuts import render
from django.http import JsonResponse
from System_Web.models import Bmw
from django.core.paginator import Paginator
from funtions.extract_keyword import ExtractKeyword
from funtions.sentiment import PredictSentiment
from funtions.text_classification_multi import Classify


# 跳转到主页
def Home(request):
    return render(request, 'home.html')


# 跳转到文本分析页面
def Text(request):
    page = request.GET.get('page')  # 获取url中page参数的值
    if page:
        page = int(page)
    else:
        page = 1
    print('PAGE 参数为：', page)
    data_list = Bmw.objects.all()  # 获取Bmw数据库中所有数据
    paginator = Paginator(data_list, 250)  # 实例化分页组件,每页250条数据
    page_data_list = paginator.page(page)  # 获得指定page的列表
    page_num = paginator.num_pages  # 获得列表被分页处理后，总共被分为多少页
    print('page_num:', page_num)

    if page_data_list.has_next():  # 判断是否存在下一页
        next_page = page + 1
    else:
        next_page = page

    if page_data_list.has_previous():   # 是否存在上一页
        previous_page = page - 1
    else:
        previous_page = page

    return render(request, 'text.html', {
        'data_list': page_data_list,  # 当前page的数据
        'page_num': page_num,  # 被分了几页，这个返回一个数组，前端直接for循环渲染
        'cur_page': page,
        'next_page': next_page,
        'previous_page': previous_page
    })


# 跳转到统计分析页面
def Statistic(request):

    models = set(Bmw.objects.values_list('car_series', flat=True))  # 车型

    data = Bmw.objects.all()  # 完整表单
    model = request.GET.get('model', '全部')
    if model != '全部':
        data = data.filter(car_series=model)

    positive = data.filter(sentiment=1).count()  # 正面评价的数据个数
    negative = data.filter(sentiment=-1).count()  # 正面评价的数据个数
    neutral = data.filter(sentiment=0).count()  # 中性评价的数据个数

    class_temp = list(data.values_list('car_class', flat=True))
    car_class = []
    for item in class_temp:
        car_class.extend(str(item).split())

    space = car_class.count('空间')  # 空间
    power = car_class.count('动力')  # 动力
    control = car_class.count('操控')  # 操控
    consumption = car_class.count('能耗')  # 能耗
    comfort = car_class.count('舒适性')  # 舒适性
    appearance = car_class.count('外观')  # 外观
    interior = car_class.count('内饰')  # 内饰
    cost_performance = car_class.count('性价比')  # 性价比
    equipment = car_class.count('配置')  # 配置
    duration = car_class.count('续航')  # 续航
    safety = car_class.count('安全性')  # 安全性
    environmental = car_class.count('环保')  # 环保
    quality = car_class.count('质量与可靠性')  # 质量与可靠性
    charge = car_class.count('充电')  # 充电
    service = car_class.count('服务')  # 服务
    car_brand = car_class.count('品牌')  # 品牌
    intelligent = car_class.count('智能驾驶')  # 智能驾驶
    others = car_class.count('其它')  # 其它

    others += duration + charge + service + intelligent



    return render(request, 'statistic.html', {'positive': positive, 'negative': negative, 'space': space,
                                              'power': power, 'control': control, 'consumption': consumption,
                                              'comfort': comfort, 'appearance': appearance, 'interior': interior,
                                              'cost_performance': cost_performance,'equipment':equipment,
                                              'duration':duration,'safety':safety,'environmental':environmental,
                                              'quality':quality,'charge':charge,'service':service,
                                              'car_brand':car_brand,'intelligent':intelligent,'neutral':neutral,
                                              'models':models,'others':others,'model':model})


# 跳转到测试页面
def Test(request):
    return render(request, 'test.html')


# 测试页面提交数据
def PostText(request):
    text = request.GET.get('text')
    print(text)
    keywords = ExtractKeyword(text)
    sentiment = PredictSentiment(text)
    car_class = Classify(text)
    print(keywords)
    print(sentiment)
    print(car_class)
    return JsonResponse({'keywords': keywords, 'car_class': car_class, 'sentiment': sentiment})


# 跳转到关于页面
def Info(request):
    return render(request, 'info.html')


# 跳转到模板页面
def Temp(request):
    return render(request, 'template_page.html')
