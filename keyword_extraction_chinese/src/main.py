# -*- coding: utf-8 -*-
import jieba.analyse
import jieba

class Extractor():
    def __init__(self):
        self.topK = 10

    def run(self,document):
        tags = jieba.analyse.extract_tags(document, topK = self.topK)
        return tags

if __name__=='__main__':
    extractor = Extractor()
    text = "全国政协委员郁瑞芬在本次两会中主要关注食品领域高质量发展及城市供应链建设方面的话题。她认为，要实现食品领域高质量发展，需要行业各方进一步理解消费者对健康优质产品的诉求。谈到如何完善城市供应链建设话题，郁瑞芬建议将全国不同商贸业网点进行统筹，形成和提升城市供应链合力。   【责任编辑:高瑞东】"
    keywords = extractor.run(text)
    seg_list = jieba.cut(text, cut_all=False)
    lst = list(list(set(seg_list)))
    True_or_False = []
    for i in keywords:
        if i in text:
            True_or_False.append(True)
        else:
            True_or_False.append(False)
    res = all(True_or_False)
    print(res)

    
    