# -*- coding:utf-8 -*-

"""
Create on 2020/9/4 11:40 上午
@Author: dfsj
@Description:     加载数据集及数据处理
"""
import gensim
import pandas as pd
import jieba
import numpy as np
from config import *
from common.time_cost import calculate_runtime
from util import clear_character, drop_stopwords

from multiprocessing import Pool


# 定义并行分词函数
def parallel_cut(sentence):
    return jieba.lcut(sentence)


class Data:
    """
    默认加载的是新浪新闻数据集，对数据集进行数据清洗、分词、gram 化处理等， 以满足 LDA 算法的输入需求，数据处理示例如下：
    原始数据：
        马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有摆脱雨水的困扰。
        7月31日下午6点，国奥队的日常训练再度受到大雨的干扰，无奈之下队员们只慢跑了25分钟就草草收场。31日上午10点，
        国奥队在奥体中心外场训练的时候，天就是阴沉沉的，气象预报显示当天下午沈阳就有大雨，但幸好队伍上午的训练并没
        有受到任何干扰。下午6点，当球队抵达训练场时，大雨已经下了几个小时，而且丝毫没有停下来的意思。抱着试一试的态度，
        球队开始了当天下午的例行训练，25分钟过去了，天气没有任何转好的迹象，为了保护球员们，国奥队决定中止当天的训练，
        全队立即返回酒店。在雨中训练对足球队来说并不是什么稀罕事，但在奥运会即将开始之前，全队变得“娇贵”了。在沈阳最
        后一周的训练，国奥队首先要保证现有的球员不再出现意外的伤病情况以免影响正式比赛，因此这一阶段控制训练受伤、
        控制感冒等疾病的出现被队伍放在了相当重要的位置。而抵达沈阳之后，中后卫冯萧霆就一直没有训练，冯萧霆是7月27日在
        长春患上了感冒，因此也没有参加29日跟塞尔维亚的热身赛。队伍介绍说，冯萧霆并没有出现发烧症状，但为了安全起见，
        这两天还是让他静养休息，等感冒彻底好了之后再恢复训练。由于有了冯萧霆这个例子，因此国奥队对雨中训练就显得特别谨慎，
        主要是担心球员们受凉而引发感冒，造成非战斗减员。而女足队员马晓旭在热身赛中受伤导致无缘奥运的前科，也让在沈阳的
        国奥队现在格外警惕，“训练中不断嘱咐队员们要注意动作，我们可不能再出这样的事情了。”一位工作人员表示。从长春到沈阳，
        雨水一路伴随着国奥队，“也邪了，我们走到哪儿雨就下到哪儿，在长春几次训练都被大雨给搅和了，没想到来沈阳又碰到这种
        事情。”一位国奥球员也对雨水的“青睐”有些不解。

    第一步：数据清洗，只保留中文、英文、数字
        马晓旭意外受伤让国奥警惕无奈大雨格外青睐殷家军记者傅亚雨沈阳报道来到沈阳国奥队依然没有摆脱雨水的困扰7月31日下午
        6点国奥队的日常训练再度受到大雨的干扰无奈之下队员们只慢跑了25分钟就草草收场31日上午10 。。。。。。。
        有些不解

    第二步：分词
        '马晓旭', '意外', '受伤', '让', '国奥', '警惕', '无奈', '大雨', '格外', '青睐', '殷家', '军', '记者',
        '傅亚雨', '沈阳', '报道', '来到', '沈阳', '国奥队', '依然', '没有', '摆脱', '雨水', '的', '困扰', '7',
        '月', '31', '日', '下午', '6', '点', '国奥队', '的', '日常', '训练', '再度', '受到', '大雨', '的',
        '干扰', '无奈', '之下', '队员', '们', '只', '慢跑', '了', '25', '分钟', '就', '草草收场', '31', '日',
        '上午', '10', '点', '国奥队', '在', '奥体中心', '外场', '训练', '的', '时候', '天', '就是', '阴沉沉', '的',
        '气象预报', '显示', '当天', '下午', '沈阳', '就', '有', '大雨', '但', '幸好', 。。。。。。。。。。。

    第三步：去停用词，在这里一开始只是用基本开源停用词表，但后续的 LDA 效果不理想，所以把数字和单个字全部加入到停用词表中
        '马晓旭', '意外', '受伤', '国奥', '警惕', '无奈', '大雨', '格外', '青睐', '殷家', '傅亚雨', '沈阳', '报道',
        '来到', '沈阳', '国奥队', '依然', '摆脱', '雨水', '困扰', '下午', '国奥队', '日常', '训练', '再度', '大雨',
        '干扰', '无奈', '之下', '队员', '慢跑', '分钟', '草草收场', '上午', '国奥队', '奥体中心', '外场', '训练',
        '阴沉沉', '气象预报', '显示', '当天', '下午', '沈阳', '大雨', '幸好', '队伍', '上午', '训练', '干扰', '下午',
        '点当', '球队', '抵达', '训练场', '大雨', '几个', '小时', '丝毫', '停下来', '试一试', '态度', '球队', '当天',
        '下午', '例行', '训练', '分钟', '天气', '转好', '迹象', '保护', '球员', '国奥队', '中止', '当天', '训练',
        '全队', '返回', '酒店', '训练', '足球队', '来说', '稀罕', '奥运会', '即将', '全队', '变得', '娇贵', '沈阳',
        '一周', '训练', '国奥队', '保证', '现有', '球员', '不再', '出现意外', '伤病', '情况', '影响', '正式', '比赛',
        '这一', '阶段', '控制', '训练', '受伤', '控制', '感冒', '疾病', '队伍', '放在', '位置', '抵达', '沈阳', '后卫',
        '冯萧霆', '训练', '冯萧霆', '长春', '患上', '感冒', '参加', '塞尔维亚', '热身赛', '队伍', '介绍', '冯萧霆',
        '发烧', '症状', '两天', '静养', '休息', '感冒', '恢复', '训练', '冯萧霆', '例子', '国奥队', '对雨中', '训练',
        '显得', '特别', '谨慎', '担心', '球员', '受凉', '引发', '感冒', '非战斗', '减员', '女足', '队员', '马晓旭',
        '热身赛', '受伤', '导致', '无缘', '奥运', '前科', '沈阳', '国奥队', '格外', '警惕', '训练', '嘱咐', '队员',
        '动作', '再出', '事情', '工作人员', '长春', '沈阳', '雨水', '一路', '伴随', '国奥队', '长春', '几次', '训练',
         '大雨', '搅和', '没想到', '沈阳', '碰到', '事情', '国奥', '球员', '雨水', '青睐', '不解'

    第四步：bigram 或 trigram 处理，例如下文的 "患上_感冒", "无奈_之下" 等
        '马晓旭', '意外', '受伤', '国奥', '警惕', '无奈', '大雨', '格外', '青睐', '殷家', '傅亚雨', '沈阳', '报道',
        '来到', '沈阳', '国奥队', '依然', '摆脱', '雨水', '困扰', '下午', '国奥队', '日常', '训练', '再度', '大雨',
        '干扰', '无奈_之下', '队员', '慢跑', '分钟', '草草收场', '上午', '国奥队', '奥体中心', '外场', '训练',
        '阴沉沉', '气象预报', '显示', '当天', '下午', '沈阳', '大雨', '幸好', '队伍', '上午', '训练', '干扰', '下午',
        '点当', '球队', '抵达', '训练场', '大雨', '几个', '小时', '丝毫', '停下来', '试一试', '态度', '球队',
        '当天', '下午', '例行_训练', '分钟', '天气', '转好', '迹象', '保护', '球员', '国奥队', '中止', '当天',
        '训练', '全队', '返回', '酒店', '训练', '足球队', '来说', '稀罕', '奥运会', '即将', '全队', '变得', '娇贵',
        '沈阳', '一周', '训练', '国奥队', '保证', '现有', '球员', '不再', '出现意外', '伤病', '情况', '影响', '正式',
        '比赛', '这一', '阶段', '控制', '训练', '受伤', '控制', '感冒', '疾病', '队伍', '放在', '位置', '抵达',
        '沈阳', '后卫', '冯萧霆', '训练', '冯萧霆', '长春', '患上_感冒', '参加', '塞尔维亚', '热身赛', '队伍',
        '介绍', '冯萧霆', '发烧_症状', '两天', '静养', '休息', '感冒', '恢复', '训练', '冯萧霆', '例子', '国奥队',
        '对雨中', '训练', '显得', '特别', '谨慎', '担心', '球员', '受凉', '引发', '感冒', '非战斗', '减员', '女足',
        '队员', '马晓旭', '热身赛', '受伤', '导致', '无缘', '奥运', '前科', '沈阳', '国奥队', '格外', '警惕',
        '训练', '嘱咐', '队员', '动作', '再出', '事情', '工作人员', '长春_沈阳', '雨水', '一路', '伴随', '国奥队',
        '长春', '几次', '训练', '大雨', '搅和', '没想到', '沈阳', '碰到', '事情', '国奥', '球员', '雨水', '青睐', '不解'

    第五步：创建词典 id2word，例如 id2word[100] --->  "草草收场"

    第六步：对文本进行词频统计，即文本向量化表示，下文表示二元组的含义是 第几个词在该条新闻中出现的次数
        [(0, 1), (1, 1), (2, 2), (3, 4), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 2), (10, 1), (11, 1),
        (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 2),
        (23, 1), (24, 1), (25, 4), (26, 1), (27, 1), (28, 1), (29, 1), (30, 2), (31, 1), (32, 1), (33, 1),
        (34, 1), (35, 1), (36, 3), (37, 1), (38, 1), (39, 1), (40, 1), 。。。。。。。。。]

    第六步可视化：将上文翻译为可读模式（仅仅是为了方便人读）
        [('一周', 1), ('一路', 1), ('上午', 2), ('下午', 4), ('不再', 1), ('不解', 1), ('丝毫', 1), ('两天', 1),
        ('中止', 1), ('事情', 2), ('介绍', 1), ('休息', 1), ('伤病', 1), ('伴随', 1), ('位置', 1), ('例子', 1),
        ('例行_训练', 1), ('依然', 1), ('保护', 1), ('保证', 1), ('停下来', 1), ('傅亚雨', 1), ('全队', 2),
        ('再出', 1), ('再度', 1), ('冯萧霆', 4), ('减员', 1), ('几个', 1), ('几次', 1), ('出现意外', 1), ('分钟', 2),
        ('前科', 1), ('动作', 1), ('即将', 1), ('参加', 1), ('发烧_症状', 1), 。。。。。。。。。。。。]

    """

    def __init__(self, data_path=None, bigram=True, trigram=False):
        """
        加载处理数据集
        :param data_path:     数据集位置
        :param bigram:        是否对分词结果进行 bigram 处理
        :param trigram:       是否对分词结果进行 trigram 处理
        """
        self.data_path = data_path
        self.bigram = bigram
        self.trigram = trigram
        self.df = None
        self.train_st_text = None
        self.bigram_mod = None
        self.trigram_mod = None

    @calculate_runtime
    def load_data(self):
        """ 加载数据集，假设数据集有两列，分别是人工标注的标签列和数据列 """
        src_df = pd.read_excel(self.data_path, engine='openpyxl', )

        self.df = src_df[[u"正文"]].rename(columns={"正文": "data"})
        newdf = pd.DataFrame(np.repeat(self.df.values, 1, axis=0))
        newdf.columns = self.df.columns
        self.df = newdf
        # self.df = pd.read_csv(self.data_path, delimiter="\t", header=None, names=["label", "data"])
        # logger.info("已加载数据集，原标注数据集具有以下 label：{}".format(",".join(self.df.label.unique())))

    @calculate_runtime
    def data_cut(self, datas=None):
        if not datas:
            datas = self.df["data"]
        logger.info("对数据集进行清洗并分词，请稍后 ...")
        train_clean = [clear_character(data) for data in datas]
        logger.info("数据清洗完，示例为：{}".format(train_clean[0]))
        train_seg_text = [jieba.lcut(s) for s in train_clean]
        # pool = Pool()
        # train_seg_text = pool.imap(parallel_cut, train_clean)
        # pool.close()
        # pool.join()
        # logger.info("数据分词完成，示例为：{}".format(train_seg_text[0]))
        self.train_st_text = [drop_stopwords(s) for s in train_seg_text]
        logger.info("去停用词完成，示例为：{}".format(self.train_st_text[0]))

    @calculate_runtime
    def data_gram(self):
        if self.bigram and self.trigram:
            logger.warning("bigram 与 trigram 同时为 True，按 trigram 处理")
            self.bigram = False

        if self.bigram:
            bigram = gensim.models.Phrases(self.train_st_text, min_count=5, threshold=100)
            self.bigram_mod = gensim.models.phrases.Phraser(bigram)
            data_words_bigrams = [self.bigram_mod[doc] for doc in self.train_st_text]
            logger.info("bigram 处理完成，示例为：{}".format(data_words_bigrams[0]))
            return data_words_bigrams

        if self.trigram:
            bigram = gensim.models.Phrases(self.train_st_text, min_count=5, threshold=100)
            trigram = gensim.models.Phrases(bigram[self.train_st_text], threshold=100)
            self.bigram_mod = gensim.models.phrases.Phraser(bigram)
            self.trigram_mod = gensim.models.phrases.Phraser(trigram)
            data_words_trigrams = [self.trigram_mod[self.bigram_mod[doc]] for doc in self.train_st_text]
            logger.info("trigram 处理完成，示例为：{}".format(data_words_trigrams[0]))
            return data_words_trigrams

    @calculate_runtime
    def get_seg_corpus(self, datas=None):
        """ 分词或者进行 gram 处理后的语料集 """
        if self.data_path:  # 指定了文件位置，从文件中读取数据
            self.load_data()
            self.data_cut()
        else:
            assert datas is not None, "params `datas` cann't be None"
            self.data_cut(datas)

        if not self.bigram and not self.trigram:
            return self.train_st_text
        else:
            return self.data_gram()

    def get_labels(self):
        """ 用于获取数据标签 """
        if self.df is not None:
            return self.df.label
        else:
            logger.warning("未发现数据标签信息")
