# -*- coding:utf-8 -*-

"""
Create on 2020/9/4 2:22 下午
@Author: dfsj
@Description: 
"""
import re
import pickle
from algorithms.config import *

stop_words = set([item.strip() for item in open(PATH_STOPWORDS, 'r').readlines()])


def clear_character(sentence):
    """ 只保留汉字、字母、数字 """
    pattern = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    line = re.sub(pattern, '', sentence.lower())
    new_sentence = ''.join(line.split())  # 去除空白
    return new_sentence


def drop_stopwords(line):
    """ 去停用词 """
    line_clean = []
    for word in line:
        if word in stop_words:
            continue
        line_clean.append(word)
    return line_clean


def get_jieba_keywords():
    # todo 从数据库获取、从文件获取、从接口获取
    return ["chatgpt", "stablediffusion"]


def save_texts(texts, field_name):
    # 序列化并保存到文件
    with open('{}_list.pkl'.format(field_name), 'wb') as f:
        pickle.dump(texts, f)


def load_texts(field_name):
    with open('{}_list.pkl'.format(field_name), 'rb') as f:
        loaded_list = pickle.load(f)

    return loaded_list
