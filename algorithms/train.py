# -*- coding:utf-8 -*-

"""
Create on 2020/9/27 9:15 上午
@Author: dfsj
@Description:   LDA 模型训练
"""
import os.path

from common.time_cost import calculate_runtime
from load_datas import Data
# from algorithms.lda import LDA
from kmeans import KMEANS
import joblib
from algorithms.config import *
from algorithms.util import load_texts, save_texts

data = Data(TRAIN_SETS)
field_name = "my"
if os.path.exists('{}_list.pkl'.format(field_name)):
    base_texts = load_texts(field_name)
    base_texts.extend(data.get_seg_corpus())
    texts = base_texts
else:
    texts = data.get_seg_corpus()
save_texts(texts, field_name)


# labels = data.get_labels()
# with open("texts", "w") as f:
#     f.write("\n".join([" ".join(items) for items in texts]))
# texts = [item.strip().split() for item in open("data/texts", "r").readlines()]


def train_lda():
    lda = LDA(texts=texts, num_topics=10)
    model = lda.train()
    model.save(MODEL_LDA)


@calculate_runtime
def train_kmeans():
    corpus = [' '.join(line) for line in texts]
    kmeans = KMEANS(corpus, num_clusters=2)
    model = kmeans.train()
    # joblib.dump(model, MODEL_KMEANS)

    kmeans.print_top_terms()
    kmeans.draw()
    # kmeans.find_optimal_clusters(20)

    # kmeans.print_summary(labels=labels)


if __name__ == "__main__":
    # train_lda()
    train_kmeans()
    # import psutil
    #
    # # 获取当前进程的内存占用情况
    # process = psutil.Process()
    # memory_info = process.memory_info()

    # 打印内存占用信息
    # print(f"当前进程的内存占用：{memory_info.rss} 字节")
