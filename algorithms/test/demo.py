#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/20 17:24

import os
import random
import pandas as pd

from algorithms.kmeans import KMEANS
from algorithms.load_datas import Data


def get_test_data():
    # 定义文件夹路径
    folder_path = "/Users/qinbinbin/Downloads/社交媒体数据"

    # 获取文件夹下所有子目录
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # 创建一个空的DataFrame
    df = pd.DataFrame()

    # 遍历每个子目录
    for subfolder in subfolders:
        # 拼接文件路径
        file_path = os.path.join(subfolder, "社媒.xlsx")

        # 读取Excel文件为DataFrame
        subfolder_df = pd.read_excel(file_path, engine='openpyxl', )

        # 在DataFrame中增加属性列，标记数据来自哪个子目录
        subfolder_df["子目录"] = os.path.basename(subfolder)

        # 将子目录的DataFrame添加到总的DataFrame中
        df = df.append(subfolder_df)
        df["data"] = df["正文"]

    return df


def load_files_from_subdirectories(directory, num_files):
    file_contents = []
    labels = []

    # 获取主目录下的所有子目录
    subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]

    for subdir in subdirectories:
        subdir_path = os.path.join(directory, subdir)

        # 获取子目录下所有txt文件
        txt_files = [file for file in os.listdir(subdir_path) if file.endswith('.txt')]

        # 随机选择n个txt文件
        selected_files = random.sample(txt_files, num_files)

        for file in selected_files:
            file_path = os.path.join(subdir_path, file)

            # 读取txt文件内容
            with open(file_path, 'r') as f:
                file_content = f.read()

            # 保存文件内容和子目录名作为标签
            file_contents.append(file_content)
            labels.append(subdir)

    # 创建DataFrame
    df = pd.DataFrame({'content': file_contents, 'label': labels})

    return df


def main2():
    # 示例用法
    directory = "/Users/qinbinbin/Downloads/THUCNews"  # 替换为实际的文件夹路径
    num_files = 500  # 随机选择的文件数量

    df = load_files_from_subdirectories(directory, num_files)
    print(df)


def main(file_name, field_name, num_clusters=3):
    # df = get_test_data()
    # df.to_excel("output.xlsx")

    data = Data(file_name, field_name=field_name)
    texts = data.get_seg_corpus()
    corpus = [' '.join(line) for line in texts]
    kmeans = KMEANS(corpus, num_clusters=num_clusters)
    model = kmeans.train()
    # joblib.dump(model, MODEL_KMEANS)

    # kmeans.print_top_terms()
    kmeans.draw_new("pca")
    result = list(model.labels_)
    print(dict([(i, result.count(i)) for i in result]))


if __name__ == '__main__':
    main(file_name="output.xlsx", field_name="正文", num_clusters=8)
    # main(file_name="thucnews.xlsx", field_name="content", num_clusters=14)
