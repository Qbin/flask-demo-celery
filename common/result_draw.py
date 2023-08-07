import matplotlib.pyplot as plt
import numpy as np

import json


def get_result_params_by_file(file_name: str):
    # 打开文本文件
    with open(file_name, 'r') as file:
        # 读取文件内容
        content = file.read()
        # 将JSON字符串转换为字典
        data = json.loads(content)

    # 打印字典cmeans.py
    # print(data)
    return data["data"]


def draw_kmeans():
    data = get_result_params_by_file("kmeans.result_newtxt.txt")
    # data = {
    #     "X": [],
    #     "centroids": [],
    #     "labels": []
    # }

    X = np.array(data["X"])
    labels = np.array(data["labels"])
    # centroids = np.array(data["centroids"])

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='r')
    plt.show()


def draw_dbscan():
    data = get_result_params_by_file("dbscan.result_new.txt")
    # data_scan = {
    #     "X": [],
    #     "labels": [],
    #     "model_params": {
    #         "eps": 0.03,
    #         "min_samples": 2
    #     }
    # }
    labels = np.array(data["labels"])
    points = np.array(data["X"])

    # 提取聚类结果的唯一标签
    unique_labels = set(labels)

    # 创建一个颜色列表，用于给不同的聚类结果分配不同的颜色
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # 画出聚类结果
    for label in unique_labels:
        if label == -1:
            # 如果标签为-1，则表示为噪声点，用黑色表示
            color = 'k'
        else:
            color = colors[label % len(colors)]

        # 提取属于当前聚类结果的点
        cluster_points = [points[i] for i in range(len(points)) if labels[i] == label]

        # 提取属于当前聚类结果的 x 和 y 值
        x = [point[0] for point in cluster_points]
        y = [point[1] for point in cluster_points]

        # 画出当前聚类结果的散点图
        plt.scatter(x, y, color=color)

    # 显示图形
    plt.show()


draw_dbscan()
# draw_kmeans()
