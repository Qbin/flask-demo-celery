#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/8 8:43 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : test_model.py
import os

from mongoengine import Document, StringField, ListField

from common.time_cost import calculate_runtime


class AnalyzedData(Document):
    data_id = StringField(required=True)
    analyzed_data = ListField(required=True)
    meta = {'collection': 'analyzed_data'}

    @classmethod
    def switch_collection(cls, collection_name, keep_created=True):
        cls._meta['collection'] = collection_name

    @classmethod
    def batch_insert(cls, documents, filed_name='analyzed_data'):
        cls.switch_collection(filed_name)
        cls.objects.insert(documents)

    @classmethod
    def batch_upsert(cls, indexes, texts, filed_name='analyzed_data'):
        cls.switch_collection(filed_name)
        for data_id, text in zip(indexes, texts):
            cls.objects(data_id=str(data_id)).update_one(set__analyzed_data=text, upsert=True)
        print('Upserted:', cls.objects.count())

    # @classmethod
    # def batch_insert(cls, documents):
    #     bulk_operations = []
    #     for document in documents:
    #         data_id = document['data_id']
    #         operation = UpdateOne({'data_id': data_id}, {'$set': document}, upsert=True)
    #         bulk_operations.append(operation)
    #
    #     cls.objects.insert(bulk_operations)

    @classmethod
    @calculate_runtime
    def batch_find_by_ids(cls, id_list, filed_name='analyzed_data'):
        cls.switch_collection(filed_name)
        if os.getenv("DB_MODE") == "mongo":
            # 使用聚合操作提取数据
            pipeline = [
                {
                    '$match': {
                        'data_id': {'$in': id_list}
                    }
                },
                {
                    '$project': {
                        'analyzed_data': 1
                    }
                }
            ]
            return cls.objects.aggregate(*pipeline)

        if id_list:
            return cls.objects(data_id__in=id_list)
            # return cls.objects(data_id__in=id_list).only("analyzed_data")
        else:
            return cls.objects()

    @classmethod
    @calculate_runtime
    def get_exist_data_id(cls, id_list, filed_name='analyzed_data'):
        cls.switch_collection(filed_name)

        pipeline = [
            {
                '$match': {
                    'data_id': {'$in': id_list}
                }
            },
            {
                '$project': {
                    'data_id': 1
                }
            }
        ]
        return cls.objects.aggregate(*pipeline)


# 示例用法
if __name__ == "__main__":
    # 假设有一批要插入的文档
    documents = [
        AnalyzedData(data_id='1', analyzed_data=['分词结果1']),
        AnalyzedData(data_id='2', analyzed_data=['分词结果2']),
        AnalyzedData(data_id='3', analyzed_data=['分词结果3'])
    ]

    # 批量插入文档
    AnalyzedData.batch_insert(documents)

    # 假设有一批要查询的id列表
    id_list = ['1', '3']

    # 根据id列表批量查找文档
    results = AnalyzedData.batch_find_by_ids(id_list)
    for result in results:
        print(result.data_id, result.analyzed_data)
