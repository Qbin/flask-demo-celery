from gensim.models import Word2Vec
from sklearn.cluster import KMeans


class TextClustering:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None

    def train_model(self, sentences):
        self.model = Word2Vec(sentences, min_count=1)
        if self.model_path:
            self.model.save(self.model_path)

    def load_model(self, model_path):
        self.model = Word2Vec.load(model_path)

    def get_sentence_vectors(self, sentences):
        sentence_vectors = []
        for sentence in sentences:
            vector = sum([self.model.wv[word] for word in sentence]) / len(sentence)
            sentence_vectors.append(vector)
        return sentence_vectors

    def cluster_sentences(self, sentences, num_clusters):
        sentence_vectors = self.get_sentence_vectors(sentences)
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(sentence_vectors)
        return kmeans.labels_


if __name__ == '__main__':
    # 准备文本数据
    sentences = [
        ['this', 'is', 'the', 'first', 'sentence'],
        ['this', 'is', 'the', 'second', 'sentence'],
        ['this', 'is', 'the', 'third', 'sentence'],
        ['this', 'is', 'the', 'fourth', 'sentence']
    ]

    # 创建TextClustering对象
    clustering = TextClustering()

    # 训练模型
    clustering.train_model(sentences)

    # 获取句子向量
    sentence_vectors = clustering.get_sentence_vectors(sentences)

    # 聚类句子
    num_clusters = 2
    labels = clustering.cluster_sentences(sentences, num_clusters)
    print(labels)
