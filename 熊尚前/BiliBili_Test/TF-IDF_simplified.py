import math
from collections import Counter

# 准备语料库
corpus = [
    '这 是 第一个 文档',
    '这是 第二个 文档',
    '这是 最后 一个 文档',
    '没有 文档 了 文档'
]

# 分词
tokenized_corpus = [doc.split() for doc in corpus]

# 计算词频
word_counts = [Counter(doc) for doc in tokenized_corpus]

# 计算 TF
def compute_tf(word_count):
    total_words = sum(word_count.values())
    tf = {word: freq / total_words for word, freq in word_count.items()}
    return tf

# 计算 IDF
def compute_idf(tokenized_corpus):
    num_docs = len(tokenized_corpus)
    idf = {}
    all_words = set([word for doc in tokenized_corpus for word in doc])
    for word in all_words:
        doc_count = sum([1 for doc in tokenized_corpus if word in doc])
        idf[word] = math.log(num_docs / (1 + doc_count))
    return idf

# 计算 TF-IDF
def compute_tf_idf(tokenized_corpus):
    word_counts = [Counter(doc) for doc in tokenized_corpus]
    tf_list = [compute_tf(count) for count in word_counts]
    idf = compute_idf(tokenized_corpus)
    tf_idf_list = []
    for tf in tf_list:
        tf_idf = {word: tf_value * idf[word] for word, tf_value in tf.items()}
        tf_idf_list.append(tf_idf)
    return tf_idf_list

# 计算并输出结果
tf_idf_result = compute_tf_idf(tokenized_corpus)
for i, doc_tf_idf in enumerate(tf_idf_result):
    print(f'第 {i + 1} 个文档的 TF-IDF:')
    print(doc_tf_idf)