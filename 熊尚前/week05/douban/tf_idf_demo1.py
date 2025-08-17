# 导入必要的库
import csv  # 用于读取CSV文件
import jieba  # 中文分词库
from sklearn.feature_extraction.text import TfidfVectorizer  # 用于计算TF-IDF值
from sklearn.metrics.pairwise import cosine_similarity  # 用于计算余弦相似度
import numpy as np  # 用于数值计算


def load_data(filePath):
    """
    加载图书评论数据
    
    参数:
        filePath: 数据文件路径
    
    返回:
        book_comments: 字典，键为书名，值为该图书的所有评论分词列表
    """
    # 图书评论集合 {书名: [评论分词列表]}
    book_comments = {}

    with open(filePath, 'r', encoding='utf-8') as f:
        # 使用csv.DictReader读取文件，指定分隔符为制表符\t
        reader = csv.DictReader(f, delimiter='\t')
        for item in reader:
            book = item['book']  # 获取书名
            comment = item['body']  # 获取评论内容
            comment_words = jieba.lcut(comment)  # 对评论进行分词
            if book == '':  # 跳过空书名
                continue

            # 将当前评论的分词结果添加到对应图书的列表中
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comment_words)

    return book_comments


if __name__ == '__main__':
    # 加载停用词列表
    stop_words = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8')]

    # 加载图书评论信息
    book_comments = load_data('fixed_book_common.txt')
    
    # 提取书名和评论文本
    book_names = []  # 存储所有书名
    book_comms = []  # 存储每本书的评论分词列表
    for book, comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)

    # 构建TF-IDF矩阵
    # TfidfVectorizer会将文本转换为TF-IDF特征矩阵
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    # 对每本书的评论分词列表进行连接，形成字符串，然后构建TF-IDF矩阵
    tfidf_matrix = vectorizer.fit_transform(' '.join(commons) for commons in book_comms)

    # 计算图书之间的余弦相似度
    # 余弦相似度矩阵的每个元素(i, j)表示第i本书与第j本书的相似度
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 输入要推荐的图书名称
    book_list = list(book_comments.keys())
    print("可用的图书列表:", book_list)

    book_name = input('请输入图书名称: ')
    try:
        book_idx = book_list.index(book_name)  # 获取图书索引
    except ValueError:
        print(f"错误: 找不到图书 '{book_name}'")
        exit()

    # 获取与输入图书最相似的图书
    # np.argsort对相似度进行排序，取前10个最相似的图书(排除自身)
    recomment_book_index = np.argsort(-similarity_matrix[book_idx])[1:11]
    
    # 输出推荐的图书
    print(f"\n与《{book_name}》相似的图书推荐:")
    for idx in recomment_book_index:
        print(f'《{book_list[idx]}》\t 相似度: {similarity_matrix[book_idx][idx]:.4f}')







