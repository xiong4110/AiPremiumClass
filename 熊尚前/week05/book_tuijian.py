import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data(filename):
    # 图书评论信息集合
    book_comments = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for item in reader:
            book = item['book']
            comment = item['body']
            comment_words = jieba.lcut(comment)
            
            if book == '':continue

            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comment_words)
    return book_comments

if __name__ == '__main__':
    # 加载停用词表
    stop_words = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8')]

    # 加载图书评论区
    book_comments = load_data('douban_comments_fixed.txt')

    book_names = []
    book_comms = []
    for book, comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)
    # 构建TF-IDF特征矩阵
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([' '.join(comments) for comments in book_comms])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    # print(similarity_matrix)

    # 输入要推荐的图书名称
    book_list = list(book_comments.keys())
    print(book_list)
    book_name = input('请输入图书名称：')
    book_idx = book_names.index(book_name)

    recommend_book_index = np.argsort(similarity_matrix[book_idx])[::-1][1:6]
    for idx in recommend_book_index:
        print(book_names[idx])

