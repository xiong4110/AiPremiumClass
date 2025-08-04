from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 示例数据
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "Quick brown foxes leap over lazy dogs in summer"
]

# 停用词列表
stop_words = ['the', 'over', 'in']

# 使用停用词处理后，计算TF-IDF
vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = vectorizer.fit_transform(documents)
# ['brown' 'dog' 'dogs' 'fox' 'foxes' 'jump' 'jumps' 'lazy' 'leap' 'never' 'quick' 'quickly' 'summer']
print(vectorizer.get_feature_names_out())

# 计算余弦相似度
cosine_similarities = cosine_similarity(tfidf_matrix)

# 打印相似度矩阵
print("Cosine Similarity Matrix:")
print(cosine_similarities)