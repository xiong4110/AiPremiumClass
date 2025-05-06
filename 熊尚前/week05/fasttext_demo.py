import fasttext

# 准备训练数据
# 定义一个包含多个句子的列表，用于后续训练词向量模型
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "A quick brown dog outpaces a quick fox"
]

# 将训练数据写入到 train.txt 文件中，每个句子占一行
with open('train.txt', 'w', encoding='utf-8') as f:
    for sentence in sentences:
        f.write(sentence + '\n')

# 训练模型
# 使用 fasttext 的无监督学习方法训练词向量模型
# 'train.txt' 为训练数据文件路径
# epoch=10 表示训练的轮数为 10 轮
# lr=0.1 表示学习率为 0.1
# dim=100 表示词向量的维度为 100
# wordNgrams=2 表示考虑的词 n-gram 为 2
# minCount=1 表示词频至少为 1 的词才会被考虑
model = fasttext.train_unsupervised('train.txt', epoch=10, lr=0.1, dim=100, wordNgrams=2, minCount=1) 

# 保存模型
# 将训练好的模型保存到 word_vectors.bin 文件中
model.save_model('word_vectors.bin')  # 保存模型到文件

# 获取某个词的词向量
# 定义要获取词向量的目标词
word = "quick"
# 调用模型的 get_word_vector 方法获取指定词的词向量
word_vector = model.get_word_vector(word)  # 获取词向量
# 打印目标词及其对应的词向量
print(f'Vector for {word}:{word_vector}')  # 打印词向量