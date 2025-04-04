# -*- coding: utf-8 -*-
"""
这个代码的作用是将豆瓣读书评论数据中的书名和评论分开，并将其写入一个新的文件中。
"""
# 修复后的文件
fixed = open('熊尚前\week05\douban_comments_fixed.txt', 'w', encoding='utf-8')

# 读取文件

lines = [line for line in open('熊尚前\week05\douban_book.txt', 'r', encoding='utf-8')]

for i,line in enumerate(lines):
    # 保存标题列
    if i == 0:
        fixed.write(line)
        pre_line = ''
        continue
    # 提取书名和评论
    terms = line.split('\t')
    if terms[0] == pre_line.split('\t')[0]:
        if len(pre_line.split('\t')) == 6: 
            fixed.write(pre_line + '\n')
            pre_line = line.strip() # 保存当前行
        else:
            pre_line = '' 
    else:
        if len(terms) == 6:
            #fixed.write(line + '\n')
            pre_line = line.strip() # 保存当前行
        else:
            pre_line += line.strip() # 合并当前行和上一行
fixed.close()