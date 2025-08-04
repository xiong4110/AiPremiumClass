

# 修复后内容存盘文件
fixed = open('fixed_book_common.txt', 'w', encoding='utf-8')

# 修复前内容读取
lines = [line for line in open('../douban_book.txt', 'r', encoding='utf-8')]

for i, line in enumerate(lines):
    if i == 0:
        fixed.write(line) # 写标题
        pre_line = '' 
        continue
    # 提取书名和文本
    items = line.split('\t')
    if items[0] == pre_line.split('\t')[0]:
        if (len(pre_line.split('\t')) == 6):
            fixed.write(pre_line + '\n') # 如果当前行与上一行的书名相同，则说明上一行的信息时全的，直接写入到新文件
            pre_line = line.strip() # 更新上一行的信息
        else:
            pre_line = ''
    else:
        if len(items) == 6:
            pre_line = line.strip()
        else:    
            pre_line += line.strip()
fixed.close()
