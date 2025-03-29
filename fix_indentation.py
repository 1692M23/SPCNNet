with open('preprocessdata7.py', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 修复第1236行的缩进
if len(lines) >= 1236:
    if lines[1235].strip().startswith('if z == 0 and'):
        # 获取if语句的缩进级别
        spaces = len(lines[1235]) - len(lines[1235].lstrip())
        # 检查下一行的缩进是否正确
        next_line = lines[1236]
        if next_line.strip().startswith('z = matches'):
            # 需要增加缩进
            proper_indent = ' ' * (spaces + 4)  # 增加一级缩进（4个空格）
            fixed_line = proper_indent + next_line.lstrip()
            lines[1236] = fixed_line
            print(f"修复了第1236行的缩进")

# 保存修改后的文件
with open('preprocessdata7.py', 'w', encoding='utf-8') as file:
    file.writelines(lines)

print("处理完成") 