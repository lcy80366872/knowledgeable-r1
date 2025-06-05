import json

# 读取jsonl文件
jsonl_file = 'para_with_hyperlink.jsonl'

# 初始化一个列表存储所有数据
data_list = []

# 读取文件并解析每一行
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        # 将每一行转换为字典
        data = json.loads(line.strip())
        data_list.append(data)

# 如果文件不为空，获取第一条数据作为示例
if data_list:
    sample_data = data_list[0]
    # 打印所有属性（键）
    print("所有属性（字段）:", list(sample_data.keys()))
    # 打印示例数据
    print("示例数据:", sample_data)
else:
    print("文件为空或没有数据。")