
import os
import pandas as pd
from itertools import combinations
from collections import Counter

def bool_mapping(value):
    """将字符串映射为布尔值或保持原样"""
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ['true', 'yes']:
            return True
        elif value in ['false', 'no']:
            return False
    return value

def normalize_answer(s):
    """标准化答案字符串"""
    if isinstance(s, bool):
        return 'yes' if s else 'no'
    if not isinstance(s, str):
        s = str(s)
    return s.lower().strip()

def cau_f1_score(prediction, ground_truth):
    """自定义F1分数计算方法"""
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = 0

    if (normalized_prediction in ["yes", "no", "noanswer"] and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC
    if (normalized_ground_truth in ["yes", "no", "noanswer"] and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_metrics(folder):
    """主计算函数"""
    file_list = [f for f in os.listdir(folder) if f.endswith('.csv')]
    
    data_dict = {}
    f1_dict = {}
    for fname in file_list:
        df = pd.read_csv(os.path.join(folder, fname))
        data_dict[fname] = df['is_correct'].astype(bool).reset_index(drop=True)
        # 计算每行的F1分数并取平均
        f1_scores = df.apply(lambda row: cau_f1_score(row['prediction'], row['ground_truth']), axis=1)
        f1_dict[fname] = f1_scores.mean()

    total_num = len(next(iter(data_dict.values())))
    stat = []
    for fname, col in data_dict.items():
        true_count = col.sum()
        stat.append({
            'file': fname, 
            'true_ratio': true_count / total_num,
            'f1_score': f1_dict[fname]
        })

    stat_df = pd.DataFrame(stat)
    print("每个文件的true占比和F1分数：")
    print(stat_df)

    results = []
    for f1, f2 in combinations(file_list, 2):
        v1 = data_dict[f1]
        v2 = data_dict[f2]
        intersection = (v1 & v2).sum() / total_num
        union = (v1 | v2).sum() / total_num
        results.append({
            'file1': f1, 'file2': f2,
            'intersection_true_ratio': intersection,
            'union_true_ratio': union,
        })

    result_df = pd.DataFrame(results)
    print("\n两两文件的交集和并集占比：")
    print(result_df)

    all_cols = list(data_dict.values())
    all_intersection = pd.concat(all_cols, axis=1).all(axis=1).sum() / total_num
    all_union = pd.concat(all_cols, axis=1).any(axis=1).sum() / total_num

    print("\n所有文件的交集和并集占比：")
    print(f"所有文件交集true占比: {all_intersection:.4f}")
    print(f"所有文件并集true占比: {all_union:.4f}")

    with pd.ExcelWriter(folder+'/is_correct_stats.xlsx') as writer:
        stat_df.to_excel(writer, sheet_name='单文件')
        result_df.to_excel(writer, sheet_name='两两组合')
        pd.DataFrame({
            'all_intersection_true_ratio': [all_intersection],
            'all_union_true_ratio': [all_union]
        }).to_excel(writer, sheet_name='全部交并集', index=False)

if __name__ == '__main__':
    folder = "emnlp_hotpot"  # 替换为你的文件夹路径
    calculate_metrics(folder)
