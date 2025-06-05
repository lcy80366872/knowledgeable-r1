
import pandas as pd
import json
from typing import List, Dict

def format_search_results(contexts: List[Dict], max_results: int) -> str:
    """格式化检索结果为指定文本格式"""
    return "\n".join(
        f"[webpage {ctx['idx']+1} begin]\n{ctx['context']}\n[webpage {ctx['idx']+1} end]"
        for ctx in contexts[:max_results]
    )

def choice(data_path: str, max_results: int = 5) -> List[Dict]:
    """处理xlsx问答数据并生成结构化输出"""
    df = pd.read_excel(data_path)
    QAs = []
    
    for _, row in df.iterrows():
        # 处理问题和选项
        question = row['query']
        options = eval(row['option']) if isinstance(row['option'], str) else row['option']
        option_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
        
        # 处理检索上下文
        contexts = []
        for i in range(1, 6):
            context = row.get(f'top{i}', '')
            if pd.notna(context):
                contexts.append({
                    "idx": i-1,
                    "title": f"Result {i}",
                    "context": str(context)
                })
        
        QAs.append({
            "Q": f"{question}\n{option_text}",
            "P": format_search_results(contexts, max_results),
            "A": row['answer']
        })
    
    return QAs

def save_to_json(data: List[Dict], output_path: str):
    """将处理结果保存为JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 使用示例
if __name__ == "__main__":
    qa_data = process_qa_data('input.xlsx')
    save_to_json(qa_data, 'output.json')
