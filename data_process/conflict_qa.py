

import json


def conflict(data_path, max_results=10):
    """主处理函数，保持与原始定义完全一致"""
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    
    QAs = []
    for item in dataset:
        
        
        QAs.append({
            "Q": item['question'],
            "P": item['context'],
            "A": item['answer']
        })
    return QAs
def conflict_mix(data_path, max_results=10):
    """主处理函数，保持与原始定义完全一致"""
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    
    QAs = []
    for item in dataset:
        
        
        QAs.append({
            "Q": item['question'],
            "P": item['mixcontext'],
            "A": item['answer']
        })
    return QAs
