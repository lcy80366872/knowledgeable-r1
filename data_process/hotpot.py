import json
def format_search_results(paragraphs, max_results=10):
    """格式化搜索结果，支持自定义最大条目数并优化内容可读性"""
    formatted = []
    for idx, p in enumerate(paragraphs[:max_results], 1):
        # 清洗标题和内容（示例：移除换行符和多余空格）
        title = p.get("title", "").replace("\n", " ").strip()
        context = p.get("context", "").replace("\n", " ").strip()
        # 构建结构化条目
        entry = f"[webpage {idx} begin]\nTitle: {title}\nContent: {context}\n[webpage {idx} end]"
        formatted.append(entry)
    return "\n\n".join(formatted)

def hotpotqa(data_path,max_results): 
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    QAs = []
    for item in dataset:
        processed_context = []
        for idx, entry in enumerate(item['context']):
            # 解析二维结构（标题 + 段落列表）
            if isinstance(entry, list) and len(entry) >= 2:
                title = str(entry[0])
                paragraphs = entry[1] if isinstance(entry[1], list) else []
            else:
                title, paragraphs = "Unknown", []
            
            # 合并段落
            context_text = " ".join(paragraphs).strip()
            
            processed_context.append({
                "idx": idx,
                "title": title,
                "context": context_text
            })
        
        QAs.append({
            "Q": item['question'],
            "P": format_search_results(processed_context,max_results),
            "A": item['answer']
        })
    return QAs
