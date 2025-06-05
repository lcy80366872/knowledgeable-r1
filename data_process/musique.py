import json
def format_search_results(paragraphs, max_results=10):
    formatted = []
    for idx, p in enumerate(paragraphs, 1):
        if idx >=max_results+1:  #默认取前10个检索结果
            break
        content = f"[webpage {idx} begin]\n"
        content += f"Title: {p.get('title', '')}\n"
        content += f"Content: {p.get('paragraph_text', '')}\n"
        content += f"[webpage {idx} end]"
        formatted.append(content)
    return "\n\n".join(formatted)

def musique(data_path,max_results): 

    QAs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'question' in data and 'answer' in data and 'paragraphs' in data:
                QAs.append({
                    'Q': data['question'],
                    'P': format_search_results(data['paragraphs'],max_results),  # paragraphs是list类型
                    'A': data['answer']
                })
    return QAs