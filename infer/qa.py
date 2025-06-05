import re
import string
from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
from typing import Tuple
import logging
from datetime import datetime
from data_process.musique import musique
from reward_score.qa_em import em_check,subem_check,normalize_answer
from config import train_config
from data_process.hotpot import hotpotqa
from data_process.choice import choice
from data_process.conflict_qa import conflict
# 配置日志和预编译正则表达式
logging.basicConfig(level=logging.INFO)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
prompt_path = train_config["prompt_path"]
max_results = train_config["search_results"]
dataset_name = train_config['dataset_name']
DATASET_PATH=  "data/conflict_qa/ConFiQA-MR-test.json"#"data/MuSiQue/musique_ans_v1.0_dev.jsonl" #"data/hotpot/hotpot_dev_fullwiki_v1.json"
MAX_TOKENS=1024
OUTPUT_FILE = "emnlp_conflict/3b-intern-MR.csv"
MODEL_PATH = "model/Qwen2.5-3B-Instruct"  # 替换为实际模型路径
if dataset_name == 'hotpotqa':
    dataset_process= hotpotqa
elif dataset_name == 'musique':
    dataset_process= musique
elif dataset_name == 'choice':
    dataset_process= choice
elif dataset_name == 'conflictqa':
    dataset_process= conflict
else:
    raise NotImplementedError
def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s
def parse_answer(text: str) -> str:
    """使用预编译正则表达式提高解析速度"""
    match = ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else None

def exact_match_score(prediction, ground_truth):
    # 统一处理ground_truth为列表形式
    truths = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    normalized_pred = normalize_answer(bool_mapping(prediction))
    
    for truth in truths:
        if normalized_pred == normalize_answer(bool_mapping(truth)):
            return True
    return False
def main():
    # 1. 加载数据并预处理
    logging.info("Loading dataset...")
    dataset = dataset_process(DATASET_PATH,max_results)
    with open(prompt_path[0], "r", encoding="utf-8") as f:
        system_prompt_template = f.read()
    
    # 2. 初始化模型（根据GPU情况调整并行度）
    logging.info("Initializing model...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,  # 提高显存利用率
        enforce_eager=True,          # 对于小模型可以加速
        enable_prefix_caching= True,
    )
    
    # 3. 准备所有prompts（批量处理提高效率）
    logging.info("Preparing prompts...")
    tokenizer = llm.get_tokenizer()
    # current_date = datetime.now().strftime("%Y-%m-%d")
        
    # 批量生成prompts
    all_prompts = []
        
    for item in dataset:
        if '{search_results}' in system_prompt_template:
            dynamic_system_prompt = system_prompt_template.format(search_results=item['P'])
        else:
            dynamic_system_prompt = system_prompt_template
        all_prompts.append(tokenizer.apply_chat_template([
            {"role": "system", "content": dynamic_system_prompt},
            {"role": "user", "content": item['Q']}], tokenize=False, add_generation_prompt=True))
    

    # 4. 一次性生成所有结果
    logging.info("Generating answers...")
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        skip_special_tokens=True,  # 避免生成特殊token
    )

    
    outputs = llm.generate(all_prompts, sampling_params)
    results = []
    for item, output in zip(dataset, outputs):
       
        raw_text = output.outputs[0].text
        parsed_pred = parse_answer(raw_text)
        gt = item['A']
        
        results.append({
            "question": item['Q'],
            "ground_truth": gt,
            "prediction": parsed_pred,
            "is_correct": exact_match_score(parsed_pred,gt) if parsed_pred else False,
            "raw_response": raw_text,
        })
            


    # 5. 保存结果并计算指标
    logging.info("Analyzing results...")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    valid_results = df[df['prediction'].notnull()]
    total_samples = len(df)
    
    metrics = {
        "format_success_rate": len(valid_results) / total_samples,
        "valid_accuracy": valid_results['is_correct'].mean(),
        "absolute_accuracy": valid_results['is_correct'].sum() / total_samples
    }

    print(f"\nPerformance Metrics:")
    print(f"1. Format Success Rate: {metrics['format_success_rate']:.2%}")
    print(f"2. Accuracy (Valid): {metrics['valid_accuracy']:.2%}")
    print(f"3. Absolute Accuracy: {metrics['absolute_accuracy']:.2%}")

if __name__ == "__main__":
    main()
