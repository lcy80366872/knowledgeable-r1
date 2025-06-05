import re
from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import Tuple
import logging
from config import train_config
# 配置参数
MODEL_PATH = "save_model/step_500"  # 替换为实际模型路径
DATASET_PATH = "data/gsm8k/test-00000-of-00001.parquet"  # 替换为数据集路径
BATCH_SIZE = 16  # 根据显存调整
MAX_TOKENS = 1024
prompt_path = train_config["prompt_path"]
dataset_name = train_config['dataset_name']
OUTPUT_FILE = "pred_math/predictions.csv"
with open(prompt_path[0], "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
def debug_parser(text: str) -> dict:
    """解析调试信息"""
    return {
        "raw_text": text,
        "has_answer_tag": bool(re.search(r"<answer>", text, re.IGNORECASE)),
        "found_numbers": re.findall(r"\d+\.\d+|\d+/\d+|\d+", text),
        "last_number": re.findall(r"\d+\.\d+|\d+/\d+|\d+", text)[-1] if re.findall(r"\d+\.\d+|\d+/\d+|\d+", text) else None
    }

def load_dataset(path: str) -> Tuple[list[str], list[str]]:
    """加载数据集并验证格式"""
    try:
        df = pd.read_parquet(path)
        if 'answer' not in df.columns or 'question' not in df.columns:
            raise ValueError("Dataset missing required columns")
        
        ground_truths = []
        for ans in df['answer']:
            if "####" not in ans:
                logging.warning(f"Missing separator in answer: {ans}")
            parts = ans.split("####")
            ground_truths.append(parts[-1].strip())
            
        return df['question'].tolist(), ground_truths
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise
from simpleeval import simple_eval
def parse_answer(text: str) -> str:
    """增强的答案解析函数"""
    # 提取<answer>标签内容（支持大小写和换行）
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    content = match.group(1).strip()
    
    # 预处理步骤
    processed = content.replace(',', '')               # 移除千分位逗号
    processed = processed.replace(' ', '')              # 移除空格
    processed = processed.replace('%', '/100')          # 转换百分号
    processed = processed.replace('−', '-')             # 统一减号格式
    
    # 匹配数学表达式和数字
    patterns = [
        r'^[$]?[\d\.]+[$]?$',                        # 简单数字
        r'^[\d\.]+[+\-*/][\d\.]+$',                     # 基本四则运算
        r'^[\d\.]+[+\-*/][\d\.]+[+\-*/][\d\.]+$',      # 复合运算
        r'^\d+/\d+$'                                    # 分数
    ]
    
    # 查找所有可能匹配项
    candidates = []
    for pattern in patterns:
        candidates += re.findall(pattern, processed)
    
    if not candidates:
        return None
    
    # 尝试解析最后一个候选值
    last_candidate = candidates[-1]
    
    try:
        # 使用安全表达式计算
        value = simple_eval(last_candidate)
        return f"{value:.10f}".rstrip('0').rstrip('.')  # 统一数字格式
    except:
        try:
            # 尝试直接转换为浮点数
            return f"{float(last_candidate):.10f}".rstrip('0').rstrip('.')
        except:
            return None

def compare_answers(pred: str, true: str) -> bool:
    """增强的比较函数"""
    def normalize(s):
        s = str(s).strip().lower()
        s = s.replace(',', '').replace(' ', '')
        
        # 处理特殊表示
        s = s.replace('k', 'e3').replace('m', 'e6')
        s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)  # 处理10k→10*1e3
        s = s.replace('%', '*0.01')
        
        try:
            return round(float(simple_eval(s)), 6)
        except:
            return None
    
    pred_norm = normalize(pred) if pred else None
    true_norm = normalize(true)
    
    if pred_norm is None or true_norm is None:
        return False
    
    return np.isclose(pred_norm, true_norm, atol=1e-6)
def main():
    # 1. 加载数据
    print("Loading dataset...")
    questions, ground_truths = load_dataset(DATASET_PATH)
    
    # 2. 初始化模型
    print("Loading model...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,  # 根据GPU数量调整
        gpu_memory_utilization=0.8
    )
    
    # 3. 配置生成参数
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        #stop=["</answer>"]
    )
    tokenizer = llm.get_tokenizer()
    # 4. 准备提示模板
    def format_prompt(question):
        return tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ], tokenize=False, add_generation_prompt=True)
    
    # 5. 分批推理
    print("Generating answers...")
    results = []
    try:
        for i in tqdm(range(0, len(questions), BATCH_SIZE)):
            batch_questions = questions[i:i+BATCH_SIZE]
            prompts = [format_prompt(q) for q in batch_questions]
            
            outputs = llm.generate(prompts, sampling_params)
            for q, gt, output in zip(batch_questions, ground_truths[i:i+BATCH_SIZE], outputs):
                    raw_text = output.outputs[0].text
                    parsed_pred = parse_answer(raw_text)
                    debug_info = debug_parser(raw_text)
                    
                    results.append({
                        "question": q,
                        "ground_truth": gt,
                        "prediction": parsed_pred,
                        "is_correct": compare_answers(parsed_pred, gt) if parsed_pred else False,
                        "raw_response": raw_text,
                        **debug_info
                    })
    except Exception as e:
        logging.error(f"Generation failed: {str(e)}")
        return
    # 6. 解析和评估
    # 保存结果
    logging.info("Saving results...")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Results saved to {OUTPUT_FILE}")

    # 计算指标
    valid_results = df[df['prediction'].notnull()]
    format_error_rate = 1 - len(valid_results)/len(df)
    accuracy = valid_results['is_correct'].mean()

    print(f"\nFinal Metrics:")
    print(f"Format Success Rate: {(1-format_error_rate):.2%}")
    print(f"Accuracy (valid samples): {accuracy:.2%}")
    print(f"Absolute Accuracy: {(valid_results['is_correct'].sum()/len(df)):.2%}")

if __name__ == "__main__":
    main()