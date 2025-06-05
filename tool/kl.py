import re
import string
from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
from typing import Tuple
import logging
from torch.nn.utils.rnn import pad_sequence
import torch
from datetime import datetime
from data_process.musique import musique
from reward_score.qa_em import em_check,subem_check,normalize_answer
from config import train_config
from data_process.hotpot import hotpotqa
# 配置日志和预编译正则表达式
logging.basicConfig(level=logging.INFO)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
prompt_path = train_config["prompt_path"]
max_results = train_config["search_results"]
DATASET_PATH="data/MuSiQue/musique_ans_v1.0_dev.jsonl" #"data/hotpot/hotpot_dev_fullwiki_v1.json"
MAX_TOKENS=1024
OUTPUT_FILE = "pred_csv/hotpot-3b-rag.csv"
MODEL_PATH = "model/Qwen2.5-7B-Instruct"  # 替换为实际模型路径
dataset_name = train_config['dataset_name']
if dataset_name == 'hotpotqa':
    dataset_process= hotpotqa
elif dataset_name == 'musique':
    dataset_process= musique
else:
    raise NotImplementedError
import re
import random

def augment_search_results(formatted_text, max_remove=5):
    """对格式化后的搜索结果进行数据增强，随机移除开头段落"""
    # 分割原始文本为独立段落
    chunks = formatted_text.split("\n\n")
    
    # 移除空字符串并验证段落格式
    valid_chunks = []
    for chunk in chunks:
        if chunk.startswith("[webpage") and chunk.endswith("end]"):
            valid_chunks.append(chunk)
    
    # 随机决定要移除的段落数量（0 到 max_remove 或剩余段落数-1的最小值）
    max_possible_remove = min(max_remove, len(valid_chunks)-1)
    remove_num = random.randint(1, max_possible_remove) #random.randint(0, max_possible_remove)
    
    # 保留剩余段落（如果全部移除则返回空字符串）
    augmented_chunks = valid_chunks[:-remove_num] if remove_num < len(valid_chunks) else []
    
    return "\n\n".join(augmented_chunks)
def shuffle_search_results(text, n=8):
    import re
    import random
    # 使用正则表达式提取所有网页块
    pattern = r'\[webpage \d+ begin\].*?\[webpage \d+ end\]'
    blocks = re.findall(pattern, text, flags=re.DOTALL)
    print("blocks",blocks)
    if n is None or n >= len(blocks):
        blocks_to_shuffle = blocks
        prefix_blocks = []
    else:
        prefix_blocks = blocks[:-n]
        blocks_to_shuffle = blocks[-n:]
    
    # 打乱后n块
    random.shuffle(blocks_to_shuffle)
    
    # 重新编号并替换标签
    result = []
    for idx, block in enumerate(prefix_blocks + blocks_to_shuffle, 1):
        # 替换开始和结束标签中的编号
        new_block = re.sub(
            r'\[webpage \d+ begin\]',
            f'[webpage {idx} begin]',
            block,
            count=1
        )
        new_block = re.sub(
            r'\[webpage \d+ end\]',
            f'[webpage {idx} end]',
            new_block,
            count=1
        )
        result.append(new_block)
    
    # 用两个换行符连接所有块
    return '\n\n'.join(result)
def add_noise_by_word(text, noise_ratio=0.005, noise_str="..."):
    """
    随机对输入文本中的部分英文单词用固定字符串替换。
    :param text: 原始字符串
    :param noise_ratio: 替换比例（0~1之间），如0.1表示10%单词被干扰
    :param noise_str: 用于干扰的固定字符串
    :return: 被添加噪声后的字符串
    """
    matches = list(re.finditer(r'[a-zA-Z]+', text))
    num_words = len(matches)
    print("num_words",num_words)
    num_noisy = max(0, int(num_words * noise_ratio)) if num_words > 0 else 0
    print("num_noisy",num_noisy)
    noisy_indices = set(random.sample(range(num_words), num_noisy)) if num_words > 0 else set()
    print("noisy_indices",noisy_indices)
    result = []
    last_end = 0
    for idx, match in enumerate(matches):
        start, end = match.span()
        result.append(text[last_end:start])
        if idx in noisy_indices:
            result.append(noise_str)
        else:
            result.append(text[start:end])
        last_end = end
    result.append(text[last_end:])
    return ''.join(result)

def main():
    # 1. 加载数据并预处理
    logging.info("Loading dataset...")
    dataset = dataset_process(DATASET_PATH,max_results)[:10]
    with open(prompt_path[0], "r", encoding="utf-8") as f:
        system_prompt_template = f.read()
    
    # 2. 初始化模型（根据GPU情况调整并行度）
    logging.info("Initializing model...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.4,  # 提高显存利用率
        # enforce_eager=True,          # 对于小模型可以加速
        #enable_prefix_caching= True,
    )
    
    # 3. 准备所有prompts（批量处理提高效率）
    logging.info("Preparing prompts...")
    
    # current_date = datetime.now().strftime("%Y-%m-%d")
        
    # 批量生成prompts

    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)
    tokenizer = llm.get_tokenizer()
    all_prompts = []
    prompt_pair_indices = []  # 记录每两种prompt属于哪条数据
    for idx, item in enumerate(dataset):
        search = add_noise_by_word(item['P'])
        # print("search",search)
        # search = augment_search_results(item['P'])
        if '{search_results}' in system_prompt_template:
            dynamic_prompt = system_prompt_template.format(search_results=item['P'])
            dynamic_prompt_shuf = system_prompt_template.format(search_results=search)
            
        else:
            dynamic_prompt = system_prompt_template
            dynamic_prompt_shuf = system_prompt_template
        prompt1 = tokenizer.apply_chat_template([
            {"role": "system", "content": dynamic_prompt},
            {"role": "user", "content": item['Q']}], tokenize=False, add_generation_prompt=True)
        prompt2 = tokenizer.apply_chat_template([
            {"role": "system", "content": dynamic_prompt_shuf},
            {"role": "user", "content": item['Q']}], tokenize=False, add_generation_prompt=True)
        all_prompts.append(prompt1)
        all_prompts.append(prompt2)
        prompt_pair_indices.append(idx)  # 两个prompt对应同一数据

    

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
    results = []
    for i in range(0, len(outputs), 2):
        # outputs[i]和outputs[i+1]一组
        results.append((outputs[i].outputs[0].text, outputs[i+1].outputs[0].text))
    kl_results = []
    for i, ((ans1, ans2), idx) in enumerate(zip(results, prompt_pair_indices)):
        # 编码答案和prompt
        ans_token_ids1 = tokenizer(ans1, return_tensors="pt", padding=True, add_special_tokens=False)['input_ids']
        ans_token_ids2 = tokenizer(ans2, return_tensors="pt", padding=True, add_special_tokens=False)['input_ids']
        input_token_ids1 = tokenizer(all_prompts[2*i], return_tensors="pt", padding=True, add_special_tokens=False)['input_ids']
        input_token_ids2 = tokenizer(all_prompts[2*i], return_tensors="pt", padding=True, add_special_tokens=False)['input_ids']
        plen1 = input_token_ids1.shape[1]
        plen2 = input_token_ids2.shape[1]
        # 合并prompt+answer
        merged_ids1 = torch.cat([input_token_ids1[0], ans_token_ids1[0]])
        merged_ids2 = torch.cat([input_token_ids2[0], ans_token_ids2[0]])
        # print("merged_ids1",merged_ids1)
        # print("merged_ids2",merged_ids2)
        # pad对齐
        # min_len = min(len(merged_ids1), len(merged_ids2))
        # merged_ids1 = merged_ids1[:min_len]
        # merged_ids2 = merged_ids2[:min_len]
        # 获取logprobs
        zz1 = llm.generate(prompt_token_ids=[merged_ids1.tolist()], sampling_params=gen_logps_sp, use_tqdm=False)
        zz2 = llm.generate(prompt_token_ids=[merged_ids2.tolist()], sampling_params=gen_logps_sp, use_tqdm=False)
        logprobs1 = torch.tensor([list(x.values())[0].logprob for x in zz1[0].prompt_logprobs[plen1:]])
        logprobs2 = torch.tensor([list(x.values())[0].logprob for x in zz2[0].prompt_logprobs[plen2:]])
        # print("logprobs1",logprobs1)
        # print("logprobs2",logprobs2)
        # 对齐长度
        min_len = min(len(logprobs1), len(logprobs2))
        logprobs1 = logprobs1[:min_len]
        logprobs2 = logprobs2[:min_len]
        # print("a-logprobs1",logprobs1)
        # print("a-logprobs2",logprobs2)
        # KL散度：KL(P1||P2) = sum(P1 * (logP1 - logP2)), 但logprobs已是logP
        p1 = torch.exp(logprobs1)
        p2 = torch.exp(logprobs2)
        kl = torch.sum(p1 * (logprobs1 - logprobs2)).item()
        kl_results.append(kl)
        print(f"Sample {i+1}: KL divergence = {kl:.4f}")

    # 可选：输出所有KL值
    print("All KL results:", kl_results)
    print(f"Average KL divergence: {np.mean(kl_results):.4f}")


if __name__ == "__main__":
    main()
