import json
import os
import warnings
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

# ------------------- 基础工具函数 -------------------
def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]

def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval().cuda()
    return model.half() if use_fp16 else model, AutoTokenizer.from_pretrained(model_path, use_fast=True)

def pooling(pooler_output, last_hidden_state, attention_mask, method="mean"):
    if method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif method == "cls":
        return last_hidden_state[:, 0]
    elif method == "pooler":
        return pooler_output
    else:
        raise ValueError(f"Unsupported pooling method: {method}")

# ------------------- 编码器类 -------------------
class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length=256, use_fp16=False):
        self.model, self.tokenizer = load_model(model_path, use_fp16)
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts: List[str], is_query=True) -> np.ndarray:
        if "e5" in self.model_name.lower():
            texts = [f"query: {t}" if is_query else f"passage: {t}" for t in texts]
        elif "bge" in self.model_name.lower() and is_query:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

        inputs = self.tokenizer(texts, 
                               max_length=self.max_length,
                               padding=True,
                               truncation=True,
                               return_tensors="pt").to("cuda")

        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros((inputs['input_ids'].shape[0], 1), dtype=torch.long).cuda()
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids)
            emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs)
            emb = pooling(output.pooler_output, 
                         output.last_hidden_state,
                         inputs['attention_mask'],
                         self.pooling_method)
            if "dpr" not in self.model_name.lower():
                emb = torch.nn.functional.normalize(emb, dim=-1)

        return emb.detach().cpu().numpy().astype(np.float32)

# ------------------- 检索器基类 -------------------
class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.topk = config.retrieval_topk
        self.corpus = load_corpus(config.corpus_path)

    def batch_search(self, queries: List[str], num: int = None) -> List[List[Dict]]:
        raise NotImplementedError

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(config.index_path)
        if config.faiss_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.encoder = Encoder(
            model_name=config.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            use_fp16=config.retrieval_use_fp16
        )
        self.batch_size = config.retrieval_batch_size

    def batch_search(self, queries: List[str], num: int = None) -> List[List[Dict]]:
        num = num or self.topk
        all_results = []
        
        for i in tqdm(range(0, len(queries), self.batch_size), desc="Searching"):
            batch = queries[i:i+self.batch_size]
            embs = self.encoder.encode(batch)
            scores, indices = self.index.search(embs, num)
            
            batch_docs = []
            for idxs in indices:
                batch_docs.append(load_docs(self.corpus, idxs))
            
            all_results.extend(batch_docs)
        return all_results

# ------------------- 主程序 -------------------
def main():
    parser = argparse.ArgumentParser(description="文档检索系统")
    parser.add_argument("--index", type=str, required=True, help="FAISS索引路径")
    parser.add_argument("--corpus", type=str, required=True, help="语料库JSONL路径")
    parser.add_argument("--queries", type=str, required=True, help="查询JSONL文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出JSON路径")
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5", help="检索模型名称或路径")
    parser.add_argument("--topk", type=int, default=5, help="检索文档数量")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")

    args = parser.parse_args()

    # 配置参数
    class Config:
        def __init__(self):
            self.retrieval_method = "dense"
            self.retrieval_topk = args.topk
            self.index_path = args.index
            self.corpus_path = args.corpus
            self.faiss_gpu = args.gpu
            self.retrieval_model_path = args.model
            self.retrieval_pooling_method = "cls"
            self.retrieval_use_fp16 = True
            self.retrieval_batch_size = args.batch_size

    # 初始化检索器
    retriever = DenseRetriever(Config())

    # 加载查询数据
    print("🔍 加载查询数据...")
    with open(args.queries) as f:
        dataset = [json.loads(line) for line in f]

    queries = [item["question"] for item in dataset]

    # 执行检索
    print("🚀 开始检索...")
    results = retriever.batch_search(queries)

    # 构建输出格式
    print("📦 格式化输出...")
    output_data = []
    for orig_item, docs in zip(dataset, results):
        context = []
        for doc in docs:
            # 解析文档内容
            title = doc.get("title", "未知标题")
            content = doc.get("text", doc.get("contents", ""))
            
            # 分割段落（假设原始文本用换行符分隔）
            paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
            
            # 如果没有有效段落，使用整个内容作为段落
            if not paragraphs:
                paragraphs = [content.strip()] if content.strip() else []
            
            context.append([title, paragraphs])
        
        output_item = {
            "question": orig_item["question"],
            "answer": orig_item["answer"],
            "search": context,
        }
        output_data.append(output_item)

    # 保存结果
    print("💾 保存结果到", args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

    #输入格式
#     {
#   "question": "问题文本",
#   "answer": "答案文本",
#   "id": "唯一标识符（可选）",
#   "supporting_facts": ["相关事实列表（可选）"]
# }
# 命令示例
# python retrieval_system.py \
#   --index path/to/faiss.index \
#   --corpus path/to/corpus.jsonl \
#   --queries path/to/questions.jsonl \
#   --output path/to/output.json \
#   --model BAAI/bge-base-zh-v1.5 \
#   --topk 5 \
#   --batch_size 64 \
#   --gpu

# DATA_NAME=hotpot

# DATASET_PATH="/workspace/work/exp/grpo_kl/data/$DATA_NAME/hotpot_train_v1.1.json"

# TOPK=3

# INDEX_PATH=workspace/work/exp/grpo_kl/wiki/wiki-18
# CORPUS_PATH=workspace/work/exp/grpo_kl/wiki/wiki-18.jsonl
# SAVE_NAME=e5_${TOPK}_wiki18.json



# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python retrieval.py --retrieval_method e5 \
#                     --topk $TOPK \
#                     --index $INDEX_PATH \
#                     --corpus $CORPUS_PATH \
#                     --queries $DATASET_PATH \
#                     --output $SAVE_NAME \
#                     --model "wiki" \
#                     --batch_size 64 \
#                     --gpu
#输出文件示例
# {
#   "question": "问题文本",
#   "answer": "答案文本",
#   "context": [
#     ["文档标题1", ["段落1", "段落2", ...]],
#     ["文档标题2", ["段落1", ...]],
#     // ...最多topk个文档
#   ],
#   "_id": "原始ID",
#   "supporting_facts": ["相关事实"]
# }