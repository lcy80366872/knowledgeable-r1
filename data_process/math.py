from datasets import load_dataset
from math_verify import parse, verify, ExprExtractionConfig
import re
def gsm8k(data_path,_): 
    dataset = load_dataset(data_path, split="train")
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]

    return QAs
def reward_correct(answer,gt):
    # pattern = r'\d+\.\d+|\d+/\d+|\d+'
    # nums = re.findall(pattern, answer) 
    # if len(nums) == 0: return -1.0
    # lastnum = nums[-1]
    ans = parse(answer, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(gt, extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1