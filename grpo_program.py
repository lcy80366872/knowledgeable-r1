from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb,traceback
import pdb
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from config import train_config, ds_config
from ref_server_old import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list
from transformers import GenerationConfig
from data_process.hotpot import hotpotqa
from data_process.musique import musique
from data_process.conflict_qa import conflict
#from data_process.math import gsm8k,reward_correct
from data_process.choice import choice
from reward_score.qa_em import em_check,cau_f1_score,exact_match_score,subem_check
from tqdm import tqdm
from datetime import datetime
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ["NCCL_P2P_DISABLE"]="1"


wandb_name = train_config['wandb_name']
wandb_project = train_config['wandb_project']
wandb_key = train_config['wandb_key']
wandb_offline = train_config['wandb_offline']
model_path = train_config['model_path']
save_path = train_config['save_path']
record_path = train_config['record_path']
gen_data_path = train_config['gen_data_path']
gen_device = train_config['gen_device']   
all_steps = train_config['all_steps']
Q_batch_size = train_config['Q_batch_size']
num_pre_Q = train_config['num_pre_Q']
train_batch_size = train_config['train_batch_size']
gen_update_steps = train_config['gen_update_steps']
save_steps = train_config['save_steps']
compute_gen_logps = train_config['compute_gen_logps']
clip_param = train_config['clip_param']
ref_server = train_config['ref_server']
dataset_name = train_config['dataset_name']
beta = train_config['beta']
prompt_paths = train_config["prompt_path"]
search_results = train_config["search_results"]
sample_max_tokens = train_config["sample_max_tokens"]
random_remove = train_config["random_remove"]
use_ref_kl= train_config["use_ref_kl"]
noise_ratio =train_config['noise_ratio']
global update_model_num
update_model_num = 0
token_level_loss = train_config['token_level_loss']
os.environ["WANDB_API_KEY"] = wandb_key
# generation_config = GenerationConfig(
#             max_new_tokens=600,
#             temperature=0)
if dataset_name == 'hotpotqa':
    dataset_process= hotpotqa
    reward_acc =exact_match_score
elif dataset_name == 'musique':
    dataset_process= musique
    reward_acc =exact_match_score
elif dataset_name == 'choice':
    dataset_process= choice
    reward_acc =exact_match_score
# elif dataset_name == 'gsm8k':
#     dataset_process= gsm8k
#     reward_acc =reward_correct
elif dataset_name == 'conflictqa':
    dataset_process= conflict
    reward_acc =exact_match_score
else:
    raise NotImplementedError
def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0])     
    data['inputs'] = bytes_to_tensor(dd[1])  
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    data['gen_logps'] = bytes_to_tensor(dd[4])
    data['acc_scores'] = bytes_to_tensor(dd[5])
    data['format_scores'] = bytes_to_tensor(dd[6])
    data['f1_scores'] = bytes_to_tensor(dd[7])
    data['is_aguments'] = bytes_to_tensor(dd[8])
    data['merged_ids_main'] = bytes_to_tensor(dd[9])   
    data['sub_plen'] = bytes_to_tensor(dd[10])  
    data['single_rewards'] = bytes_to_tensor(dd[11])  
    data['gen_logps_ori'] = bytes_to_tensor(dd[12])  
    data['sample'] = bytes_to_tensor(dd[13]) 
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
import random

def GRPO_step(batch):
    prompt_length = batch['sub_plen']
    inputs = batch['inputs'].to(engine.device)
    #print(f"!!! rank: {torch.distributed.get_rank()} inputs shape: {inputs.shape} ")
    advantages_all = batch['rewards'].to(engine.device).unsqueeze(1)
    advantages = batch['single_rewards'].to(engine.device).unsqueeze(1)
     #bz, 1
    logits = engine(inputs).logits
    # print(f"!!! rank: {torch.distributed.get_rank()} cal the logits successfully!!")

    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps_sliced= []
    for i in range(per_token_logps.shape[0]):  # batch_size
        plen = prompt_length[i].item()   # 如2或10
        per_token_logps_sliced.append(per_token_logps[i, plen-1:])
    per_token_logps = pad_sequence(per_token_logps_sliced, batch_first=True, padding_value=float('-inf'))

    if use_ref_kl:
        ref_per_token_logps = batch['refs'].to(per_token_logps.device)
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_masks = [
        (inputs[i, prompt_length[i]:] != tokenizer.pad_token_id).float()
        for i in range(inputs.shape[0])
    ]
    completion_mask = pad_sequence(completion_masks, batch_first=True, padding_value=0.0)


    gen_probs_list = [
    F.softmax(logits[i, prompt_length[i]-1:, :], dim=-1)
    for i in range(logits.shape[0])
    ]   

    probs = pad_sequence(gen_probs_list, batch_first=True, padding_value=0.0)
    lengths = [x.shape[0] for x in gen_probs_list]
    max_len = probs.shape[1]

    mask = torch.arange(max_len, device=probs.device)[None, :] < torch.tensor(lengths, device=probs.device)[:, None]
    mask = mask.float()  # shape: (B, L_max)

    gen_logits = [
     logits[i, prompt_length[i]-1:, :] for i in range(logits.shape[0])
    ]
    gen_logits = pad_sequence(gen_logits, batch_first=True, padding_value=0.0)
    log_probs = F.log_softmax(gen_logits, dim=-1)
    # print("completion_masks1.shape",mask.shape)
    # print("completion_mask.shape",completion_mask.shape)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, L_gen)
    entropy = (entropy * completion_mask*mask).mean(-1)   # 应用mask过滤pad部分

    if torch.distributed.get_rank() == 0:
        wandb.log({"avg_entropy": entropy.mean().item()})
    # advantages = advantages - alpha * entropyjiuz

    main_prompt_length = batch['plen']
    inputs_main = batch['merged_ids_main'].to(engine.device)
    logits_main = engine(inputs_main).logits
    # print(f"!!! rank: {torch.distributed.get_rank()} cal the logits successfully!!")

    logits_main = logits_main[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids_main = inputs_main[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps_main = get_per_token_logps(logits_main, input_ids_main)
    per_token_logps_main = per_token_logps_main[:,main_prompt_length-1:]

    if 'gen_logps' in batch:


        is_args = batch['is_aguments'].to(engine.device).to(torch.bool)      
        gen_logps = batch['gen_logps'].to(engine.device)                    # (B, L)
        # gen_logps_ori = batch['gen_logps_ori'].to(engine.device)                    # (B, L)
        # 1. 计算ratio1和per_token_loss1
        prob = torch.exp(per_token_logps_main )  #代表的是初始的prompt和其他prompt答案的结合
        # old_prob = torch.exp(gen_logps_ori)
        ratio1 = prob
        # clipped_ratio1 = torch.clamp(ratio1, 1-clip_param, 1+clip_param)
        alpha = 0.025
        p = 2
        advantages_all=F.leaky_relu(advantages_all, negative_slope=alpha) * p
        # per_token_loss1 = torch.min(ratio1 * advantages_all, clipped_ratio1 * advantages_all)       # (B, L)
        per_token_loss1 = ratio1*advantages_all
        # 2. 计算ratio2和per_token_loss2
        ratio2 = torch.exp(per_token_logps - gen_logps)
        clipped_ratio2 = torch.clamp(ratio2, 1-clip_param, 1+clip_param)
        per_token_loss2 = torch.min(ratio2 * advantages, clipped_ratio2 * advantages)    # (B, L)

        # 3. 逐样本判定is_args是否有大于0的token
        is_args_mask = (is_args > 0)            # (B,) bool

        # 如果你的loss是(B, L)，需要扩展成(B, 1)或(B, L)以便broadcast
        if per_token_loss1.dim() == 2 and is_args_mask.dim() == 1:
            is_args_mask = is_args_mask.unsqueeze(1)      # (B, 1)
            # 不需要expand_as, broadcast会自动匹配(B, 1)到(B, L)

        per_token_loss = torch.where(is_args_mask, per_token_loss1+per_token_loss2 , per_token_loss2)
        final_ratio = torch.where(is_args_mask, ratio1, ratio2)  # (B, L)
        if torch.distributed.get_rank() == 0:
            wandb.log({"avg_ratio1": ratio1.mean().item()})
            wandb.log({"avg_ratio2": ratio2.mean().item()})
            wandb.log({"avg_ratio": final_ratio.mean().item()})
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        print("don;t use gen logps\n\n")
        assert compute_gen_logps is False
    if use_ref_kl:
        per_token_loss = -(per_token_loss - beta * per_token_kl)
        if torch.distributed.get_rank() == 0:
            wandb.log({"ref_kl": beta * per_token_kl.mean().item()})
    else:
        per_token_loss = -per_token_loss
    if token_level_loss:
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    else:
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    # print(f"!!! rank: {torch.distributed.get_rank()} cal the loss successfully!!")
    return loss

import signal
import time
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
def handler(signum, frame):
    raise TimeoutError("Code execution timed out")

def gen_worker(Q, physics_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)
    print(f"!!!Generation worker process uses GPU {physics_device}")
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=model_path, 
                   gpu_memory_utilization=0.45,
                                  )
    ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload

    sampling_params = SamplingParams(n=num_pre_Q, temperature=0.9, max_tokens=sample_max_tokens)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    
    data_path = train_config['data_path']   
    QAs =  dataset_process(data_path,search_results)
    print("QAs", len(QAs))
    system_prompt=[]
    for prompt_path in prompt_paths:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt.append(f.read())       
    def run(input_string):
        start_index = input_string.find('```python')
        end_index = input_string.find('```', start_index + 9)
        # 提取代码
        if start_index != -1 and end_index != -1:
            code = input_string[start_index + 9:end_index].strip()
            output_capture = io.StringIO()
            sys.stdout = output_capture  
            try:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(1)  
                local_vars = {}
                exec(code, {}, local_vars) 
            except TimeoutError as e:
                return f"Error! The Code Execution timeout!"
            except Exception as e:
                # Get detailed error information
                error_msg = traceback.format_exc()
                return f"Error! {type(e).__name__}: {str(e)}"
            finally:
                signal.alarm(0)
                sys.stdout = sys.__stdout__ 
                
            output_result = output_capture.getvalue().strip()
            return output_result if output_result else "Error! No output"
        else:
            return "Error! Python code block not found."

    stop_sentences = "The result of executing this Python code is:"
    #sampling_params_stop = SamplingParams(n=1, temperature=0.9, max_tokens=800, stop=stop_sentences, include_stop_str_in_output=True)
    def get_completions(prompts, num):
        outputs = vllm_gen.generate(prompts, sampling_params, use_tqdm=False)
        responses = [output.outputs[0].text for output in outputs]
        if num > 5:
            return responses
        recursive_ids = []
        for i, c in enumerate(responses):
            # if c.endswith("<<<"):
            if c.endswith(stop_sentences):
                recursive_ids.append(i)
        # 如果需要递归
        if len(recursive_ids)>0:
            recursive_prompts = []
            for i in recursive_ids:
                responses[i] = responses[i]+  run(responses[i])
                recursive_prompts.append(prompts[i] + responses[i])
            rec_resps = get_completions(recursive_prompts, num+1)
            for org_id, rec_resp in zip(recursive_ids, rec_resps):
                responses[org_id] += rec_resp
        return responses
    
    def gen_answers(inputs,step):
        tip_text = []
        
        augments =[]
        for x in inputs:
            for i in range(len(system_prompt)):
                for n in range(num_pre_Q//len(system_prompt) ):
                
                    prompt = system_prompt[i]
                    search = x['P']
                    
                    if i==0:
                        augments.append(0)
                    else:
                        augments.append(1)
                    
                    if '{search_results}' in prompt:
                        dynamic_system_prompt = prompt.format(search_results=search)
                    else:
                        dynamic_system_prompt = prompt
                    tip_text.append(tokenizer.apply_chat_template([
                        {"role": "system", "content": dynamic_system_prompt},
                        {"role": "user", "content": x['Q']}], tokenize=False, add_generation_prompt=True))
        #answers = get_completions(tip_text,0)
        outputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
        answers = [output.outputs[0].text for output in outputs]
        return tip_text,answers,augments

    def reward_format(answer):
        pattern = r"^<think>.*?</think>[\n ]<answer>.*?</answer>$"
        think_count = answer.count("<think>") + answer.count("</think>")
        answer_count = answer.count("<answer>") + answer.count("</answer>")
        reward = 1 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count == 2 and answer_count == 2 else 0
        
        return reward
    def parse_answer(text: str) -> str:
        match = re.search(r'.*<answer>(.*?)</answer>', text, re.DOTALL)
        return match.group(1).strip() if match else ''
    
    def call_python(answer):
        error_cnt = answer.count("Error!")
        python_cnt = answer.count("```python")
        return (python_cnt - error_cnt) * 0.1

    def gen_samples(inputs,step):
        # prompts = [x["Q"] for x in inputs]
        prompts,answers,if_augments = gen_answers(inputs,step)
        
        rewards = []
        scores = []
        record_gen= []
        acc_scores= []
        f1_scores = []
        format_scores =[]
        # if_augments=[]
        for i, inp in enumerate(inputs):
            pre_Q_correct_acc = 0
            pre_Q_correct_format = 0
            
            for idx,a in enumerate(answers[i*num_pre_Q:(i+1)*num_pre_Q]):
                format_score = reward_format(a)
                a =parse_answer(a)
                acc_score = reward_acc( a,inp["A"])
                if dataset_name == 'conflictqa':
                    f1_score=0
                else:
                    f1_score = cau_f1_score( a,inp["A"])
       
                #call_python_score = call_python(a)
                acc_scores.append(acc_score)
                f1_scores.append(f1_score)
                format_scores.append(format_score)
 
                rewards.append(acc_score + format_score+f1_score)
                record_gen.append({"question": inp, "answer": a, "acc_score":acc_score, "format_score": format_score,"f1_score": f1_score})
                
                if acc_score>0: pre_Q_correct_acc += 1
                if format_score>0: pre_Q_correct_format +=1
            scores.append((pre_Q_correct_acc, pre_Q_correct_format))
        
        #record the generation data the score
        if os.path.exists(gen_data_path) and os.path.getsize(gen_data_path) > 0:
            with open(gen_data_path, 'r') as f:
                try:
                    gen_data = json.load(f)
                except json.JSONDecodeError:
                    gen_data = [] 
        else:
            gen_data = []  
        gen_data.extend(record_gen)
        with open(gen_data_path, 'w') as file:
            json.dump(gen_data, file, indent=4)
        prompts_text=[]
        for x in inputs:

            search =x['P']
            # current_date = datetime.now().strftime("%Y-%m-%d")
            dynamic_system_prompt = system_prompt[0].format(
                search_results=search,
                # cur_date=current_date
            )
            prompts_text.append(tokenizer.apply_chat_template([
                    {"role": "system", "content": dynamic_system_prompt},
                    {"role": "user", "content": x['Q']}], tokenize=False, add_generation_prompt=True) )
        
        return prompts_text,prompts, torch.tensor(rewards, dtype=torch.float32), answers, torch.tensor(acc_scores, dtype=torch.float32),torch.tensor(f1_scores, dtype=torch.float32), torch.tensor(format_scores, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32),torch.tensor(if_augments, dtype=torch.float32)

    def try_update_model():
        try:
            
            update_data = Q.get_nowait()  
            current_step = update_data["step"]
            new_state_dict = update_data["weights"]
            
            print(f'[VLLM PROC] receive (step={current_step}) ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print(f'[VLLM PROC] update to step {current_step}')
            
            del new_state_dict  # 释放内存
            return  current_step

        except Exception as e:
            
            return
        
        
    from torch.nn.utils.rnn import pad_sequence

    fout = open(f'{record_path}', 'w')
    for it in range(9999999999):
      
        if it==0:
            start = 0
        else:
            start = 100000
            print("training finished")
            break
        cur_step =0
        for j in range(start,len(QAs), Q_batch_size):
            inputs = QAs[j:j+Q_batch_size]
            print("!!SEND the ",j ,"sample")
            if j % 2 == 0: 
                cur_step1=try_update_model()
            
            cur_step = cur_step1 if cur_step1 is not None else cur_step
            ori_prompt_inputs,prompt_inputs, rewards, answers, acc_scores,f1_scores, format_scores, scores, is_aguments = gen_samples(inputs,cur_step)
            fout.write(str(scores) + '\n')
            if cur_step % 32 == 0:
                print(str(answers[0])+ '\n\n')
            if it % 10000 == 0:
                fout.write(str(prompt_inputs[0])+"\n"+str(answers[0]) + '\n\n')
                fout.flush()
            ans_token_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)['input_ids']
            input_token_ids = tokenizer(prompt_inputs, padding=False, add_special_tokens=False)['input_ids']
            for i, pp in enumerate(ori_prompt_inputs):
                prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
                plen = prompt_ids.shape[1]
                curr_prompt_ids = input_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_plen = [len(ids) for ids in curr_prompt_ids]
                curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]  #提取出单个问题那一组的rewards
                curr_acc_scores = acc_scores[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_f1_scores = f1_scores[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_format_scores = format_scores[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_is_aguments = is_aguments[i*num_pre_Q:(i+1)*num_pre_Q]
                # pdb.set_trace()

                if curr_rewards.max() - curr_rewards.min() < 1e-4:
                    print("abandon batch")
                    continue

                if ref_server_ver == 'tensor':
                    unique_args = curr_is_aguments.unique()
                    normed_rewards = torch.empty_like(curr_rewards)

                    for arg in unique_args:
                        mask = (curr_is_aguments == arg)             # 当前组的布尔索引
                        group_rewards = curr_rewards[mask]           # 取出当前组的reward
                        mean = group_rewards.mean()
                        std = group_rewards.std()
                        # 不同提示组内归一化
                        normed_rewards[mask] = (group_rewards - mean) / (std + 1e-4)
                    curr_rewards = (curr_rewards - curr_rewards.mean())/ (curr_rewards.std() + 1e-4)
                    for ii in range(0, num_pre_Q, train_batch_size):
                        
                        sub_rewards = curr_rewards[ii:ii+train_batch_size]
                        sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                        sub_acc_scores = curr_acc_scores[ii:ii+train_batch_size]
                        sub_f1_scores = curr_f1_scores[ii:ii+train_batch_size]
                        sub_format_scores = curr_format_scores[ii:ii+train_batch_size]
                        sub_is_aguments = curr_is_aguments[ii:ii+train_batch_size]
                        sub_plen = curr_plen[ii:ii+train_batch_size]
                        sub_plen =[torch.tensor(lst) for lst in sub_plen]
                        sub_prompt_ids = curr_prompt_ids[ii:ii+train_batch_size]
                        sub_single_rewards=normed_rewards[ii:ii+train_batch_size]  



                        tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                        Qrep = [torch.tensor(lst) for lst in sub_prompt_ids]
                        merged_list = [torch.cat([q, a]) for q, a in zip(Qrep, tensor_list)]
                        merged_ids = pad_sequence(merged_list, batch_first=True, padding_value=tokenizer.pad_token_id)
                        
                        output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
                        Qrep_ori = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                        merged_ids_ori = torch.cat([Qrep_ori, output_ids], dim=1)
                        
                        data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]              

                        if compute_gen_logps:
                            zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                            zz = [ xx.prompt_logprobs[plen:] if xx.prompt_logprobs is not None else [] for xx,plen in zip(zz,sub_plen)]
                            # zz = [xx.prompt_logprobs[plen:] for xx in zz]
                            if not zz:
                                print("[!!! SPEICIAL CASE]")
                                continue
                            gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                            data.append(tensor_to_bytes(gen_logps))

                        zz_ori = vllm_gen.generate(prompt_token_ids=merged_ids_ori.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                        zz_ori = [xx.prompt_logprobs[plen:] for xx in zz_ori]
                        gen_logps_ori = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz_ori])

                        data.append(tensor_to_bytes(sub_acc_scores))
                        data.append(tensor_to_bytes(sub_format_scores))
                        data.append(tensor_to_bytes(sub_f1_scores))
                        data.append(tensor_to_bytes(sub_is_aguments))
                
                        data.append(tensor_to_bytes(merged_ids_ori))
                        data.append(tensor_to_bytes(sub_plen))
                        data.append(tensor_to_bytes(sub_single_rewards))
                        data.append(tensor_to_bytes(gen_logps_ori))
                        data.append(tensor_to_bytes(torch.tensor(j)))
                        # print("!!data length:", len(data))
                        xdata = make_bytes_list(data)
                        
                        # print("!!start to upload")
                        r = requests.post(f"{ref_server}/upload", data=xdata)
                        if r.content == b'queue_full':
                            time.sleep(1) 
                            print("queue full")
                            continue
                        if r.content == b'string': ref_server_ver = 'string'
                elif ref_server_ver == 'string':
                    xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(), 
                                            tensor_to_bytes(curr_rewards)])
                    r = requests.post(f"{ref_server}/upload", data=xdata)
                    if r.content == b'tensor': ref_server_ver = 'tensor'


tokenizer = AutoTokenizer.from_pretrained(model_path)
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()
    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, gen_device))
        p.start()

    print("!!!!! LOADING MODEL !!!!!")
    model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
  # "sdpa"
    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                                model_parameters=model.parameters())

    progress = range(1, all_steps+1)
    if dist.get_rank() == 0: 
        progress = tqdm(progress)

    total_output_length = 0
    total_acc_correct = 0
    total_acc_correct_0 = 0
    total_acc_correct_1 = 0
    total_format_correct = 0
    total_num = 0
    total_num0 = 1e-6  
    total_num1 = 1e-6  

    if torch.distributed.get_rank() == 0:
        if wandb_offline:
            wandb.init(project=wandb_project, 
                        mode="dryrun",
                        name=wandb_name,
                        config=train_config)
                        
        else:
            wandb.init(project=wandb_project, 
                        name=wandb_name,
                        config=train_config)
    should_break = False
    for step in progress:
        if should_break: 
            break
        print("!!!waiting for batch...")
        batch = get_batch()
        timeout = 600  # 
        start_time = time.time() 
        while batch is None:
            current_time = time.time()
            if current_time - start_time > timeout:
                
                dist.barrier()
                if dist.get_rank() == 0:
                    print('finished training , saving model...')
                    save_name = f"{save_path}/final_model"
                    state_dict = engine.module.state_dict()
                    state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                    engine.module.save_pretrained(save_name, state_dict=state_dict)
                    tokenizer.save_pretrained(save_name)
                dist.barrier()
                should_break = True  
                break

            # print('wait for batch...') 
            time.sleep(1)
            batch = get_batch()

       
        sample = batch['sample'].item()
        if torch.distributed.get_rank() == 0:
            batch_length = batch['gen_logps'].shape[0] * batch['gen_logps'].shape[1]
            total_output_length += batch_length
            
            acc_scores = batch['acc_scores']
            is_aguments = batch['is_aguments']

            total_acc_correct += (acc_scores > 0).sum().item()
            total_format_correct += (batch['format_scores'] > 0).sum().item()

            f1_score = batch['f1_scores'].mean().item()
            rewards = batch['rewards'].mean().item()

            total_num += batch['inputs'].shape[0]
         
            mask0 = (is_aguments == 0)
            mask1 = (is_aguments != 0)
            num0 = mask0.sum().item()
            num1 = mask1.sum().item()
            total_num0 += num0
            total_num1 += num1
            print("get_sample", sample)
           
            acc_correct_0 = ((acc_scores > 0) & mask0).sum().item()
            acc_correct_1 = ((acc_scores > 0) & mask1).sum().item()
            total_acc_correct_0 += acc_correct_0
            total_acc_correct_1 += acc_correct_1
            wandb.log({
                "sample": sample, 
                "is_augment": is_aguments.mean().item(),
                "avg_output_token_length": float(total_output_length) / total_num,
                "acc_correct_ratio": float(total_acc_correct) / total_num,
                "acc_correct_ratio_is_arguments0": float(total_acc_correct_0) / total_num0,
                "acc_correct_ratio_is_arguments1": float(total_acc_correct_1) / total_num1,
                "f1_score": float(f1_score),
                "rewards": float(rewards),
                "format_correct_ratio": float(total_format_correct) / total_num,

            })
        loss = GRPO_step(batch)
        engine.backward(loss)
        # print(f"!!!!rank:{torch.distributed.get_rank()} backward successfully ")
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")
            wandb.log({"loss": loss.item()})
        if step % gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                Q.put({
                    "step": step, 
                    "weights": state_dict
                })  
                print('[TRAINING PROC] send state_dict ok!')
                update_model_num += 1
                print('!!The number of update the genmodel:',update_model_num)
            dist.barrier()

        if step % save_steps == 0  and step > 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"{save_path}/step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()
        if step== all_steps and step > 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"{save_path}/final_model"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()
        
