
import json, os, shutil, re, random, io, time
import torch
from config import train_config
from torch.nn.utils.rnn import pad_sequence
def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()
def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)
def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()
def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

if __name__ == '__main__':   
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn as nn
    from vllm import LLM, SamplingParams
    from bottle import request
    import bottle, threading, queue
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    model_path = train_config['model_path']
    # vllm_gen = vllm_gen = LLM(model=model_path, 
    #                gpu_memory_utilization=0.4,)
    #                enforce_eager=True,          # 对于小模型可以加速
    #             enable_prefix_caching= True,)
    ref_model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.bfloat16, _attn_implementation="fsdp").to('cuda') #"flash_attention_2"
    ref_model.eval()
    ref_model.requires_grad_(False)

    def get_per_token_logps(input_ids):
        logits = ref_model(input_ids).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    raw_queue = queue.LifoQueue()
    result_queue = queue.Queue()

    app = bottle.Bottle()

    # 在Bottle路由中显式设置更大的body限制
    @app.route('/upload', method='POST')
    def do_upload():
        request.body._max_http_body_size = 1024 * 1024 * 500  # 500MB  # 100MB
        dd = request.body.read()  # 现在可读取更大数据
        dd = bytes_list_to_list(dd)
        data = {'base': json.loads(dd[0])} 
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        # if 'gen_logps' in dd: 
        data['gen_logps'] = bytes_to_tensor(dd[3])
        data['acc_scores'] = bytes_to_tensor(dd[4])
        data['format_scores'] = bytes_to_tensor(dd[5])
        data['f1_scores'] = bytes_to_tensor(dd[6])
        data['is_aguments'] = bytes_to_tensor(dd[7])
        data['merged_ids_main'] = bytes_to_tensor(dd[8])   #只包含主prompt
        data['sub_plen'] = bytes_to_tensor(dd[9])  #混合的提示长度
        data['single_rewards'] = bytes_to_tensor(dd[10]) 
        data['gen_logps_ori'] = bytes_to_tensor(dd[11])
        data['sample'] = bytes_to_tensor(dd[12]) 
        # print("[f1_scores]", data['f1_scores'])
        raw_queue.put(data)
        # print('receive', data['inputs'].shape, data['rewards'], 
        #       data['gen_logps'].shape if 'gen_logps' in data else '')
        return b'tensor'

    @app.route('/get', method='GET')
    def do_get():
        if result_queue.empty(): return b'empty'
        return result_queue.get()
    
    def run_server(): bottle.run(app, host='0.0.0.0', port=train_config['port'], server='tornado', workers=4)
    threading.Thread(target=run_server, daemon=False).start()

    while True:
        try:
            d = raw_queue.get(timeout=1)
            prompt_length = d['sub_plen']
            per_token_logps_sliced=[]
            with torch.inference_mode():
                per_token_logps = get_per_token_logps(d['inputs'].to(ref_model.device))
            # per_token_logps = per_token_logps[:,prompt_length-1:]
            for i in range(per_token_logps.shape[0]):  # batch_size
                plen = prompt_length[i].item()   # 如2或10
                per_token_logps_sliced.append(per_token_logps[i, plen-1:])
            per_token_logps = pad_sequence(per_token_logps_sliced, batch_first=True, padding_value=float('-inf'))

            # zz = vllm_gen.generate(prompt_token_ids=d['inputs'].tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
            # zz = [ xx.prompt_logprobs[ prompt_length:] if xx.prompt_logprobs is not None else [] for xx in zz]
            # per_token_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
            # if d['inputs'].shape[1] > 2200:
            #     continue
            data = [json.dumps(d['base']).encode(), tensor_to_bytes(d['inputs']), 
                    tensor_to_bytes(d['rewards']), tensor_to_bytes(per_token_logps)]
            if 'gen_logps' in d: data.append(tensor_to_bytes(d['gen_logps']))
            if 'acc_scores' in d: data.append(tensor_to_bytes(d['acc_scores']))
            if 'format_scores' in d: data.append(tensor_to_bytes(d['format_scores']))
            if 'f1_scores' in d: data.append(tensor_to_bytes(d['f1_scores']))#;print("f1_scores", d['f1_scores'])
            if 'is_aguments' in d: data.append(tensor_to_bytes(d['is_aguments']))
            if 'merged_ids_main' in d: data.append(tensor_to_bytes(d['merged_ids_main']))
            if 'sub_plen' in d: data.append(tensor_to_bytes(d['sub_plen']))
            if 'single_rewards' in d: data.append(tensor_to_bytes(d['single_rewards']))
            if 'gen_logps_ori' in d: data.append(tensor_to_bytes(d['gen_logps_ori']))
            if 'sample' in d: data.append(tensor_to_bytes(d['sample']))
            xdata = make_bytes_list(data)
            result_queue.put(xdata)
        except queue.Empty:
            continue

    
