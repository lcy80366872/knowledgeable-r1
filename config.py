train_config = {
    "all_steps": 1600,
    "save_steps": 100,
    "wandb_name":"ConFiQA-QA",
    "wandb_project":"knowledgeable-r1",
    "save_path":  "save_model",
    "record_path": "record.json",   #"record the generated data during the training process"
    "gen_data_path":"gen_data.json", # "record all the generated data during the training process"
    "gen_device":0,  
    "data_path":"data/conflict_qa/ConFiQA-QA-train.json",   #"data/hotpot/hotpot_train_v1.1.json" data/MuSiQue/musique_ans_v1.0_train.jsonl
    "dataset_name":'conflictqa',    # 'musique' or 'hotpotqa' or 'conflictqa_mix' or '2wiki'
    "beta": 0.04,
    "token_level_loss": True,
    "model_path": "model/Qwen2.5-3B-Instruct" ,
    "Q_batch_size": 1,   # number of questiozns in generate data vllm
    "num_pre_Q": 16,   # number of per question generating samples 
    "train_batch_size":1,
    "random_remove": 10,
    "gen_update_steps": 16,
    "sample_max_tokens":512,
    "compute_gen_logps": True,
    "noise_ratio": 0.2,
    "use_ref_kl": True,
    "search_results": 20,
    "prompt_path":["system_prompt/system_prompt_0516_external.txt","system_prompt/system_prompt_0508_internal.txt"],    
    "clip_param": 0.2,
    "ref_server": "http://localhost:59807",
    "port": 59807,
    "wandb_key":"PUT YOUR KEYS",
    "wandb_offline": True ,
}

ds_config = {
    "train_micro_batch_size_per_gpu": train_config['train_batch_size'],
    "gradient_accumulation_steps": 8,
    "steps_per_print": 5,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "distributed": {
        "timeout": 12000  # 单位：秒 
    },
    "logging": {
        "gradient_norm": True,
        "gradient_values": True
        },

    "bf16": {"enabled": True},
    "activation_checkpointing": {
         "partition_activations": True,
    #     "cpu_checkpointing": True,  
         "contiguous_memory_optimization": True
     },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
        
        
    }
}
