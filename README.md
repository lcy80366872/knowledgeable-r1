<div align="center">

# Knowledgeable-r1: Policy Optimization for Knowledge Exploration in Retrieval-Augmented Generation
[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2504.13055) 

</div>

## ⚡ Updates
* 05/06/2025: 🎉 We release our paper and codebase.

## 🚀 Introduction
<p align="center">
  <img src="./images/introduction.png" width=100%/>
</p>

**Knowledgeable-r1** is an effective strategy for the RL training of LLMs that using joint sampling and define multi policy distributions in knowledge capability exploration to stimulate large language models’ self-integrated utilization of parametric and contextual knowledge. Experiments show that Knowledgeable-r1 significantly enhances robustness and reasoning accuracy in both parameters and contextual conflict tasks and general RAG tasks, especially outperforming baselines in counterfactual scenarios and demonstrating consistent gains across RAG tasks.

🎯 **Key Benefits**:
- **No additional cost** — only the rollout strategy and RL objective is modified 
- **Easy to adopt** — no additional components or complex multiple prompt pipelines are required in application  
- **Superior generalization** — Knowledgeable-r1 significantly enhances robustness and reasoning accuracy in both parameters and contextual conflict tasks and general RAG tasks


## 🙌 Environment
The runtime environment is in the requirements.txt
so you can
``` bash
pip install -r requirements.txt
```
At least two GPUs are needed.

## Usage
Run the following command:
``` bash
CUDA_VISIBLE_DEVICES=7 python ref_server.py
```
This just uses one GPU to collect and run the reference model.

In *config.py*, set the generation device index ​relative to the visible devices​ in next step:
``` bash
gen_device = 0
```
Then, open another bash:
``` bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 deepspeed grpo_program.py
```

## Citation
If you find our works useful for your research, please consider citing:
```bibtex
@misc{,
      title={Knowledgeable-r1: Policy Optimization for Knowledge Exploration in Retrieval-Augmented Generation}, 
      author={},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={ }, 
}
```

