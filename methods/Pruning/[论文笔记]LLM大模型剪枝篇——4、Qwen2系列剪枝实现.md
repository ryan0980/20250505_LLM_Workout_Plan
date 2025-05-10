---
aliases: 
tags: 
date created: 20250417 02:33
date updated: 20250505 02:25
---

# [论文笔记]LLM 大模型剪枝篇——4、Qwen2 系列剪枝实现

工作：

把 shortgpt 的 llama 代码改成了 Qwen 的剪枝。
具体方法：

用 wikitext 数据，计算每层的影响力分数即 BI 分数 (1- 层前后隐层状态余弦相似度)，剪掉影响力低的 P% 的层数。

剪枝脚本：bash short\_qwen\_xin.sh
推理脚本：bash short\_qwen\_xin\_test.sh

```java
bash short_qwen_xin.sh
```

## short\_qwen\_xin.sh

```java
# 模型路径
model_names="/cpfs/074bqrkckm2dg5dq9nc/shared/AI-QIHUAN/OpenModels/Qwen2/Qwen2-0.5B-Instruct /cpfs/074bqrkckm2dg5dq9nc/shared/AI-QIHUAN/OpenModels/Qwen2/Qwen2-1.5B-Instruct"
# model_names="/cpfs/074bqrkckm2dg5dq9nc/shared/AI-QIHUAN/OpenModels/Qwen2.5/Qwen2.5-0.5B-Instruct"
# 遍历模型
# 遍历剪枝层数从1到20
for model_name in ${model_names};do
echo "[^.^] 模型名称为:${model_name}"
for prune_layers in {1..1};do
    echo "[^.^] 正在剪枝，剪枝层数=${prune_layers}层的模型..."
    python short_qwen_xin.py --model_name "$model_name" --prune_layers $prune_layers >>log
done
done
```

## short\_qwen\_xin.py

```java
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from short_qwen import ShortQwen  # ShortQwen class for model pruning
import os
import json
import shutil
import argparse
import warnings
warnings.simplefilter('ignore')

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description='剪枝模型层数')
parser.add_argument('--model_name', type=str, default="/cpfs/074bqrkckm2dg5dq9nc/shared/AI-QIHUAN/OpenModels/Qwen2/Qwen2-0.5B-Instruct", required=True, help='模型的路径')
parser.add_argument('--prune_layers', type=int, default=4, required=True, help='要剪枝的层数')
# 解析命令行参数
args = parser.parse_args()

#🌹 Step1: Load the dataset
# data = load_dataset("pg19", split="validation")  #下载太慢了，弃用
# data = load_dataset("wikitext", "wikitext-103-v1", split="validation") #英语
data = load_dataset("openai/MMMLU", "default", split="test") #多语言
def concatenate_fields(batch):
    return f"Question: {batch['Question'][0]} A: {batch['A'][0]} B: {batch['B'][0]} C: {batch['C'][0]} D: {batch['D'][0]} Answer: {batch['Answer'][0]}"
data_split = data.map(lambda x: {"text": concatenate_fields(x)})

dataloader = DataLoader(
    data_split,
    batch_size=1,
    shuffle=True,
    generator=torch.Generator(device="cpu")
)
print("前5个批次的数据展示：")
for i, batch in enumerate(dataloader):
    if i >= 5:
        break
    print(f"Batch {i+1}:")
    print(batch)

MAX_SEQ_LEN = 1024  # Set context width to 1024 for Qwen

# 🌟 Step2: Choose the model size (Qwen2-1.5B or Qwen2-0.5B)
model_name = args.model_name
# model_name = "/cpfs/074bqrkckm2dg5dq9nc/shared/AI-QIHUAN/OpenModels/Qwen2/Qwen2-0.5B-Instruct"  # Replace with "Qwen/Qwen2-0.5B" as needed, # /cpfs/074bqrkckm2dg5dq9nc/shared/AI-QIHUAN/OpenModels/Qwen2/Qwen2-1.5B
qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
qwen_model = AutoModelForCausalLM.from_pretrained(model_name)

# 🌟 Step3: ShortQwen
# 🌹 类实例化: Create ShortQwen instance and specify the number of layers to prune
# prune_layers=4 #20%: 4/24
prune_layers = args.prune_layers
short_qwen = ShortQwen(model_name=model_name, n_prune_layers=prune_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # kexin debug cuda
short_qwen.model.model.to(device) # kexin debug cuda
# 🌹 Print model layers before pruning # print(short_qwen.model.model.transformer.h) #llama
print(short_qwen.model.model) #qwen
short_qwen.model.generate(
    input_ids=qwen_tokenizer("I am an avid fan of ", return_tensors="pt").input_ids.cuda(),
    max_length=20,
    use_cache=False  # 禁用缓存，避免因层数不匹配产生错误
)
# 🌹 Run the evaluation loop for pruning importance
for batch in tqdm(dataloader, desc="Processing batches"):
    prompts = batch['text']

    # Tokenize the prompts / 开始符:llama:bos_token=True,eos_token=False Qwen:无
    prompt_tokens = [qwen_tokenizer.encode(x, return_tensors="pt").squeeze(0).cuda() for x in prompts]
    max_prompt_len = max(len(t) for t in prompt_tokens)

    # Sliding window of size 1024 with a shift of 256
    for start in range(0, max_prompt_len, 256):
        inputs = [p[start:start+MAX_SEQ_LEN] for p in prompt_tokens if len(p) > start]

        short_qwen.eval_importance(
            prompt_tokens=inputs,
            max_gen_len=0
        )

# 🌹 剪枝：Print the layer importance scores and remove layers accordingly
print("[^V^] importances: ", short_qwen.importances)
print("[O.O] remove layers: ", short_qwen.remove_layers())

# 🌹 Check the model layers after pruning
print(short_qwen.model.model.layers)
print(f"Model layers after pruning: {len(short_qwen.model.model.layers)}")

# ================================================ save pruned Qwen ========================================================
def save_pruned_qwen_model(short_qwen, model_name, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    torch.save(short_qwen.model.state_dict(), f'{save_directory}/pytorch_model.bin')
    # 原始模型的路径
    original_model_dir = model_name
    # 配置文件路径
    config_path = os.path.join(original_model_dir, 'config.json')
    generation_config_path = os.path.join(original_model_dir, 'generation_config.json')
    merges_path = os.path.join(original_model_dir, 'merges.txt')
    tokenizer_json_path = os.path.join(original_model_dir, 'tokenizer.json')
    tokenizer_config_path = os.path.join(original_model_dir, 'tokenizer_config.json')
    vocab_path = os.path.join(original_model_dir, 'vocab.json')
    # 复制配置文件
    print(config_path, save_directory) #kexin debug
    shutil.copy(config_path, os.path.join(save_directory, 'config.json'))
    shutil.copy(generation_config_path, os.path.join(save_directory, 'generation_config.json'))
    shutil.copy(merges_path, os.path.join(save_directory, 'merges.txt'))
    shutil.copy(tokenizer_json_path, os.path.join(save_directory, 'tokenizer.json'))
    shutil.copy(tokenizer_config_path, os.path.join(save_directory, 'tokenizer_config.json'))
    shutil.copy(vocab_path, os.path.join(save_directory, 'vocab.json'))
    # 修改 config.json 中的层数, 更新配置中的层数
    with open(os.path.join(save_directory, 'config.json'), 'r') as f:
        config = json.load(f)
    config['num_hidden_layers'] = len(short_qwen.model.model.layers)
    with open(os.path.join(save_directory, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)
    print(f"The pruned model and tokenizer have been saved to: {save_directory}")

# 🌟 Step4: Save the pruned Qwen model
print("[^V^]Save the pruned Qwen model!")
# "-"+str(len(qwen_model.model.model.layers))+"to"+str(len(short_qwen.model.model.layers))+"-"
save_directory="/cpfs/074bqrkckm2dg5dq9nc/shared/AI-QIHUAN/MarcoMini/code/git_code/ShortGPT/short_gpt/models/"+model_name.split("/")[-1]+"-pruned"+"-"+str(qwen_model.config.num_hidden_layers)+"to"+str(len(short_qwen.model.model.layers))+"-"+str(prune_layers)+"prune_layers"
save_pruned_qwen_model(short_qwen, model_name, save_directory)

# ================================================ load pruned Qwen & generate ========================================================
def load_pruned_qwen_model(model_name, save_directory):
    qwen_model = AutoModelForCausalLM.from_pretrained(save_directory)
    print("Config layers:", qwen_model.config.num_hidden_layers)
    qwen_model.config.num_hidden_layers = len(short_qwen.model.model.layers)  # 修改层数为剪枝后的层数
    print("Config layers:", qwen_model.config.num_hidden_layers)
    qwen_model.config.use_cache = False
    print(f"The pruned model has been loaded from {save_directory}")
    return qwen_model

# Load the pruned Qwen model
print("[^V^]loading the pruned Qwen model!")
pruned_qwen = load_pruned_qwen_model(model_name, save_directory)

# 🌟 Step5: Sample text completion after pruning
generated = short_qwen.model.generate(
    # input_ids=qwen_tokenizer("I am an avid fan of ", return_tensors="pt").input_ids.cuda(), # kexin debug cuda
    input_ids=qwen_tokenizer("请你翻译成英文: 香港代购SK2神仙水限量版", return_tensors="pt").input_ids.cuda(), # kexin debug cuda
    max_length=20,
    use_cache=False  # 禁用缓存，避免因层数不匹配产生错误
)
print("Generated text:", qwen_tokenizer.decode(generated[0], skip_special_tokens=True))
```

## short\_qwen.py

```java
from typing import List, Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # Import from transformers

from metrics import *  # Assuming metrics is a custom module

def sample_top_p(probs: torch.Tensor, p: float):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        return_hiddens: Optional[bool] = False
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.
            (Optional) return_hiddens (bool): Whether to return hidden states. Defaults to False.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.
            (Optional) List[torch.Tensor]: Hidden states for each transformer block.
        """
        outputs = self.model(input_ids=tokens, output_hidden_states=return_hiddens)
        logits = outputs.logits
        hiddens = outputs.hidden_states if return_hiddens else None
        return logits, hiddens if return_hiddens else logits

class ShortQwen:
    def __init__(self, model_name: str, n_prune_layers: Optional[int] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # self.model = TransformerWrapper(self.model)  # wrap transformer to collect hidden states
        self.n_prune_layers = n_prune_layers

        # Attempt to access the correct attribute depending on model structure
        # Adjust the layers based on the model's internal architecture
        # print(dir(self.model)) # 如果模型是self.model.model结构，打印其属性
        # print(dir(self.model.model)) # 如果模型是self.model.model结构，打印其属性
        try:
            # 根据层数初始化importances
            self.importances = [0 for _ in range(self.model.config.num_hidden_layers)]
            # self.importances = [0 for _ in range(len(self.model.model.h))]  # llama layer-wise importance scores
        except AttributeError:
            print("Model does not have 'h / layers' attribute, please verify the model structure.")

    def remove_layers(
        self,
        layers_to_remove: Optional[List[int]] = [],
        angular: Optional[bool] = False
    ):
        if angular:
            assert self.importances, "Need to compute importances with eval_importance()"
            assert self.n_prune_layers, "Need number of layers to prune, set `n_prune_layers`"
            start_layer = np.argsort(np.array(self.importances[:-self.n_prune_layers+1]))[0]
            layers_to_remove = list(range(start_layer, start_layer + self.n_prune_layers))
        elif not layers_to_remove and self.n_prune_layers:
            assert self.importances, "Need to compute importances with eval_importance()"
            layers_to_remove = np.argsort(np.array(self.importances))[:self.n_prune_layers].tolist()

        # remove layers in reverse to avoid indexing errors
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                # kexin debug
                del self.model.model.layers[layer_idx] #transformer
                # del self.model.model.h[layer_idx] #llama
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
                return []
        
        return layers_to_remove
    
    def compute_bi(self, hiddens: List[torch.Tensor], angular: bool):
        n = 1
        if angular:
            assert self.n_prune_layers is not None, "Set number of layers to prune to use angular importance"
            n = self.n_prune_layers

        for i in range(len(hiddens) - n):
            in_hidden = hiddens[i]
            out_hidden = hiddens[i+n]
            if angular:
                # use only last token for angular distance as described in section 3.2
                in_hidden = in_hidden[:,-1:]
                out_hidden = out_hidden[:,-1:]
            
            self.importances[i] += block_influence(
                in_hidden,
                out_hidden,
                angular=angular
            ).sum().cpu().item()

    @torch.inference_mode()
    def eval_importance(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: Optional[int] = 0,
        temperature: Optional[float] = 0.6,
        top_p: Optional[float] = 0.9,
        angular: Optional[bool] = False
    ):
        """
        Computes layer-wise importances over input tokens.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            (Optional) max_gen_len (int): Maximum length of the generated text sequence.
            (Optional) temperature (float): Temperature value for controlling randomness in sampling.
            (Optional) top_p (float): Top-p probability threshold for nucleus sampling.
            (Optional) angular (bool): Whether to use angular distance.

        Returns:
            None
        """
        bsz = len(prompt_tokens)
        # assert bsz <= self.model.model.config.max_batch_size, (bsz, self.model.model.config.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.model.model.config.max_position_embeddings
        total_len = min(self.model.model.config.max_position_embeddings, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_token_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        
        for cur_pos in range(min_prompt_len, total_len):
            logits, _ = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_token_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break
        
        # Compute block influence over full sequences
        outputs = self.model(tokens, output_hidden_states=True)
        hiddens = outputs.hidden_states #tuple
        #for i, hidden in enumerate(hiddens):
        #    print(f"Layer {i} hidden state shape: {hidden.shape}")
        # _, hiddens = self.model.forward(tokens, 0, return_hiddens=True)
        self.compute_bi(hiddens, angular=angular)
        return
```

## short\_qwen\_xin\_test.sh

```java
# 模型路径
save_directory_list=$(ls -d /cpfs/074bqrkckm2dg5dq9nc/shared/AI-QIHUAN/MarcoMini/code/git_code/ShortGPT/short_gpt/models/*/)
# 遍历模型
# 遍历剪枝层数从1到20
for save_directory in ${save_directory_list}; do
    if [[ "$save_directory" == *"0.5B"* ]]; then
        model_name="/cpfs/074bqrkckm2dg5dq9nc/shared/AI-QIHUAN/OpenModels/Qwen2/Qwen2-0.5B-Instruct"
    elif [[ "$save_directory" == *"1.5B"* ]]; then
        model_name="/cpfs/074bqrkckm2dg5dq9nc/shared/AI-QIHUAN/OpenModels/Qwen2/Qwen2-1.5B-Instruct"
    fi
    python short_qwen_xin_test.py --model_name "$model_name" --save_directory "$save_directory" >> test_log
done
```

## short\_qwen\_xin\_test.py

```java
from tqdm import tqdm
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from short_qwen import ShortQwen  # ShortQwen class for model pruning
import os
import json
import shutil
import argparse
import warnings
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='剪枝模型层数')
parser.add_argument('--model_name', type=str, default="/cpfs/074bqrkckm2dg5dq9nc/shared/AI-QIHUAN/OpenModels/Qwen2/Qwen2-0.5B-Instruct", required=True, help='模型的路径')
parser.add_argument('--prune_layers', type=int, default=4, required=True, help='要剪枝的层数')
parser.add_argument('--save_directory', type=str, default="/cpfs/074bqrkckm2dg5dq9nc/shared/AI-QIHUAN/MarcoMini/code/git_code/ShortGPT/short_gpt/models/Qwen2-0.5B-Instruct-pruned-24to23-1prune_layers", required=True, help='保存模型')
args = parser.parse_args()

MAX_SEQ_LEN = 1024  # Set context width to 1024 for Qwen
model_name = args.model_name
save_directory = args.save_directory
# ================================================ load pruned Qwen & generate ========================================================
def load_pruned_qwen_model(model_name, save_directory):
    qwen_model = AutoModelForCausalLM.from_pretrained(save_directory)
    print("Config layers:", qwen_model.config.num_hidden_layers)
    qwen_model.config.num_hidden_layers = len(short_qwen.model.model.layers)  # 修改层数为剪枝后的层数
    print("Config layers:", qwen_model.config.num_hidden_layers)
    qwen_model.config.use_cache = False
    print(f"The pruned model has been loaded from {save_directory}")
    return qwen_model

# Load the pruned Qwen model
print("[^V^]loading the pruned Qwen model!")
pruned_qwen = load_pruned_qwen_model(model_name, save_directory)

# 🌟 Step5: Sample text completion after pruning
generated = short_qwen.model.generate(
    # input_ids=qwen_tokenizer("I am an avid fan of ", return_tensors="pt").input_ids.cuda(), # kexin debug cuda
    input_ids=qwen_tokenizer("请你翻译成英文: 香港代购SK2神仙水限量版", return_tensors="pt").input_ids.cuda(), # kexin debug cuda
    max_length=20,
    use_cache=False  # 禁用缓存，避免因层数不匹配产生错误
)
print("Generated text:", qwen_tokenizer.decode(generated[0], skip_special_tokens=True))
```

## llama\_removal.py

```java
from collections import OrderedDict

import torch.nn as nn


def layer_removal(
    model: nn.Module,
    layers_to_remove: OrderedDict
):
    """
    Generic removal implementation
    """

    for layer_name, layer_idx in layers_to_remove.items():
        modules = layer_name.split(".")
        mod = model
        for m in modules[:-1]:
            mod = getattr(mod, m)
        
        if layer_idx is None:
            del getattr(mod, modules[-1])
        else:
            del getattr(mod, modules[-1])[layer_idx]
```

## metrics.py

```java
import torch


def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular=False,
):
    """
    input_hidden_state: B, S, D
    output_hidden_state: B, S, D
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    return 1 - sim
```
