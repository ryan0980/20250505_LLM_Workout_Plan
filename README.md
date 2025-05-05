# Qwen2.5-3B Model Compression Study

This repository contains the implementation and evaluation of various compression techniques for the Qwen2.5-3B language model. The study explores different combinations of quantization, pruning, and fine-tuning methods to achieve efficient model deployment while maintaining performance.

## Abstract

Large language models (LLMs) have achieved remarkable performance across diverse domains. However, their large size poses challenges for deployment, especially in resource-constrained settings. In this work, we study systematic compression of the Qwen2.5-3B model via multiple techniques. Specifically, we leverage AutoAWQ for 4-bit quantization, ShortGPT for model pruning, and Low-Rank Adaptation (LoRA) for fine-tuning. We examine 16 different compression pipelines arising from combinations of these techniques. We evaluate each pipeline on domain-specific benchmarks (LawBench, MMLU-STEM, MMLU-Law) to study accuracy vs. compression trade-offs.

## Project Structure

- `awq/`: Implementation of 4-bit quantization using AutoAWQ
- `prune_short_Gpt/`: Model pruning using ShortGPT
- `llamafactory/`: LoRA fine-tuning implementation
- `opencompass/`: Evaluation framework for model performance
- `dataset/`: Domain-specific datasets for evaluation
- `model/`: Compressed model checkpoints
- `GPU_Use_Analysis/`: Analysis of GPU resource utilization
- `gptq/`: Additional quantization experiments

## Key Findings

1. Combining quantization with LoRA fine-tuning achieves optimal balance between compression and accuracy
2. Aggressive pruning can negatively impact specialized task performance
3. Different compression techniques show varying effectiveness across domains

## Installation

```bash
# Clone the repository
git clone https://github.com/ryan0980/20250505_LLM_Workout_Plan.git
cd 20250505_LLM_Workout_Plan

# Install dependencies
pip install -r requirements.txt
```

## Models

The compressed models are available on Hugging Face Hub under the [tusrau](https://huggingface.co/tusrau) organization. The following models are available:

- `tusrau/q3bft_q_p`: Quantized, fine-tuned, and pruned model
- `tusrau/q3b_q_ft_p`: Quantized, fine-tuned, and pruned model (different order)
- `tusrau/q3b_p_ft_q`: Pruned, fine-tuned, and quantized model
- `tusrau/q3b_ft_p_q`: Fine-tuned, pruned, and quantized model
- `tusrau/q3bp_q`: Pruned and quantized model
- `tusrau/q3bft_q`: Fine-tuned and quantized model
- `tusrau/q3bft`: Fine-tuned model
- `tusrau/q3bq`: Quantized model
- `tusrau/q3b_p_ft`: Pruned and fine-tuned model
- `tusrau/q3bp`: Pruned model

## Usage

### Loading Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "tusrau/q3bft_q_p"  # or any other model from the list above
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### Quantization

```bash
cd awq
python quantize.py --model_path [path_to_model] --output_dir [output_path]
```

### Pruning

```bash
cd prune_short_Gpt
python prune.py --model_path [path_to_model] --sparsity [sparsity_ratio]
```

### Fine-tuning with LoRA

```bash
cd llamafactory
python train.py --model_path [path_to_model] --dataset [dataset_path]
```

### Evaluation

```bash
cd opencompass
python run_eval.py --model_path [path_to_model] --benchmark [benchmark_name]
```

## Results

The study evaluates 16 different compression pipelines on:

- LawBench
- MMLU-STEM
- MMLU-Law

Detailed results and analysis can be found in the paper.

## Citation

If you use this code in your research, please cite our paper:

```
[Citation information to be added]
```

## License

[License information to be added]
