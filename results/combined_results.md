# Results
## Result for different operation sequence

| Operation Sequence                                | LawBench | stem | mmlu_law | LawBench Change (%) | stem Change (%) | mmlu_law Change (%) | Model Size |
| ------------------------------------------------- | -------- | ---- | -------- | ------------------- | --------------- | ------------------- | ---------- |
| Original Model                                    | 7.85     | 3.13 | 4.76     | 0.00%               | 0.00%           | 0.00%               | 5.8G       |
| Fine-tuning                                       | 15.53    | 3.69 | 3.78     | +97.83%             | +17.89%         | –20.59%             | 5.8G       |
| Pruning                                           | 5.77     | 2.89 | 3.06     | –26.49%             | –7.67%          | –35.71%             | 2.6G       |
| Quantization                                      | 4.44     | 3.62 | 2.30     | –43.44%             | +15.65%         | –51.68%             | 4.8G       |
| Pruning → Fine-tuning                              | 8.05     | 5.26 | 3.17     | +2.55%              | +68.05%         | –33.40%             | 5.8G       |
| Pruning → Quantization                             | 1.13     | 2.98 | 3.46     | –85.60%             | –4.79%          | –27.31%             | 1.7G       |
| Quantization → Fine-tuning                         | 12.94    | 3.70 | 3.52     | +64.78%             | +18.21%         | –26.05%             | 5.8G       |
| Quantization → Pruning                             | 3.30     | 3.42 | 2.16     | –57.95%             | +9.27%          | –54.62%             | 1.7G       |
| Fine-tuning → Pruning                              | 1.60     | 2.51 | 2.48     | –79.62%             | –19.81%         | –47.90%             | 4.8G       |
| Fine-tuning → Quantization                         | 6.70     | 4.57 | 5.29     | –14.65%             | +46.00%         | +11.13%             | 2.0G       |
| Fine-tuning → Pruning → Quantization                | 1.79     | 2.72 | 3.69     | –77.20%             | –13.13%         | –22.48%             | 1.7G       |
| Fine-tuning → Quantization → Pruning                | 0.11     | 2.65 | 4.02     | –98.60%             | –15.34%         | –15.55%             | 1.7G       |
| Pruning → Fine-tuning → Quantization                | 6.74     | 5.49 | 3.52     | –14.01%             | +75.40%         | –26.05%             | 2.0G       |
| Quantization → Fine-tuning → Pruning                | 5.08     | 2.42 | 4.29     | –35.32%             | –22.68%         | –9.87%              | 4.8G       |
| Quantization → Pruning → Fine-tuning                | Not Applicable |      |          |                     |                 |                     |            |
| Pruning → Quantization → Fine-tuning                | Not Applicable |      |          |                     |                 |                     |            |

![result for operation sequence](/results/figs/op_sq.png)

---
## Comparison on different method
<!-- |比较|LawBench 变化率%|stem 变化率%|mmlu_law 变化率%|
|---|---|---|---|
|原始模型 → 微调|+97.83%|+17.89%|-20.59%|
|剪枝 → 剪枝微调|+39.52%|+82.01%|+3.59%|
|量化 → 量化微调|+191.44%|+2.21%|+53.04%| -->

| Comparison                                 | LawBench Change (%) | stem Change (%) | mmlu_law Change (%) |
| ------------------------------------------ | ------------------- | --------------- | ------------------- |
| Original Model → Fine-tuning               | +97.83%             | +17.89%         | –20.59%             |
| Pruning → Pruning Fine-tuning              | +39.52%             | +82.01%         | +3.59%              |
| Quantization → Quantization Fine-tuning    | +191.44%            | +2.21%          | +53.04%             |

![result for Method Comparison](/results/figs/me_cp.png)

## Results for Pruning

<!-- |比较|LawBench 变化|LawBench 变化率%|stem 变化|stem 变
化率%|mmlu_law 变化|mmlu_law 变化率%|
|---|---|---|---|---|---|---|
|原始模型 → 剪枝|-2.08|-26.49%|-0.24|-7.67%|-1.70|-35.71%|
|微调 → 微调剪枝|-13.93|-89.68%|-1.18|-19.81%|-1.30|-34.39%|
|量化 → 量化剪枝|-1.14|-25.68%|-0.20|-5.52%|-0.14|-6.09%|
|量化微调 → 量化微调剪枝|-7.86|-60.72%|-1.28|-34.59%|+0.77|+21.88%|
|微调量化 → 微调量化剪枝|-6.59|-98.36%|-1.92|-42.01%|-1.27|-24.00%| -->
| Comparison                               | LawBench Change | LawBench Change (%) | stem Change | stem Change (%) | mmlu_law Change | mmlu_law Change (%) |
|------------------------------------------|-----------------|---------------------|-------------|-----------------|-----------------|---------------------|
| Original Model → Pruning                 | -2.08           | –26.49%             | -0.24       | –7.67%          | -1.70           | –35.71%             |
| Fine-tuning → Fine-tuning Pruning        | -13.93          | –89.68%             | -1.18       | –19.81%         | -1.30           | –34.39%             |
| Quantization → Quantization Pruning      | -1.14           | –25.68%             | -0.20       | –5.52%          | -0.14           | –6.09%              |
| Quantization → Fine-tuning → Pruning     | -7.86           | –60.72%             | -1.28       | –34.59%         | +0.77           | +21.88%             |
| Fine-tuning → Quantization → Pruning     | -6.59           | –98.36%             | -1.92       | –42.01%         | -1.27           | –24.00%             |

![result for pruning](/results/figs/res_pr.png)
## Results for Quantization

<!-- | 比较            | LawBench 变化 | LawBench 变化率% | 
stem 变化 | stem 变化率% | mmlu_law 变化 | mmlu_law 变化率% |
| ------------- | ----------- | ------------- | ------- | --------- | ----------- | ------------- |
| 原始模型 → 量化     | -3.41       | -43.44%       | +0.49   | +15.65%   | -2.46       | -51.68%       |
| 微调 → 微调量化     | -8.83       | -56.86%       | +0.88   | +23.84%   | +1.51       | +39.95%       |
| 剪枝 → 剪枝量化     | -4.64       | -80.42%       | +0.09   | +3.11%    | +0.40       | +13.07%       |
| 微调剪枝 → 微调剪枝量化 | +0.19       | +11.88%       | +0.21   | +8.37%    | +1.21       | +48.79%       |
| 剪枝微调 → 剪枝微调量化 | -1.31       | -16.27%       | +0.23   | +4.37%    | +0.35       | +11.04%       | -->

| Comparison                                | LawBench Change | LawBench Change (%) | stem Change | stem Change (%) | mmlu_law Change | mmlu_law Change (%) |
|-------------------------------------------|-----------------|---------------------|-------------|-----------------|-----------------|---------------------|
| Original Model → Quantization             | -3.41           | –43.44%             | +0.49       | +15.65%         | -2.46           | –51.68%             |
| Fine-tuning → Quantization                | -8.83           | –56.86%             | +0.88       | +23.84%         | +1.51           | +39.95%             |
| Pruning → Quantization                    | -4.64           | –80.42%             | +0.09       | +3.11%          | +0.40           | +13.07%             |
| Fine-tuning → Pruning → Quantization      | +0.19           | +11.88%             | +0.21       | +8.37%          | +1.21           | +48.79%             |
| Pruning → Fine-tuning → Quantization      | -1.31           | –16.27%             | +0.23       | +4.37%          | +0.35           | +11.04%             |

![result for quantization](/results/figs/res_qt.png)


