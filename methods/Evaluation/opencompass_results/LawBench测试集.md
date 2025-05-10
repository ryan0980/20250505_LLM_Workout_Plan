---
aliases: 
tags: 
date created: 20250215 10:15
date updated: 20250426 11:17
---

# LawBench 测试集

| 方法组合          | LawBench 分数 |
| ------------- | ----------- |
| 原始（3B）        | 7.85        |
| 微调            | 15.53       |
| 量化            | 4.44        |
| 剪枝            | 5.77        |
| 微调 → 剪枝       | 1.60        |
| 微调 → 量化       | 6.70        |
| 量化 → 剪枝       | 3.30        |
| 量化 → 微调       | 12.94       |
| 剪枝 → 量化       | 1.13        |
| 剪枝 → 微调       | 8.05        |
| 微调 → 量化 → 剪枝  | 0.11        |
| 微调 → 剪枝 → 量化  | 1.79        |
| 量化 → 微调 → 剪枝  | 5.08        |
| 剪枝 → 微调 → 量化  | 6.74        |
| 量化 → 剪枝 → 微调¹ | 不适用         |
| 剪枝 → 量化 → 微调¹ | 不适用         |

```bash

python run.py --hf-type chat \
    --hf-path /home/sagemaker-user/DQFP/model/qwen_1_5b_I \
    --tokenizer-path /home/sagemaker-user/DQFP/model/qwen_1_5b_I \
    --datasets lawbench_zero_shot_gen_002588 \
    --max-seq-len 2048 \
    --max-out-len 100 \
    --batch-size 4 \
    --hf-num-gpus 1 \
    --debug

```

## 1.5b O

跑了 2 个半小时
20250216031508

| dataset                                               | version | metric          | mode | qwen_1_5b_I_hf |
| ----------------------------------------------------- | ------- | --------------- | ---- | -------------- |
| lawbench-1-1-article_recitation-0-shot                | 056a7f  | score           | gen  | 16.92          |
| lawbench-1-2-knowledge_question_answering-0-shot      | 056a7f  | score           | gen  | 50.80          |
| lawbench-1-2-knowledge_question_answering-0-shot      | 056a7f  | abstention_rate | gen  | 0.00           |
| lawbench-2-1-document_proofreading-0-shot             | 056a7f  | score           | gen  | 8.83           |
| lawbench-2-2-dispute_focus_identification-0-shot      | 056a7f  | score           | gen  | 16.40          |
| lawbench-2-2-dispute_focus_identification-0-shot      | 056a7f  | abstention_rate | gen  | 1.00           |
| lawbench-2-3-marital_disputes_identification-0-shot   | 056a7f  | score           | gen  | 38.93          |
| lawbench-2-3-marital_disputes_identification-0-shot   | 056a7f  | abstention_rate | gen  | 9.20           |
| lawbench-2-4-issue_topic_identification-0-shot        | 056a7f  | score           | gen  | 26.60          |
| lawbench-2-4-issue_topic_identification-0-shot        | 056a7f  | abstention_rate | gen  | 35.00          |
| lawbench-2-5-reading_comprehension-0-shot             | 056a7f  | score           | gen  | 66.15          |
| lawbench-2-6-named_entity_recognition-0-shot          | 056a7f  | score           | gen  | 46.45          |
| lawbench-2-6-named_entity_recognition-0-shot          | 056a7f  | anstention_rate | gen  | 6.80           |
| lawbench-2-7-opinion_summarization-0-shot             | 056a7f  | score           | gen  | 32.23          |
| lawbench-2-8-argument_mining-0-shot                   | 056a7f  | score           | gen  | 42.40          |
| lawbench-2-8-argument_mining-0-shot                   | 056a7f  | abstention_rate | gen  | 0.00           |
| lawbench-2-9-event_detection-0-shot                   | 056a7f  | score           | gen  | 51.87          |
| lawbench-2-9-event_detection-0-shot                   | 056a7f  | abstention_rate | gen  | 9.00           |
| lawbench-2-10-trigger_word_extraction-0-shot          | 056a7f  | score           | gen  | 20.09          |
| lawbench-3-1-fact_based_article_prediction-0-shot     | 056a7f  | score           | gen  | 43.59          |
| lawbench-3-1-fact_based_article_prediction-0-shot     | 056a7f  | abstention_rate | gen  | 25.00          |
| lawbench-3-2-scene_based_article_prediction-0-shot    | 056a7f  | score           | gen  | 26.08          |
| lawbench-3-3-charge_prediction-0-shot                 | 056a7f  | score           | gen  | 41.36          |
| lawbench-3-3-charge_prediction-0-shot                 | 056a7f  | abstention_rate | gen  | 18.80          |
| lawbench-3-4-prison_term_prediction_wo_article-0-shot | 056a7f  | score           | gen  | 81.55          |
| lawbench-3-4-prison_term_prediction_wo_article-0-shot | 056a7f  | abstention_rate | gen  | 0.60           |
| lawbench-3-5-prison_term_prediction_w_article-0-shot  | 056a7f  | score           | gen  | 81.15          |
| lawbench-3-5-prison_term_prediction_w_article-0-shot  | 056a7f  | abstention_rate | gen  | 0.80           |
| lawbench-3-6-case_analysis-0-shot                     | 056a7f  | score           | gen  | 44.20          |
| lawbench-3-6-case_analysis-0-shot                     | 056a7f  | abstention_rate | gen  | 0.00           |
| lawbench-3-7-criminal_damages_calculation-0-shot      | 056a7f  | score           | gen  | 44.20          |
| lawbench-3-7-criminal_damages_calculation-0-shot      | 056a7f  | abstention_rate | gen  | 0.00           |
| lawbench-3-8-consultation-0-shot                      | 056a7f  | score           | gen  | 18.29          |
|                                                       |         |                 |      |                |

---

准备微调数据：

1-1.json: 共 500 条数据, train: 400 条, eval: 100 条
1-2.json: 共 500 条数据, train: 400 条, eval: 100 条
2-1.json: 共 500 条数据, train: 400 条, eval: 100 条
2-10.json: 共 500 条数据, train: 400 条, eval: 100 条
2-2.json: 共 500 条数据, train: 400 条, eval: 100 条
2-3.json: 共 500 条数据, train: 400 条, eval: 100 条
2-4.json: 共 500 条数据, train: 400 条, eval: 100 条
2-5.json: 共 500 条数据, train: 400 条, eval: 100 条
2-6.json: 共 500 条数据, train: 400 条, eval: 100 条
2-7.json: 共 500 条数据, train: 400 条, eval: 100 条
2-8.json: 共 500 条数据, train: 400 条, eval: 100 条
2-9.json: 共 500 条数据, train: 400 条, eval: 100 条
3-1.json: 共 500 条数据, train: 400 条, eval: 100 条
3-2.json: 共 500 条数据, train: 400 条, eval: 100 条
3-3.json: 共 500 条数据, train: 400 条, eval: 100 条
3-4.json: 共 500 条数据, train: 400 条, eval: 100 条
3-5.json: 共 500 条数据, train: 400 条, eval: 100 条
3-6.json: 共 500 条数据, train: 400 条, eval: 100 条
3-7.json: 共 500 条数据, train: 400 条, eval: 100 条
3-8.json: 共 500 条数据, train: 400 条, eval: 100 条

```python

import os

import json

import random

import glob

from math import ceil

  

# 输入目录（原始数据）

input_dir = '/home/sagemaker-user/DQFP/dataset/lawbench_zero_shot'

  

# 输出目录（保存分割后的数据）

output_base = '/home/sagemaker-user/DQFP/dataset/lawbench_self_split'

train_dir = os.path.join(output_base, 'train')

eval_dir = os.path.join(output_base, 'eval')

  

# 创建输出目录及子目录

os.makedirs(train_dir, exist_ok=True)

os.makedirs(eval_dir, exist_ok=True)

  

# 获取 input_dir 下所有 .json 文件（排除子目录中的文件）

json_files = glob.glob(os.path.join(input_dir, '*.json'))

  

for file_path in json_files:

    with open(file_path, 'r', encoding='utf-8') as f:

        try:

            data = json.load(f)

        except Exception as e:

            print(f"读取 {file_path} 出错: {e}")

            continue

  

    if not isinstance(data, list):

        print(f"{file_path} 中的数据格式不是列表，跳过")

        continue

  

    # 随机打乱数据顺序

    random.shuffle(data)

  

    # 按 20% 分割（使用 ceil 确保 eval 至少有一个样本）

    split_index = ceil(len(data) * 0.2)

    eval_data = data[:split_index]

    train_data = data[split_index:]

  

    base_name = os.path.basename(file_path)

    train_file = os.path.join(train_dir, base_name)

    eval_file = os.path.join(eval_dir, base_name)

  

    with open(train_file, 'w', encoding='utf-8') as f:

        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(eval_file, 'w', encoding='utf-8') as f:

        json.dump(eval_data, f, ensure_ascii=False, indent=2)

  

    print(f"{base_name}: 共 {len(data)} 条数据, train: {len(train_data)} 条, eval: {len(eval_data)} 条")
```

## 1.5b O

原始 1.5 吧模型在 2/8 分的 eval 测试里面

| dataset | version | metric | mode | qwen_1_5b_I_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench-1-1-article_recitation-1-shot | 056a7f | score | gen | 17.25 |

| lawbench-1-2-knowledge_question_answering-1-shot | 056a7f | score | gen | 53.00 |

| lawbench-1-2-knowledge_question_answering-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-2-1-document_proofreading-1-shot | 056a7f | score | gen | 7.92 |

| lawbench-2-2-dispute_focus_identification-1-shot | 056a7f | score | gen | 15.00 |

| lawbench-2-2-dispute_focus_identification-1-shot | 056a7f | abstention_rate | gen | 1.00 |

| lawbench-2-3-marital_disputes_identification-1-shot | 056a7f | score | gen | 38.27 |

| lawbench-2-3-marital_disputes_identification-1-shot | 056a7f | abstention_rate | gen | 15.00 |

| lawbench-2-4-issue_topic_identification-1-shot | 056a7f | score | gen | 27.00 |

| lawbench-2-4-issue_topic_identification-1-shot | 056a7f | abstention_rate | gen | 41.00 |

| lawbench-2-5-reading_comprehension-1-shot | 056a7f | score | gen | 63.74 |

| lawbench-2-6-named_entity_recognition-1-shot | 056a7f | score | gen | 42.04 |

| lawbench-2-6-named_entity_recognition-1-shot | 056a7f | anstention_rate | gen | 10.00 |

| lawbench-2-7-opinion_summarization-1-shot | 056a7f | score | gen | 30.29 |

| lawbench-2-8-argument_mining-1-shot | 056a7f | score | gen | 43.00 |

| lawbench-2-8-argument_mining-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-2-9-event_detection-1-shot | 056a7f | score | gen | 48.20 |

| lawbench-2-9-event_detection-1-shot | 056a7f | abstention_rate | gen | 9.00 |

| lawbench-2-10-trigger_word_extraction-1-shot | 056a7f | score | gen | 21.17 |

| lawbench-3-1-fact_based_article_prediction-1-shot | 056a7f | score | gen | 41.00 |

| lawbench-3-1-fact_based_article_prediction-1-shot | 056a7f | abstention_rate | gen | 22.00 |

| lawbench-3-2-scene_based_article_prediction-1-shot | 056a7f | score | gen | 26.58 |

| lawbench-3-3-charge_prediction-1-shot | 056a7f | score | gen | 37.47 |

| lawbench-3-3-charge_prediction-1-shot | 056a7f | abstention_rate | gen | 19.00 |

| lawbench-3-4-prison_term_prediction_wo_article-1-shot | 056a7f | score | gen | 81.79 |

| lawbench-3-4-prison_term_prediction_wo_article-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-3-5-prison_term_prediction_w_article-1-shot | 056a7f | score | gen | 81.67 |

| lawbench-3-5-prison_term_prediction_w_article-1-shot | 056a7f | abstention_rate | gen | 1.00 |

| lawbench-3-6-case_analysis-1-shot | 056a7f | score | gen | 41.00 |

| lawbench-3-6-case_analysis-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-3-7-criminal_damages_calculation-1-shot | 056a7f | score | gen | 40.00 |

| lawbench-3-7-criminal_damages_calculation-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-3-8-consultation-1-shot | 056a7f | score | gen | 17.86 |

---

## 1.5b FT

| dataset | version | metric | mode | qwen_1_5b_I_F_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench-1-1-article_recitation-1-shot | 056a7f | score | gen | 19.92 |

| lawbench-1-2-knowledge_question_answering-1-shot | 056a7f | score | gen | 58.00 |

| lawbench-1-2-knowledge_question_answering-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-2-1-document_proofreading-1-shot | 056a7f | score | gen | 35.99 |

| lawbench-2-2-dispute_focus_identification-1-shot | 056a7f | score | gen | 54.00 |

| lawbench-2-2-dispute_focus_identification-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-2-3-marital_disputes_identification-1-shot | 056a7f | score | gen | 81.18 |

| lawbench-2-3-marital_disputes_identification-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-2-4-issue_topic_identification-1-shot | 056a7f | score | gen | 41.00 |

| lawbench-2-4-issue_topic_identification-1-shot | 056a7f | abstention_rate | gen | 3.00 |

| lawbench-2-5-reading_comprehension-1-shot | 056a7f | score | gen | 75.71 |

| lawbench-2-6-named_entity_recognition-1-shot | 056a7f | score | gen | 50.38 |

| lawbench-2-6-named_entity_recognition-1-shot | 056a7f | anstention_rate | gen | 0.00 |

| lawbench-2-7-opinion_summarization-1-shot | 056a7f | score | gen | 48.89 |

| lawbench-2-8-argument_mining-1-shot | 056a7f | score | gen | 62.00 |

| lawbench-2-8-argument_mining-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-2-9-event_detection-1-shot | 056a7f | score | gen | 75.50 |

| lawbench-2-9-event_detection-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-2-10-trigger_word_extraction-1-shot | 056a7f | score | gen | 78.85 |

| lawbench-3-1-fact_based_article_prediction-1-shot | 056a7f | score | gen | 72.52 |

| lawbench-3-1-fact_based_article_prediction-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-3-2-scene_based_article_prediction-1-shot | 056a7f | score | gen | 36.36 |

| lawbench-3-3-charge_prediction-1-shot | 056a7f | score | gen | 46.10 |

| lawbench-3-3-charge_prediction-1-shot | 056a7f | abstention_rate | gen | 6.00 |

| lawbench-3-4-prison_term_prediction_wo_article-1-shot | 056a7f | score | gen | 81.20 |

| lawbench-3-4-prison_term_prediction_wo_article-1-shot | 056a7f | abstention_rate | gen | 1.00 |

| lawbench-3-5-prison_term_prediction_w_article-1-shot | 056a7f | score | gen | 78.93 |

| lawbench-3-5-prison_term_prediction_w_article-1-shot | 056a7f | abstention_rate | gen | 4.00 |

| lawbench-3-6-case_analysis-1-shot | 056a7f | score | gen | 47.00 |

| lawbench-3-6-case_analysis-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-3-7-criminal_damages_calculation-1-shot | 056a7f | score | gen | 59.00 |

| lawbench-3-7-criminal_damages_calculation-1-shot | 056a7f | abstention_rate | gen | 0.00 |

| lawbench-3-8-consultation-1-shot | 056a7f | score | gen | 27.44 |

```bash
python run.py --hf-type chat \
    --hf-path /home/sagemaker-user/DQFP/model/qwen_3b_I_awq_self \
    --tokenizer-path /home/sagemaker-user/DQFP/model/qwen_3b_I_awq_self \
    --datasets lawbench_one_shot_gen_002588 \
    --max-seq-len 512 \
    --max-out-len 100 \
    --batch-size 4 \
    --hf-num-gpus 1 \
    --debug
```

## 3b Awq Self

| dataset                                               | version | metric          | mode | qwen_3b_I_awq_self_hf |
| ----------------------------------------------------- | ------- | --------------- | ---- | --------------------- |
| lawbench-1-1-article_recitation-1-shot                | 056a7f  | score           | gen  | 11.78                 |
| lawbench-1-2-knowledge_question_answering-1-shot      | 056a7f  | score           | gen  | 51.00                 |
| lawbench-1-2-knowledge_question_answering-1-shot      | 056a7f  | abstention_rate | gen  | 0.00                  |
| lawbench-2-1-document_proofreading-1-shot             | 056a7f  | score           | gen  | 17.46                 |
| lawbench-2-2-dispute_focus_identification-1-shot      | 056a7f  | score           | gen  | 32.00                 |
| lawbench-2-2-dispute_focus_identification-1-shot      | 056a7f  | abstention_rate | gen  | 20.00                 |
| lawbench-2-3-marital_disputes_identification-1-shot   | 056a7f  | score           | gen  | 31.91                 |
| lawbench-2-3-marital_disputes_identification-1-shot   | 056a7f  | abstention_rate | gen  | 3.00                  |
| lawbench-2-4-issue_topic_identification-1-shot        | 056a7f  | score           | gen  | 26.00                 |
| lawbench-2-4-issue_topic_identification-1-shot        | 056a7f  | abstention_rate | gen  | 29.00                 |
| lawbench-2-5-reading_comprehension-1-shot             | 056a7f  | score           | gen  | 37.56                 |
| lawbench-2-6-named_entity_recognition-1-shot          | 056a7f  | score           | gen  | 3.84                  |
| lawbench-2-6-named_entity_recognition-1-shot          | 056a7f  | anstention_rate | gen  | 91.00                 |
| lawbench-2-7-opinion_summarization-1-shot             | 056a7f  | score           | gen  | 29.46                 |
| lawbench-2-8-argument_mining-1-shot                   | 056a7f  | score           | gen  | 29.00                 |
| lawbench-2-8-argument_mining-1-shot                   | 056a7f  | abstention_rate | gen  | 3.00                  |
| lawbench-2-9-event_detection-1-shot                   | 056a7f  | score           | gen  | 56.98                 |
| lawbench-2-9-event_detection-1-shot                   | 056a7f  | abstention_rate | gen  | 3.00                  |
| lawbench-2-10-trigger_word_extraction-1-shot          | 056a7f  | score           | gen  | 16.55                 |
| lawbench-3-1-fact_based_article_prediction-1-shot     | 056a7f  | score           | gen  | 39.15                 |
| lawbench-3-1-fact_based_article_prediction-1-shot     | 056a7f  | abstention_rate | gen  | 1.00                  |
| lawbench-3-2-scene_based_article_prediction-1-shot    | 056a7f  | score           | gen  | 26.27                 |
| lawbench-3-3-charge_prediction-1-shot                 | 056a7f  | score           | gen  | 41.03                 |
| lawbench-3-3-charge_prediction-1-shot                 | 056a7f  | abstention_rate | gen  | 17.00                 |
| lawbench-3-4-prison_term_prediction_wo_article-1-shot | 056a7f  | score           | gen  | 76.75                 |
| lawbench-3-4-prison_term_prediction_wo_article-1-shot | 056a7f  | abstention_rate | gen  | 7.00                  |
| lawbench-3-5-prison_term_prediction_w_article-1-shot  | 056a7f  | score           | gen  | 73.07                 |
| lawbench-3-5-prison_term_prediction_w_article-1-shot  | 056a7f  | abstention_rate | gen  | 9.00                  |
| lawbench-3-6-case_analysis-1-shot                     | 056a7f  | score           | gen  | 49.00                 |
| lawbench-3-6-case_analysis-1-shot                     | 056a7f  | abstention_rate | gen  | 0.00                  |
| lawbench-3-7-criminal_damages_calculation-1-shot      | 056a7f  | score           | gen  | 49.00                 |
| lawbench-3-7-criminal_damages_calculation-1-shot      | 056a7f  | abstention_rate | gen  | 1.00                  |
| lawbench-3-8-consultation-1-shot                      | 056a7f  | score           | gen  | 18.99                 |

## 3b Ft

```bash
python /root/autodl-tmp/opencompass/opencompass/run.py --datasets lawbench3 --hf-type base --hf-path /root/autodl-tmp/model/q3bft_out_merged --debug
/root/autodl-tmp/opencompass/opencompass/outputs/default/20250416_134746/summary/summary_20250416_134746.md
```

dataset,version,metric,mode,q3bft_out_merged_hf
lawbench3-1-1-article_recitation-test,056a7f,score,gen,4.86
lawbench3-1-2-knowledge_question_answering-test,056a7f,score,gen,0.00
lawbench3-1-2-knowledge_question_answering-test,056a7f,abstention_rate,gen,50.00
lawbench3-2-1-document_proofreading-test,056a7f,score,gen,0.00
lawbench3-2-2-dispute_focus_identification-test,056a7f,score,gen,0.00
lawbench3-2-2-dispute_focus_identification-test,056a7f,abstention_rate,gen,100.00
lawbench3-2-3-marital_disputes_identification-test,056a7f,score,gen,16.67
lawbench3-2-3-marital_disputes_identification-test,056a7f,abstention_rate,gen,50.00
lawbench3-2-4-issue_topic_identification-test,056a7f,score,gen,0.00
lawbench3-2-4-issue_topic_identification-test,056a7f,abstention_rate,gen,50.00
lawbench3-2-5-reading_comprehension-test,056a7f,score,gen,4.97
lawbench3-2-6-named_entity_recognition-test,056a7f,score,gen,4.02
lawbench3-2-6-named_entity_recognition-test,056a7f,anstention_rate,gen,50.00
lawbench3-2-7-opinion_summarization-test,056a7f,score,gen,5.22
lawbench3-2-8-argument_mining-test,056a7f,score,gen,0.00
lawbench3-2-8-argument_mining-test,056a7f,abstention_rate,gen,50.00
lawbench3-2-9-event_detection-test,056a7f,score,gen,0.00
lawbench3-2-9-event_detection-test,056a7f,abstention_rate,gen,100.00
lawbench3-2-10-trigger_word_extraction-test,056a7f,score,gen,7.14
lawbench3-3-1-fact_based_article_prediction-test,056a7f,score,gen,50.00
lawbench3-3-1-fact_based_article_prediction-test,056a7f,abstention_rate,gen,50.00
lawbench3-3-2-scene_based_article_prediction-test,056a7f,score,gen,8.58
lawbench3-3-3-charge_prediction-test,056a7f,score,gen,50.00
lawbench3-3-3-charge_prediction-test,056a7f,abstention_rate,gen,50.00
lawbench3-3-4-prison_term_prediction_wo_article-test,056a7f,score,gen,48.60
lawbench3-3-4-prison_term_prediction_wo_article-test,056a7f,abstention_rate,gen,50.00
lawbench3-3-5-prison_term_prediction_w_article-test,056a7f,score,gen,0.00
lawbench3-3-5-prison_term_prediction_w_article-test,056a7f,abstention_rate,gen,50.00
lawbench3-3-6-case_analysis-test,056a7f,score,gen,0.00
lawbench3-3-6-case_analysis-test,056a7f,abstention_rate,gen,50.00
lawbench3-3-7-criminal_damages_calculation-test,056a7f,score,gen,50.00
lawbench3-3-7-criminal_damages_calculation-test,056a7f,abstention_rate,gen,50.00
lawbench3-3-8-consultation-test,056a7f,score,gen,14.59

## 3b Awq Hf

```bash
/root/autodl-tmp/opencompass/opencompass/outputs/default/20250418_024733/summary/summary_20250418_024733.md
```

| dataset | version | metric | mode | q3bawq_hf_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 3.54 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 0.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 4.33 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 100.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 9.96 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 0.19 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 16.67 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 7.01 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 40.00 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 41.35 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 4.44 |

## 3b Prune Atp1

根据 wikitext 剪枝

```shell
/root/autodl-tmp/opencompass/opencompass/outputs/default/20250418_035052/summary/summary_20250418_035052.md
```

| dataset | version | metric | mode | q3bp_attempt1_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 0.91 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 0.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 33.33 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 4.17 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 7.41 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 50.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 4.73 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 1.17 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 25.00 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 34.36 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 5.71 |

## 3b Prune Atp 2

根据 lawbenchprune

```
/root/autodl-tmp/opencompass/opencompass/outputs/default/20250418_042323/summary/summary_20250418_042323.md
```

| dataset | version | metric | mode | q3bp_attempt2_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 1.13 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 0.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 3.70 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 100.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 2.80 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 11.11 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 0.07 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 1.69 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 37.32 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 5.77 |

## 3b_q_p

```bash
/root/autodl-tmp/opencompass/opencompass/outputs/default/20250418_044237/summary/summary_20250418_044237.md
```

| dataset | version | metric | mode | q3b_q_p_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 1.18 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 0.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 4.03 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 100.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 5.37 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 0.29 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 1.29 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 3.30 |

## 3b_f_p

```bash

/root/autodl-tmp/opencompass/opencompass/outputs/default/20250418_050548/summary/summary_20250418_050548.md
```

| dataset | version | metric | mode | q3b_ft_p_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 2.09 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 0.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 0.08 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 100.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 5.38 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 2.16 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 1.54 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 33.33 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 1.60 |

## 3b_f

| dataset | version | metric | mode | q3b_ft_adapter_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 3.04 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 50.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 40.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 5.33 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 7.74 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 50.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 4.96 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 50.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 2.26 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 2.33 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 25.00 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 48.60 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 15.53 |

## q3b_q_ft

| dataset | version | metric | mode | q3b_q_ft_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 4.09 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 50.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 40.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 4.36 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 17.50 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 50.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 16.64 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 1.16 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 3.27 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 14.29 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 48.60 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 12.94 |

## q3b_p_ft

| dataset | version | metric | mode | q3b_p_ft_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 4.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 0.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 25.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 4.41 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 5.74 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 50.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 4.59 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 33.33 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 3.85 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 16.67 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 25.37 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 48.60 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 8.05 |

## q3bft_q

```bash
/root/autodl-tmp/opencompass/opencompass/outputs/default/20250419_125414/summary/summary_20250419_125414.md
```

| dataset | version | metric | mode | q3bft_q_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 3.23 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 50.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 40.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 5.04 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 5.87 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 50.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 5.40 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 50.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 1.76 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 2.39 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 48.60 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 6.70 |

## q3bp_q

```bash
/root/autodl-tmp/opencompass/opencompass/outputs/default/20250419_132443/summary/summary_20250419_132443.md
```

| dataset | version | metric | mode | q3bp_q_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 1.53 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 0.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 3.98 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 100.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 6.27 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 0.09 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 0.58 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 44.96 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 1.13 |

## q3b_ft_p_q

```
/root/autodl-tmp/opencompass/opencompass/outputs/default/20250419_135347/summary/summary_20250419_135347.md
```

| dataset | version | metric | mode | q3b_ft_p_q_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 2.32 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 0.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 20.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 4.45 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 100.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 4.18 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 1.34 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 42.40 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 1.79 |

## q3b_p_ft_q

```
/root/autodl-tmp/opencompass/opencompass/outputs/default/20250419_142041/summary/summary_20250419_142041.md
```

| dataset | version | metric | mode | q3b_p_ft_q_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 3.78 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 0.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 4.35 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 9.07 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 50.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 19.21 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 50.00 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 6.46 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 25.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 2.96 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 20.00 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 38.87 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 6.74 |

## q3b_q_ft_p

```
/root/autodl-tmp/opencompass/opencompass/outputs/default/20250419_144555/summary/summary_20250419_144555.md
```

| dataset | version | metric | mode | q3b_q_ft_p_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 1.95 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 0.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 25.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 3.14 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 100.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 1.66 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 50.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 2.05 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 25.00 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 42.40 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 5.08 |

## q3bft_q_p

```
/root/autodl-tmp/opencompass/opencompass/outputs/default/20250419_145801/summary/summary_20250419_145801.md
```

| dataset | version | metric | mode | q3bft_q_p_hf |

|----- | ----- | ----- | ----- | -----|

| lawbench3-1-1-article_recitation-test | 056a7f | score | gen | 1.87 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | score | gen | 0.00 |

| lawbench3-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-1-document_proofreading-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-5-reading_comprehension-test | 056a7f | score | gen | 4.31 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-6-named_entity_recognition-test | 056a7f | anstention_rate | gen | 100.00 |

| lawbench3-2-7-opinion_summarization-test | 056a7f | score | gen | 3.37 |

| lawbench3-2-8-argument_mining-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-2-9-event_detection-test | 056a7f | score | gen | 0.00 |

| lawbench3-2-9-event_detection-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-2-10-trigger_word_extraction-test | 056a7f | score | gen | 0.12 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 0.76 |

| lawbench3-3-3-charge_prediction-test | 056a7f | score | gen | 33.33 |

| lawbench3-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 42.40 |

| lawbench3-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 50.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 0.00 |

| lawbench3-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 100.00 |

| lawbench3-3-8-consultation-test | 056a7f | score | gen | 0.11 |
