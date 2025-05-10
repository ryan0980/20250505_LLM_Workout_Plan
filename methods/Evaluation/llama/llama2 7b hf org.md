---
title: 
aliases: 
tags: 
author: tusrau
date created: 20250225 12:47
date updated: 20250226 03:55
---

# Llama2 7b

 [[20250225]]
lawbench eval with 2 4090
![[20250225123729_gpu_usage.csv]]

![[20250225123729_benchmark.log]]

| Dataset | Version | Metric | Mode | 7b_o_HF_hf |
|---------|---------|--------|------|------------|
| lawbench2-1-1-article_recitation-test | 056a7f | score | gen | 2.61 |
| lawbench2-1-2-knowledge_question_answering-test | 056a7f | score | gen | 1.00 |
| lawbench2-1-2-knowledge_question_answering-test | 056a7f | abstention_rate | gen | 5.00 |
| lawbench2-2-1-document_proofreading-test | 056a7f | score | gen | 1.82 |
| lawbench2-2-2-dispute_focus_identification-test | 056a7f | score | gen | 1.00 |
| lawbench2-2-2-dispute_focus_identification-test | 056a7f | abstention_rate | gen | 72.00 |
| lawbench2-2-3-marital_disputes_identification-test | 056a7f | score | gen | 17.14 |
| lawbench2-2-3-marital_disputes_identification-test | 056a7f | abstention_rate | gen | 35.00 |
| lawbench2-2-4-issue_topic_identification-test | 056a7f | score | gen | 3.00 |
| lawbench2-2-4-issue_topic_identification-test | 056a7f | abstention_rate | gen | 87.00 |
| lawbench2-2-5-reading_comprehension-test | 056a7f | score | gen | 5.38 |
| lawbench2-2-6-named_entity_recognition-test | 056a7f | score | gen | 30.31 |
| lawbench2-2-6-named_entity_recognition-test | 056a7f | abstention_rate | gen | 15.00 |
| lawbench2-2-7-opinion_summarization-test | 056a7f | score | gen | 17.32 |
| lawbench2-2-8-argument_mining-test | 056a7f | score | gen | 0.00 |
| lawbench2-2-8-argument_mining-test | 056a7f | abstention_rate | gen | 5.00 |
| lawbench2-2-9-event_detection-test | 056a7f | score | gen | 11.60 |
| lawbench2-2-9-event_detection-test | 056a7f | abstention_rate | gen | 44.00 |
| lawbench2-2-10-trigger_word_extraction-test | 056a7f | score | gen | 0.37 |
| lawbench2-3-1-fact_based_article_prediction-test | 056a7f | score | gen | 0.00 |
| lawbench2-3-1-fact_based_article_prediction-test | 056a7f | abstention_rate | gen | 28.00 |
| lawbench2-3-2-scene_based_article_prediction-test | 056a7f | score | gen | 7.84 |
| lawbench2-3-3-charge_prediction-test | 056a7f | score | gen | 6.20 |
| lawbench2-3-3-charge_prediction-test | 056a7f | abstention_rate | gen | 37.00 |
| lawbench2-3-4-prison_term_prediction_wo_article-test | 056a7f | score | gen | 61.82 |
| lawbench2-3-4-prison_term_prediction_wo_article-test | 056a7f | abstention_rate | gen | 22.00 |
| lawbench2-3-5-prison_term_prediction_w_article-test | 056a7f | score | gen | 44.46 |
| lawbench2-3-5-prison_term_prediction_w_article-test | 056a7f | abstention_rate | gen | 40.00 |
| lawbench2-3-6-case_analysis-test | 056a7f | score | gen | 0.00 |
| lawbench2-3-6-case_analysis-test | 056a7f | abstention_rate | gen | 1.00 |
| lawbench2-3-7-criminal_damages_calculation-test | 056a7f | score | gen | 14.00 |
| lawbench2-3-7-criminal_damages_calculation-test | 056a7f | abstention_rate | gen | 6.00 |
| lawbench2-3-8-consultation-test | 056a7f | score | gen | 9.05 |

---

GPU Usage Statistics (data with GPU utilization >= 10%):
-------------------------------------------------------
Sampling Time Range: 2025-02-25 12:37:38.660000 to 2025-02-25 16:00:21.125000
Duration: 0 days 03:22:42.465000

GPU Utilization (%):
	Mean: 43.69 %
	Max: 100.00 %
	95th Percentile: 52.00 %
	90th Percentile: 50.00 %

Memory Utilization (%):
	Mean: 43.94 %
	Max: 70.00 %
	95th Percentile: 55.00 %
	90th Percentile: 53.00 %

Memory Usage (MiB):
	Mean: 21311.80 MiB
	Max: 24145.00 MiB
	95th Percentile: 23011.00 MiB
	90th Percentile: 23011.00 MiB
-------------------------------------------------------
Total data points: 24666
Data points after filtering: 24164
