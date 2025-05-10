---
title: 
aliases: 
tags: 
author: tusrau
date created: 20250226 03:50
date updated: 20250226 09:15
---

# Llama2 7b Prune

 [[20250226]]
2\*4090
![[20250226092139_benchmark.log]]

![[20250226092139_gpu_usage.csv]]

| Dataset                                              | Version | Metric            | Mode | 7b_p_hf |
|------------------------------------------------------|---------|-------------------|------|---------|
| lawbench3-1-1-article_recitation-test                | 056a7f | score             | gen  | 2.58    |
| lawbench3-1-2-knowledge_question_answering-test      | 056a7f | score             | gen  | 0.00    |
| lawbench3-1-2-knowledge_question_answering-test      | 056a7f | abstention_rate   | gen  | 0.00    |
| lawbench3-2-1-document_proofreading-test             | 056a7f | score             | gen  | 0.00    |
| lawbench3-2-2-dispute_focus_identification-test      | 056a7f | score             | gen  | 0.00    |
| lawbench3-2-2-dispute_focus_identification-test      | 056a7f | abstention_rate   | gen  | 50.00   |
| lawbench3-2-3-marital_disputes_identification-test     | 056a7f | score             | gen  | 0.00    |
| lawbench3-2-3-marital_disputes_identification-test     | 056a7f | abstention_rate   | gen  | 100.00  |
| lawbench3-2-4-issue_topic_identification-test          | 056a7f | score             | gen  | 0.00    |
| lawbench3-2-4-issue_topic_identification-test          | 056a7f | abstention_rate   | gen  | 100.00  |
| lawbench3-2-5-reading_comprehension-test               | 056a7f | score             | gen  | 6.39    |
| lawbench3-2-6-named_entity_recognition-test            | 056a7f | score             | gen  | 0.00    |
| lawbench3-2-6-named_entity_recognition-test            | 056a7f | anstention_rate   | gen  | 100.00  |
| lawbench3-2-7-opinion_summarization-test              | 056a7f | score             | gen  | 17.90   |
| lawbench3-2-8-argument_mining-test                    | 056a7f | score             | gen  | 0.00    |
| lawbench3-2-8-argument_mining-test                    | 056a7f | abstention_rate   | gen  | 0.00    |
| lawbench3-2-9-event_detection-test                    | 056a7f | score             | gen  | 0.00    |
| lawbench3-2-9-event_detection-test                    | 056a7f | abstention_rate   | gen  | 100.00  |
| lawbench3-2-10-trigger_word_extraction-test           | 056a7f | score             | gen  | 0.41    |
| lawbench3-3-1-fact_based_article_prediction-test      | 056a7f | score             | gen  | 0.00    |
| lawbench3-3-1-fact_based_article_prediction-test      | 056a7f | abstention_rate   | gen  | 50.00   |
| lawbench3-3-2-scene_based_article_prediction-test     | 056a7f | score             | gen  | 9.76    |
| lawbench3-3-3-charge_prediction-test                  | 056a7f | score             | gen  | 0.00    |
| lawbench3-3-3-charge_prediction-test                  | 056a7f | abstention_rate   | gen  | 0.00    |
| lawbench3-3-4-prison_term_prediction_wo_article-test  | 056a7f | score             | gen  | 0.00    |
| lawbench3-3-4-prison_term_prediction_wo_article-test  | 056a7f | abstention_rate   | gen  | 100.00  |
| lawbench3-3-5-prison_term_prediction_w_article-test   | 056a7f | score             | gen  | 92.71   |
| lawbench3-3-5-prison_term_prediction_w_article-test   | 056a7f | abstention_rate   | gen  | 0.00    |
| lawbench3-3-6-case_analysis-test                     | 056a7f | score             | gen  | 0.00    |
| lawbench3-3-6-case_analysis-test                     | 056a7f | abstention_rate   | gen  | 50.00   |
| lawbench3-3-7-criminal_damages_calculation-test       | 056a7f | score             | gen  | 0.00    |
| lawbench3-3-7-criminal_damages_calculation-test       | 056a7f | abstention_rate   | gen  | 50.00   |
| lawbench3-3-8-consultation-test                       | 056a7f | score             | gen  | 11.94   |
