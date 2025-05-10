---
title: 
aliases: 
tags: 
author: tusrau
date created: 20250215 06:57
date updated: 20250409 02:26
---

# MedBench 测试集

 [[20250215]]
下面是关于 MedBench 数据集修改操作的完整总结，包含两部分映射调整：

---

## 1. 将 _test.jsonl 文件映射为正常 .jsonl 文件

在一些 MedBench 子任务中，下载的数据文件名称包含后缀 `_test`（例如 `Med-Exam_test.jsonl` 或 `SafetyBench_test.jsonl`），而配置文件中期望的名称为不带 `_test` 的标准文件名（例如 `Med-Exam.jsonl` 或 `SafetyBench.jsonl`）。为了解决这一问题，你可以使用符号链接将后缀文件映射为预期文件名。操作示例（在对应目录下执行）：

```bash
# 例如，对于 Med-Exam 数据集：
cd /home/sagemaker-user/DQFP/opencompass/data/MedBench/Med-Exam
ln -s Med-Exam_test.jsonl Med-Exam.jsonl
```

如果目录下存在其他数据集，也可使用类似命令进行映射。

---

## 2. 将目录名称映射调整为配置文件期望

配置文件中 MedBench 部分期望数据文件位于路径 `./data/MedBench/MedSafety/MedSafety.jsonl`，但实际数据存放目录中没有名为 `MedSafety` 的文件夹，而是存在 `SafetyBench` 文件夹，并且文件名为 `SafetyBench.jsonl`。为使配置加载正常，你可以创建以下符号链接：

1. 在 MedBench 根目录下创建一个符号链接，将 `SafetyBench` 映射为 `MedSafety`：

	```bash
    cd /home/sagemaker-user/DQFP/opencompass/data/MedBench
    ln -s SafetyBench MedSafety
    ```

2. 进入新建的 `MedSafety` 目录后，再创建内部符号链接，将实际文件 `SafetyBench.jsonl` 映射为 `MedSafety.jsonl`：

	```bash
    cd MedSafety
    ln -s SafetyBench.jsonl MedSafety.jsonl
    ```

这样，当 OpenCompass 尝试访问 `./data/MedBench/MedSafety/MedSafety.jsonl` 时，系统会自动读取实际的文件内容。

---

## 总结

- **文件名映射**：通过创建符号链接，将下载数据中带有 `_test` 后缀的文件映射为不带 `_test` 的标准文件名，确保配置加载时文件名称一致。
- **目录名称映射**：由于配置文件期望的 MedBench 子数据集目录名称为 `MedSafety`，而实际数据存放目录中使用的是 `SafetyBench`，因此在 MedBench 目录下创建符号链接 `MedSafety -> SafetyBench`，同时在内部创建映射链接，将 `SafetyBench.jsonl` 映射为 `MedSafety.jsonl`。

通过以上两步操作，配置文件中预期的路径和文件名称就能与实际数据存放保持一致，从而解决 FileNotFoundError 的问题，使 OpenCompass 正确加载 MedBench 数据集。

但最后
