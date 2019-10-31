---
layout:     post
title:      k-Fold Cross-Validation
subtitle:   Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample
date:       2019-10-17
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 深度学习
---


# k-Fold Cross-Validation

- Shuffle the dataset randomly.
- Split the dataset into k groups
- For each unique group:
    - Take the group as a hold out or test data set

    - Take the remaining groups as a training data set

    - Fit a model on the training set and evaluate it on the test set

    - Retain the evaluation score and discard the model

- Summarize the skill of the model using the sample of model evaluation scores

![](/img/grid_search_workflow.png)


![](/img/grid_search_cross_validation.png)

# 模型集成 based on k-Fold Cross-Validation

