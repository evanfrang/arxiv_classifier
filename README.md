---
author:
- Evan Frangipane
authors:
- Evan Frangipane
execute:
  echo: true
  message: false
  warning: false
title: Arxiv Abstract Classifier
toc-title: Table of contents
---

## Goal

Create an End to End machine learning pipeline to classify physics paper
abstracts into broad categories and deploy the model for anyone to test.
The baseline model is logistic regression, additionally using SVM, and
BERT. Currently the logistic regression model is being hosted on
[Render](https://arxiv-classifier.onrender.com/). Please give the app a
minute to wake up if it has gone to sleep.

## Dataset
