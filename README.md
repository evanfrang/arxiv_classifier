# Arxiv Abstract Classifier
Evan Frangipane

- [Goal](#goal)
- [Dataset](#dataset)
- [Results](#results)

## Goal

Create an End to End machine learning pipeline to classify physics paper
abstracts into broad categories and deploy the model for anyone to test.
The baseline model is logistic regression, (SVM gave essentially
identical results to LR), and BERT. Currently the logistic regression
model is being hosted on
[Render](https://arxiv-classifier.onrender.com/). Please give the app a
minute to wake up if it has gone to sleep.

## Dataset

The dataset is 100k paper entries on arxiv from 2007 to 2025. To reduce
class imbalance I created more broad categories. I did my best to
associate similar topics and also used arxiv’s classification when I
could. I present a histogram with classes and their populations.

<img src="images/class_hist.png" style="width:80.0%"
data-fig-align="center" />

## Results

Here is the confusion matrix for Logistic Regression.
<img src="images/lrconfusion.png" style="width:80.0%"
data-fig-align="center" />
