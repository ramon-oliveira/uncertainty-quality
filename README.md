# Uncertainty Quality

Experiments repositoy for Uncertainty quality paper

## Variation factors:
1) Datasets:
    - MNIST
    - CIFAR10
    - Retinopathy
    - Melanoma*
2) Inference:
    - Maximum Likelihood (?)
    - Dropout
    - VI
    - SGHMC*


## Uncertainty metrics:
    - Entropy
    - Entropy mean
    - Mean entropy
    - Standard deviation (from mean score argmax)
    - Standard deviation mean (from all class)
    - Classifier
        - Uses all previous uncertainties as features
        - Plus the mean score for all classes
        - XGBClassifier

# Applications

- http://www.biorxiv.org/content/early/2016/10/28/084210
- https://arxiv.org/abs/1708.08843
