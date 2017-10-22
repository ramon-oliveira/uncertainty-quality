# Uncertainty Quality

Experiments repository for Uncertainty quality paper

## Contributions
    - New uncertainty estimator using classifier or regressor

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


## TODO:
    - [A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks](https://arxiv.org/abs/1610.02136)
        - Identify compassion points
    - [Leveraging uncertainty information from deep neural networks for disease detection](http://www.biorxiv.org/content/early/2016/10/28/084210)
        - Run same network to collect our metrics

# Related work

- [Leveraging uncertainty information from deep neural networks for disease detection](http://www.biorxiv.org/content/early/2016/10/28/084210)
- [Uncertainties in parameters estimated with neural networks: application to strong gravitational lensing](https://arxiv.org/abs/1708.08843)
- [Modelling Uncertainty in Deep Learning for Camera Relocalization](https://arxiv.org/pdf/1509.05909.pdf)
