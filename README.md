# Uncertainty Quality

Experiments repository for Uncertainty quality paper

## Contributions
    - No need to be Bayesian to obtain uncertainties
    - New uncertainty estimator using classifier

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


## TODO:
    - Fix test data generators
        - Check test accuracy
    - Variational Inference MLP

# Applications

- [Leveraging uncertainty information from deep neural networks for disease detection](http://www.biorxiv.org/content/early/2016/10/28/084210)
    - Inappropriate baseline
    - Comparation against random split
- [Uncertainties in parameters estimated with neural networks: application to strong gravitational lensing](https://arxiv.org/abs/1708.08843)
    - Comparation against itself
    - No baseline
- [Modelling Uncertainty in Deep Learning for Camera Relocalization](https://arxiv.org/pdf/1509.05909.pdf)
>>> **A non-Bayesian system which
outputs point estimates does not interpret if the model is
making sensible predictions or just guessing at random.**
By measuring uncertainty we can understand with what
confidence we can trust the prediction.
