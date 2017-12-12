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
    - [Leveraging uncertainty information from deep neural networks for disease detection](http://www.biorxiv.org/content/early/2016/10/28/084210)
        - Run same network to collect our metrics

## Related work
    - [A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks](https://arxiv.org/abs/1610.02136)
        - Works only for classification
        - Metrics are AUCROC and AUPR (for correctly incorrectly classified examples) using max probability from softmax
        - The proposed Abnormality detection (with auxiliary decoders) works only for in-out distribution detection
    - [Leveraging uncertainty information from deep neural networks for disease detection](http://www.biorxiv.org/content/early/2016/10/28/084210)
    - [Uncertainties in parameters estimated with neural networks: application to strong gravitational lensing](https://arxiv.org/abs/1708.08843)
    - [Modelling Uncertainty in Deep Learning for Camera Relocalization](https://arxiv.org/pdf/1509.05909.pdf)


# Ideas

- Amount of data in training as a function of uncertainty
