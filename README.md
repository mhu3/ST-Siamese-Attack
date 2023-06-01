# ST-iFGSM: Enhancing Robustness of Human Mobility Signature Identification Model via Spatial-Temporal Iterative FGSM

This repository contains a Keras implementation of the algorithm presented in the paper ST-iFGSM: Enhancing Robustness of Human Mobility Signature Identification Model via Spatial-Temporal Iterative FGSM

In this paper, we aim to improve the robustness of the HuMID model by generating useful adversarial trajectories for further training the model. To accomplish this, we design a Spatial Temporal iterative Fast Gradient Sign Method with 𝐿0 regularization – ST-iFGSM – to generate adversarial attacks on state-of-the-art (SOTA) HuMID models. 


## Prerequisites
- [Python 3.9.12](https://www.continuum.io/downloads)
- [Keras 2.10.0](https://keras.io/)
- [Tensorflow 2.10.1](https://www.tensorflow.org/)
- GPU for fast training


## Usage
- Execute ```python classification.py``` to train the multi-classification model.
- Execute ```python train.py``` to train ST-siamese model.
- Execute ```python classification_attack.py`` to attack the multi-classification model
- Execute ```python siamese_attack.py`` to attack the ST-Siamese model
- Execute ```python fast_adversarial_train.py`` to do Fast ST-FGSM adversarial train


## File structure and description
```
ST-Siamese-Attack

-----------------------------------
train.py                               # Main file: train siamese
classification.py                      # Main file: train classfication
-----------------------------------
siamese_attack.py                      # Main file: attack siamese
classification_attack.py               # Main file: attack classfication
-----------------------------------
fgsm_attack.py                         # ST-iFGSM attack class (linf->l0)
cw_attack.py                           # CW attack class (L2, L0)
cw_attack_utils.py                     # CW attack utils
-----------------------------------
fast_adversarial_train.py              # Main file:
                                         Fast ST-FGSM adversarial train
-----------------------------------
data_generation.ipynb                  # Generate data for siamese
classification_data_generation.ipynb   # Generate data for classification

-----------------------------------
argument.py                            # Argument script
utils.py                               # Utility script
dataset                                # Dataset folder
│   ├── ...
models                                 # Model folder
│   ├── ...
README
```
