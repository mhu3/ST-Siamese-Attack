# ST-iFGSM : Enhancing the robustness of the HuMID model

This project TODO

It contains

- CW-attack and FGSM-attack on ST-Siamese and LSTM-based classification networks.
- TODO


## Prerequisites
---
TODO


## Usage
---
TODO


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