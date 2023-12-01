# Federated Learning with HRNet for facial key-points detection

**WIP**

This work is carried out as thesis project for the Master's in Quantum Machine Learning.

## Abstract
The objective of this study is to employ a federated approach in training a Convolutional Neural Network (CNN) for the identification of facial points in individuals afflicted by dysarthria, a neurological disorder leading to anomalies in facial movements. The localization of facial points serve the purpose of automatically monitoring the progression of the disease. The CNN employed is the [HRNet](https://github.com/HRNet/HRNet-Facial-Landmark-Detection), fine-tuned on the [Toronto NeuroFace dataset](https://slp.utoronto.ca/faculty/yana-yunusova/speech-production-lab/datasets/). 

Federated Learning represents a collaborative machine learning approach that facilitates model training across decentralized edge devices or servers. Instead of consolidating data in a central repository, federated learning allows the training of models on individual devices, with only the model updates shared with a central server. In the specific scenario under consideration, the various nodes where the training is distributed emulate distinct healthcare facilities, each housing a subset of the data utilized for neural network training. For this particular application, the decentralized training approach alleviates privacy concerns associated with centralized storage of sensitive data, as raw clinical data remains localized.

The framework selected to develop the project is Flower, an open-source Python library for federated learning, federated evaluation, and federated analytics.

### Install torchvision 0.10.0 and Flower  
```bash
  pip install -r requirements.txt
  ```

### Add python path
 ```bash
  python setup.py develop
  ```

