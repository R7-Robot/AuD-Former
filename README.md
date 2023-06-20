# AuD-Former
This repository contains the source code for our paper: "Hierarchical Transformer Network for Multimodal Audio-Based Disease Prediction", submitted to IEEE Transactions on Knowledge and Data Engineering. For more details, please refer to [our project website](https://sites.google.com/view/audformer).


## Abstract
Audio-based disease prediction is emerging as a promising supplement to traditional medical diagnosis methods, facilitating early, convenient, and non-invasive disease detection and prevention. Multimodal fusion, which integrates bio-acoustic features from different domains or modalities, has proven to be an effective solution for enhancing diagnosis performance. However, existing multimodal methods are still in the nascent stages: their shallow fusion strategies, which solely focus on either intra-modal or inter-modal fusion, impede the full exploitation of the potential of multimodal bio-acoustic data. More crucially, the limited exploration of latent dependencies within modality-specific and modality-shared spaces curtails their capacity to manage the inherent heterogeneity in multimodal fusion. To fill these gaps, we propose AuD-Former, a hierarchical transformer network designed for general multimodal audio-based disease prediction. Specifically, we seamlessly integrate intra-modal and inter-modal fusion in a hierarchical manner and proficiently encode the necessary intra-modal and inter-modal complementary correlations respectively. Through comprehensive experiments on two distinct diseases, COVID-19 and Parkinson's Disease, we observe an average accuracy of 93.13% on the Coswara dataset and 96.39% on IPVS dataset. Therefore, we prove the strengths of our proposed AuD-Former and the main components within it, showcasing its promising potential in a broad context of audio-based disease prediction tasks.


## Overview Architecture for Audformer
<div align=center>
<img src="/figures/Framework-AuDFormer.png" width="800" />
</div>  

## Illustration of the transformer network and multi-head self-attention

<div align=center>
<img src="/figures/SAAT.png" width="800" />
</div>  

## Usage
### Requirements
1. Install the required Python package and version

- Python 3.8
- [Pytorch (1.8.2+cu111) and torchvision or above](https://pytorch.org/)
- CUDA  11.1 or above
- scikit-learn  1.0.2
- numpy 1.19.5


### Run the code

0. Training command as follow. 
```
python main.py
```

1. Testing command as follow.
```
python main.py --eval
```

(The code was tested in Ubuntu 20.04 with Python 3.8.)
### Change model

#### How change the model



## Acknowledgement

Contributors:  
[Ruiqi Wang](https://github.com/R7-Robot); [Dezhong Zhao](https://github.com/zdz0086); [Jinjin Cai](https://github.com/CJR7).

Part of the code is based on the following repositories: [Husformer](https://github.com/SMARTlab-Purdue/Husformer).




