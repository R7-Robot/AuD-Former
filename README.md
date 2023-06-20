# AuD-Former
This repository contains the source code for our paper: "Hierarchical Transformer Network for Multimodal Audio-Based Disease Prediction", submitted to IEEE Transactions on Knowledge and Data Engineering. For more details, please refer to [our project website](https://sites.google.com/view/audformer).


## Abstract
Audio-based disease prediction is emerging as a promising supplement to traditional medical diagnosis methods, facilitating early, convenient, and non-invasive disease detection and prevention. Multimodal fusion, which integrates bio-acoustic features from different domains or modalities, has proven to be an effective solution for enhancing diagnosis performance. However, existing multimodal methods are still in the nascent stages: their shallow fusion strategies, which solely focus on either intra-modal or inter-modal fusion, impede the full exploitation of the potential of multimodal bio-acoustic data. More crucially, the limited exploration of latent dependencies within modality-specific and modality-shared spaces curtails their capacity to manage the inherent heterogeneity in multimodal fusion. To fill these gaps, we propose AuD-Former, a hierarchical transformer network designed for general multimodal audio-based disease prediction. Specifically, we seamlessly integrate intra-modal and inter-modal fusion in a hierarchical manner and proficiently encode the necessary intra-modal and inter-modal complementary correlations respectively. Through comprehensive experiments on two distinct diseases, COVID-19 and Parkinson's Disease, we observe an average accuracy of 91.13% on the Coswara dataset and 96.39% on IPVS dataset. Therefore, we prove the strengths of our proposed AuD-Former and the main components within it, showcasing its promising potential in a broad context of audio-based disease prediction tasks.


## Overview Architecture for Audformer
<div align=center>
<img src="/figures/Framework-AuDFormer.png" width="800" />
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
We provide 2 models in code to correspond to the 2 data number of datasets as COSWARA and IPVS. You should add or remove some modalities in code, if you want to use more or less modalities by AuD-Former.
#### How change the model
We name these files with their datasets. We put the general python files in [src](src), as the same time, the different datasets main python files are [main-COSWARA.py](main-COSWARA.py), [main-IPVS.py](main-IPVS.py) and different modalities python files in folder [src/COSWARA](src/COSWARA)and [src/IPVS](src/IPVS) , you should rename the main python files as 'main.py' and move the python files from 'src/x' to [src](src/).


## Acknowledgement

Contributors:  
[Dezhong Zhao](https://github.com/zdz0086); [Ruiqi Wang](https://github.com/R7-Robot); [Jinjin Cai](https://github.com/CJR7).





