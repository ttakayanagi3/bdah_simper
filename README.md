# SimPer: Simple Self-Supervised Learning of Periodic Targets paper reproduction
This repository was created as part of an assignment for Georgia Tech's OMSCS course CSE 6250: Big Data for Health Informatics. For more information, visit: CSE 6250: Big Data for Health Informatics
We attempted to replicate the Original Paper using PyTorch. We will conduct the learning of Periodic features using the SimPer Framework, targeting the Rotating Digits dataset.

Original Paper Link
SimPer: Simple Self-Supervised Learning of Periodic Targets
(https://arxiv.org/abs/2210.03115)

Below is a visualization of the features obtained from our learning, using t-SNE.
![t-sne](https://github.com/ttakayanagi3/bdah_simper/assets/146202307/58d41def-a2c0-4c4f-92ad-0fa1e8f10520)

This is a visualization by umap.
![umap](https://github.com/ttakayanagi3/bdah_simper/assets/146202307/21a3e336-d7cb-4862-8f61-f73b2c573f24)

# environment
To set up the environment, unvironment.yaml can be available using conda command.
```
conda env create -f environment.yaml
```

# How To Train
main.py is a program for training. Make sure that UMAP is equal to False in the training phase.During learning, the process can be monitored using MLFlow. From the perspective of computational load, the use of CUDA is recommended.
Once the training is completed, you can switch UMAP into True. Then, UMAP or t-SNE is visualized.
