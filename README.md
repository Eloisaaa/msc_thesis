# MSc Theis: Cross-domain Recommendation System and Bayesian Optimization
This is the Github page for 2022 UCL MSc Data Science and Machine Learning Thesis.

## Table of Contents

* [Project Overview](#project_overview)
* [How to use this repository](#technologies)




## Project Overview

In recent years, Cross-domain Recommendation (CDR) has been used to address the long-standing data sparsity problem in recommender systems (RecSys) by leveraging comparatively richer information from a source domain to improve recommendation performance on a target domain with sparser information. Recent advances in deep neural networks have driven recommendation system research. However, most of the advanced models that work really well on single-domain recommendation, such as Wide&Deep learning models DCN, DeepFM, and xDeepFM, and Multi-task Learning (MTL) models, have never been applied on cross-domain recommendation. This dissertation examines the role of the above advanced single-domain recommendation models on cross-domain recommendation, design a new method to adapt MTL based models on Cross-domain Learning (CDL). Furthermore, this thesis investigates how hyper-parameter optimization can be applied to the cross-domain setting, followed by an analysis of Bayesian optimization (BO) via an empirical and/or theoretical analysis of both the Gaussian Process and the bandit procedure used.

We conducted several experiments on a benchmark multi-domain dataset, and evaluated the above models on different ratios of target domain for training. The results show that our MTL based CDL models can significantly improve the performance of CTR prediction on the target domain compared with state-of-the-art single-domain models. Furthermore, we show in this paper that BO can efficiently optimise the CDL model's hyperparameter settings, with a convergence rate twice as fast as GS. Our most important contributions are not only the design of new methods and hyperparameter optimization for CDL, which outperform current state-of-the-art approaches, but we also aid with the data sparsity and cold start problems by training with less target domain data.

## How to use this repository

All of the data is stored in the **data** folder, and the data preprocessing part is in in data_pre.py.

Bayesian Optimizaition empirical analysis:
  1. Run Multi_arm_bandit.ipynb: Multii armed bandits experiments 
  2. Run hyperparameter_optimization_analysis.ipynb: Grid Search VS Bayesian Optimization on a synthetic data and visualization

Baseline:
Run Baseline.ipynb: includes all the baseline experiment
  1. FM, DCN, DeepFM, xDeepFM on single-domain
  2. FM, DCN, DeepFM, xDeepFM on multi-domain
  

MTL based CDR:
Run cross_domain_learning.py for the Shared-Bottom based CDL, and MMoE based CDL models, all of the code for shared_bottom and MMoE based CDL models are in models_mdl folder
  1. `$ pip install -U deepctr-torch`
  2. Move folder models_mdl to the downloaded deepctr_torch packages folder from step 1 <br/>
     `mv models_mdl ../deepctr_torch`
  4. run cross_domain_learning.py

Bayesian Optimization:
Run bays_opt.py for Bayesian Optimization experiments
  1. xDeepFM on multi-domain
  2. shared-bottom based CDL on multi-domain
  3. MMoE based CDL on multi-domain
  



