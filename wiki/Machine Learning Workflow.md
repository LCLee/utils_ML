# Machine Learning Workflow
* 最近一直在做data augmentation、meta learning、transfer learning相关工作，总结下参考过的一些工具
* 在针对数据集测试idea时，秉持DRY原则，提升原型开发workflow的效率
* 比赛套路参考kaggle的解决方案，分解出'标准动作'，完善工具链
* 实际算法验证往往还是单独封装工具效率高，因此应更关注提供的数据嵌入api是否轻量高效
* A. Thakur and A. Krohn-Grimberghe, “AutoCompete: A Framework for Machine Learning Competition,” arXiv:1507.02188 [cs, stat], Jul. 2015.
1. Dataset construction
2. Feature engineering
3. Algorithm Lib
4. Hyper-Parameter
5. Visualization

***
# Dataset construction
* 数据集与学习任务构建
* 参考通用评测数据集与相关paper的实验操作
* 数据采集、清洗、探索，通用工具关注点，格式转化、异常值检测、相关性分析
## ODO
* http://odo.readthedocs.org/
## Dora
* https://github.com/NathanEpstein/Dora
## DataWrangler
* http://vis.stanford.edu/wrangler/
## DataKleenr
* http://chi2innovations.com/datakleenr/
## OpenRefine
* http://openrefine.org/

***

# Feature engineering
* sklearn入门参考
* http://www.cnblogs.com/jasonfreak/p/5448385.html
* 结合具体任务分析
* https://tech.meituan.com/machinelearning-data-feature-process.html
* 参考成熟的api特征探索思路
* https://www.4paradigm.com/support/doc
## Featuretools
* https://www.featuretools.com/
## featureforge
* https://github.com/machinalis/featureforge

***

# Algorithm Lib
* sklearn
* 分支问题参考顶会paper实现,Review的发展索引
## Object Detection
### YOLO
* https://pjreddie.com/darknet/yolo/
### FAIR's research
* https://github.com/facebookresearch/Detectron
## Transfer Learning 
### ML review
* http://transferlearning.xyz/
### DL
* fine-tuning
* Deep Learning of Representations for Unsupervised and Transfer Learning
## GANs
* 比起几何不变性、插值、随机噪声等，GANs提供了探索性数据增强的思路，follow ICLR 2018
* https://medium.com/%40pfferreira
## Ensemble Learning
* boosting bagging stacking
* attention local weighted
* 关注集成学习方法的硬件加速
### StackNet
* https://github.com/kaz-Anova/StackNet
### Xgboost
* https://github.com/dmlc/xgboost
### LightGBM
* https://github.com/Microsoft/LightGBM

***

# Hyper-Parameter
## skopt 
* https://scikit-optimize.github.io/
* Bayesian optimization
* Parallel optimization
* Sklearn gridsearchcv replacement
## advisor
* https://github.com/tobegit3hub/advisor
* Gridsearch
* Random search
* Bayesian optimization
## Hyperopt
* http://hyperopt.github.io/hyperopt/

***

# Visualization
## workshop 
* http://icmlviz.github.io/
## Facets
* https://pair-code.github.io/facets/
## tensorboard-api 
* https://research.googleblog.com/2017/09/build-your-own-machine-learning.html
* creation of new and useful visualizations
1. summary op used to collect data 
2. Python backend to serve custom data
3. dashboard 
## visual DL
* http://visualdl.paddlepaddle.org/

***

# reference 
## Paper整理中