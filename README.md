# Federated Learning in CVPR2024

<img src="https://media.icml.cc/Conferences/CVPR2024/img/homepage_image.jpg" style="width:33%;">

#### Fair Federated Learning under Domain Skew with Local Consistency and Domain Diversity

> Yuhang Chen, Wenke Huang, Mang Ye

* Paper: https://arxiv.org/abs/2405.16585
* Code: https://github.com/yuhangchen0/FedHEAL
* <details>
    <summary>Abstract (Click to expand):</summary>
  Federated learning (FL) has emerged as a new paradigm for privacy-preserving collaborative training. Under domain skew, the current FL approaches are biased and face two fairness problems. 1) Parameter Update Conflict: data disparity among clients leads to varying parameter importance and inconsistent update directions. These two disparities cause important parameters to potentially be overwhelmed by unimportant ones of dominant updates. It consequently results in significant performance decreases for lower-performing clients. 2) Model Aggregation Bias: existing FL approaches introduce unfair weight allocation and neglect domain diversity. It leads to biased model convergence objective and distinct performance among domains. We discover a pronounced directional update consistency in Federated Learning and propose a novel framework to tackle above issues. First, leveraging the discovered characteristic, we selectively discard unimportant parameter updates to prevent updates from clients with lower performance overwhelmed by unimportant parameters, resulting in fairer generalization performance. Second, we propose a fair aggregation objective to prevent global model bias towards some domains, ensuring that the global model continuously aligns with an unbiased model. The proposed method is generic and can be combined with other existing FL methods to enhance fairness. Comprehensive experiments on Digits and Office-Caltech demonstrate the high fairness and performance of our method.
  </details>

#### FedAS: Bridging Inconsistency in Personalized Federated Learning

>  Xiyuan Yang, Wenke Huang, Mang Ye

* Paper: 
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>
#### Federated Online Adaptation for Deep Stereo

> Matteo Poggi, Fabio Tosi

* Paper: https://arxiv.org/abs/2405.14873
* Project: https://fedstereo.github.io/
* <details>
  <summary>Abstract:</summary>
  We introduce a novel approach for adapting deep stereo networks in a collaborative manner. By building over principles of federated learning, we develop a distributed framework allowing for demanding the optimization process to a number of clients deployed in different environments. This makes it possible, for a deep stereo network running on resourced-constrained devices, to capitalize on the adaptation process carried out by other instances of the same architecture, and thus improve its accuracy in challenging environments even when it cannot carry out adaptation on its own. Experimental results show how federated adaptation performs equivalently to on-device adaptation, and even better when dealing with challenging environments.
  </details>
#### Rethinking the Representation in Federated Unsupervised Learning with Non-IID Data

> Xinting Liao, Weiming Liu, Chaochao Chen, Pengyang Zhou, Fengyuan Yu, Huabin Zhu, Binhui Yao, Tao Wang, Xiaolin Zheng, Yanchao Tan

* Paper: https://arxiv.org/abs/2403.16398
* Code: 
* <details>
  <summary>Abstract:</summary>
      Federated learning achieves effective performance in modeling decentralized data. In practice, client data are not well-labeled, which makes it potential for federated unsupervised learning (FUSL) with non-IID data. However, the performance of existing FUSL methods suffers from insufficient representations, i.e., (1) representation collapse entanglement among local and global models, and (2) inconsistent representation spaces among local models. The former indicates that representation collapse in local model will subsequently impact the global model and other local models. The latter means that clients model data representation with inconsistent parameters due to the deficiency of supervision signals. In this work, we propose FedU2 which enhances generating uniform and unified representation in FUSL with non-IID data. Specifically, FedU2 consists of flexible uniform regularizer (FUR) and efficient unified aggregator (EUA). FUR in each client avoids representation collapse via dispersing samples uniformly, and EUA in server promotes unified representation by constraining consistent client model updating. To extensively validate the performance of FedU2, we conduct both cross-device and cross-silo evaluation experiments on two benchmark datasets, i.e., CIFAR10 and CIFAR100.
  </details>
#### Data Valuation and Detections in Federated Learning

> Wenqian Li, Shuran Fu, Fengrui Zhang, Yan Pang

* Paper: https://arxiv.org/abs/2311.05304
* Code: https://github.com/muz1lee/MOTdata
* <details>
  <summary>Abstract:</summary>
      Federated Learning (FL) enables collaborative model training while preserving the privacy of raw data. A challenge in this framework is the fair and efficient valuation of data, which is crucial for incentivizing clients to contribute high-quality data in the FL task. In scenarios involving numerous data clients within FL, it is often the case that only a subset of clients and datasets are pertinent to a specific learning task, while others might have either a negative or negligible impact on the model training process. This paper introduces a novel privacy-preserving method for evaluating client contributions and selecting relevant datasets without a pre-specified training algorithm in an FL task. Our proposed approach FedBary, utilizes Wasserstein distance within the federated context, offering a new solution for data valuation in the FL framework. This method ensures transparent data valuation and efficient computation of the Wasserstein barycenter and reduces the dependence on validation datasets. Through extensive empirical experiments and theoretical analyses, we demonstrate the potential of this data valuation method as a promising avenue for FL research.
  </details>
#### FedHCA$^2$: Towards Hetero-Client Federated Multi-Task Learning

> Yuxiang Lu, Suizhi Huang, Yuwen Yang, Shalayiding Sirejiding, Yue Ding, Hongtao Lu

* Paper: https://arxiv.org/abs/2311.13250
* Code: https://github.com/innovator-zero/FedHCA2
* <details>
  <summary>Abstract:</summary>
      Federated Learning (FL) enables joint training across distributed clients using their local data privately. Federated Multi-Task Learning (FMTL) builds on FL to handle multiple tasks, assuming model congruity that identical model architecture is deployed in each client. To relax this assumption and thus extend real-world applicability, we introduce a novel problem setting, Hetero-Client Federated Multi-Task Learning (HC-FMTL), to accommodate diverse task setups. The main challenge of HC-FMTL is the model incongruity issue that invalidates conventional aggregation methods. It also escalates the difficulties in accurate model aggregation to deal with data and task heterogeneity inherent in FMTL. To address these challenges, we propose the FedHCA2 framework, which allows for federated training of personalized models by modeling relationships among heterogeneous clients. Drawing on our theoretical insights into the difference between multi-task and federated optimization, we propose the Hyper Conflict-Averse Aggregation scheme to mitigate conflicts during encoder updates. Additionally, inspired by task interaction in MTL, the Hyper Cross Attention Aggregation scheme uses layer-wise cross attention to enhance decoder interactions while alleviating model incongruity. Moreover, we employ learnable Hyper Aggregation Weights for each client to customize personalized parameter updates. Extensive experiments demonstrate the superior performance of FedHCA2 in various HC-FMTL scenarios compared to representative methods. Our code will be made publicly available.
  </details>
#### Federated Generalized Category Discovery

> Nan Pu, Zhun Zhong, Xinyuan Ji, Nicu Sebe

* Paper: https://arxiv.org/abs/2305.14107
* Code: 
* <details>
  <summary>Abstract:</summary>
      Generalized category discovery (GCD) aims at grouping unlabeled samples from known and unknown classes, given labeled data of known classes. To meet the recent decentralization trend in the community, we introduce a practical yet challenging task, namely Federated GCD (Fed-GCD), where the training data are distributively stored in local clients and cannot be shared among clients. The goal of Fed-GCD is to train a generic GCD model by client collaboration under the privacy-protected constraint. The Fed-GCD leads to two challenges: 1) representation degradation caused by training each client model with fewer data than centralized GCD learning, and 2) highly heterogeneous label spaces across different clients. To this end, we propose a novel Associated Gaussian Contrastive Learning (AGCL) framework based on learnable GMMs, which consists of a Client Semantics Association (CSA) and a global-local GMM Contrastive Learning (GCL). On the server, CSA aggregates the heterogeneous categories of local-client GMMs to generate a global GMM containing more comprehensive category knowledge. On each client, GCL builds class-level contrastive learning with both local and global GMMs. The local GCL learns robust representation with limited local data. The global GCL encourages the model to produce more discriminative representation with the comprehensive category relationships that may not exist in local data. We build a benchmark based on six visual datasets to facilitate the study of Fed-GCD. Extensive experiments show that our AGCL outperforms the FedAvg-based baseline on all datasets.
  </details>  
#### FLHetBench: Benchmarking Device and State Heterogeneity in Federated Learning

> Junyuan Zhang, Shuang Zeng, Miao Zhang, Runxi Wang, Feifei Wang, Yuyin Zhou, Paul Pu Liang, Liangqiong Qu

* Paper: 
* Project: https://carkham.github.io/FL_Het_Bench/
* <details>
  <summary>Abstract:</summary>
      Federated learning (FL) is a powerful technology that enables collaborative training of machine learning models without sharing private data among clients. The fundamental challenge in FL lies in learning over extremely heterogeneous data distributions, device capacities, and device state availabilities, all of which adversely impact performance and communication efficiency. While data heterogeneity has been well-studied in the literature, this paper introduces FLHetBench, the first FL benchmark targeted toward understanding device and state heterogeneity. FLHetBench comprises two new sampling methods to generate real-world device and state databases with varying heterogeneity and new metrics for quantifying the success of FL methods under these real-world constraints. Using FLHetBench, we conduct a comprehensive evaluation of existing methods and find that they struggle under these settings, which inspires us to propose BiasPrompt+, a new method employing staleness-aware aggregation and fast weights to tackle these new heterogeneity challenges. Experiments on various FL tasks and datasets validate the effectiveness of our BiasPrompt+ method and highlight the value of FLHetBench in fostering the development of more efficient and robust FL solutions under real-world device and state constraints.
  </details>  
#### An Upload-Efficient Scheme for Transferring Knowledge From a Server-Side Pre-trained Generator to Clients in Heterogeneous Federated Learning

> Jianqing Zhang, Yang Liu, Yang Hua, Jian Cao

* Paper: https://arxiv.org/abs/2403.15760
* Code: https://github.com/TsingZ0/FedKTL
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts

> Jiayi Chen, Benteng Ma, Hengfei Cui, Yong Xia

* Paper: https://arxiv.org/abs/2312.02567
* Code: https://github.com/JiayiChen815/FEAL
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Revamping Federated Learning Security from a Defender's Perspective: A Unified Defense with Homomorphic Encrypted Data Space

> Naveen Kumar Kummari, Reshmi Mitra, Krishna Mohan Chalavadi

* Paper: 
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### FedSOL: Stabilized Orthogonal Learning with Proximal Restrictions in Federated Learning

> Gihun Lee, Minchan Jeong, Sangmook Kim, Jaehoon Oh, Se-Young Yun

* Paper: https://arxiv.org/abs/2308.12532
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Global and Local Prompts Cooperation via Optimal Transport for Federated Learning

> Hongxia Li, Wei Huang, Jingya Wang, Ye Shi

* Paper: https://arxiv.org/abs/2403.00041
* Code: https://github.com/HongxiaLee/FedOTP
* <details>
  <summary>Abstract:</summary>
  </details>  
#### FedMef: Towards Memory-efficient Federated Dynamic Pruning

> Hong Huang, Weiming Zhuang, Chen Chen, Lingjuan Lyu

* Paper: https://arxiv.org/abs/2403.14737
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Leak and Learn: An Attacker's Cookbook to Train Using Leaked Data from Federated Learning

> Joshua C. Zhao, Ahaan Dabholkar, Atul Sharma, Saurabh Bagchi

* Paper: https://arxiv.org/abs/2403.18144
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Relaxed Contrastive Learning for Federated Learning

> Seonguk Seo, Jinkyu Kim, Geeho Kim, Bohyung Han

* Paper: https://arxiv.org/abs/2401.04928
* Code: https://github.com/skynbe/FedRCL
* <details>
  <summary>Abstract:</summary>
  </details>  
#### FedUV: Uniformity and Variance for Heterogeneous Federated Learning

> Ha Min Son, Moon-Hyun Kim, Tai-Myoung Chung, Chao Huang, Xin Liu

* Paper: https://arxiv.org/abs/2402.18372
* Code: https://github.com/sonhamin/FedUV
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Adaptive Hyper-graph Aggregation for Modality-Agnostic Federated Learning

> Fan Qi, Shuai Li

* Paper: 
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Device-Wise Federated Network Pruning

> Shangqian Gao, Junyi Li, Zeyu Zhang, Yanfu Zhang, Weidong Cai, Heng Huang

* Paper: 
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### An Aggregation-Free Federated Learning for Tackling Data Heterogeneity

> Yuan Wang, Huazhu Fu, Renuga Kanagavelu, Qingsong Wei, Yong Liu, Rick Siow Mong Goh

* Paper: https://arxiv.org/abs/2404.18962
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Decentralized Directed Collaboration for Personalized Federated Learning

> Yingqi Liu, Yifan Shi, Qinglun Li, Baoyuan Wu, Xueqian Wang, Li Shen

* Paper: https://arxiv.org/abs/2405.17876
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### PerAda: Parameter-Efficient Federated Learning Personalization with Generalization Guarantees

> Chulin Xie, De-An Huang, Wenda Chu, Daguang Xu, Chaowei Xiao, Bo Li, Anima Anandkumar

* Paper: https://arxiv.org/abs/2302.06637
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### DiPrompT: Disentangled Prompt Tuning for Multiple Latent Domain Generalization in Federated Learning

> Sikai Bai, Jie Zhang, Shuaicheng Li, Song Guo, Jingcai Guo, Jun Hou, Tao Han, Xiaocheng Lu

* Paper: https://arxiv.org/abs/2403.08506
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Towards Efficient Replay in Federated Incremental Learning

> Yichen Li, Qunwei Li, Haozhao Wang, Ruixuan Li, Wenliang Zhong, Guannan Zhang

* Paper: https://arxiv.org/abs/2403.05890
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Text-Enhanced Data-free Approach for Federated Class-Incremental Learning

> Minh-Tuan Tran, Trung Le, Xuan-May Le, Mehrtash Harandi, Dinh Phung

* Paper: https://arxiv.org/abs/2403.14101
* Code: https://github.com/tmtuan1307/lander
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Communication-Efficient Federated Learning with Accelerated Client Gradient

> Geeho Kim, Jinkyu Kim, Bohyung Han

* Paper: https://arxiv.org/abs/2201.03172
* Code: https://github.com/geehokim/FedACG
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Efficiently Assemble Normalization Layers and Regularization for Federated Domain Generalization

> Khiem Le, Long Ho, Cuong Do, Danh Le-Phuoc, Kok-Seng Wong

* Paper: https://arxiv.org/abs/2403.15605
* Code: https://github.com/lhkhiem28/gPerXAN
* <details>
  <summary>Abstract:</summary>
  </details>  
#### FedSelect: Personalized Federated Learning with Customized Selection of Parameters for Fine-Tuning

> Rishub Tamirisa, Chulin Xie, Wenxuan Bao, Andy Zhou, Ron Arel, Aviv Shamsian

* Paper: 
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Traceable Federated Continual Learning

> Qiang Wang, Bingyan Liu, Yawen Li

* Paper: 
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Byzantine-robust Decentralized Federated Learning via Dual-domain Clustering and Trust Bootstrapping

> Peng Sun, Xinyang Liu, Zhibo Wang, Bo Liu

* Paper: 
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Unlocking the Potential of Prompt-Tuning in Bridging Generalized and Personalized Federated Learning

> Wenlong Deng, Christos Thrampoulidis, Xiaoxiao Li

* Paper: https://arxiv.org/abs/2310.18285
* Code: https://github.com/ubc-tea/SGPT
* <details>
  <summary>Abstract:</summary>
  </details>  
#### Mixed-Precision Quantization for Federated Learning on Resource-Constrained Heterogeneous Devices

> Huancheng Chen, Haris Vikalo

* Paper: https://arxiv.org/abs/2311.18129
* Code: 
* <details>
  <summary>Abstract:</summary>
  </details>
  