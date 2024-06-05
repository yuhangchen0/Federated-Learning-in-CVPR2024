# Federated Learning in CVPR2024

<img src="https://media.icml.cc/Conferences/CVPR2024/img/homepage_image.jpg" style="width:33%;">

Hi there! I'm Yuhang, and I'm thrilled to announce that **I will be attending CVPR 2024 and giving a poster presentation** on the morning of 20th June. [Click here to see the CVPR2024 FL Poster Schedule](#cvpr2024-fl-poster-schedule)

The first paper on the following list is the one I'll be presenting. I am really looking forward to discussing our work and exchanging ideas with all of you! 

**Feel free to reach out to me via email or WeChat** if you'd like to connect before or during the conference. I can't wait to meet you and explore exciting collaborations! 

Email: yhchen0@whu.edu.cn

WeChat: 

<img src=".assets/wechat-qrcode.jpg" style="width:20%;">

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
  <summary>Abstract (Click to expand):</summary>
  </details>
#### Federated Online Adaptation for Deep Stereo

> Matteo Poggi, Fabio Tosi

* Paper: https://arxiv.org/abs/2405.14873
* Project: https://fedstereo.github.io/
* <details>
  <summary>Abstract (Click to expand):</summary>
  We introduce a novel approach for adapting deep stereo networks in a collaborative manner. By building over principles of federated learning, we develop a distributed framework allowing for demanding the optimization process to a number of clients deployed in different environments. This makes it possible, for a deep stereo network running on resourced-constrained devices, to capitalize on the adaptation process carried out by other instances of the same architecture, and thus improve its accuracy in challenging environments even when it cannot carry out adaptation on its own. Experimental results show how federated adaptation performs equivalently to on-device adaptation, and even better when dealing with challenging environments.
  </details>
#### Rethinking the Representation in Federated Unsupervised Learning with Non-IID Data

> Xinting Liao, Weiming Liu, Chaochao Chen, Pengyang Zhou, Fengyuan Yu, Huabin Zhu, Binhui Yao, Tao Wang, Xiaolin Zheng, Yanchao Tan

* Paper: https://arxiv.org/abs/2403.16398
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
      Federated learning achieves effective performance in modeling decentralized data. In practice, client data are not well-labeled, which makes it potential for federated unsupervised learning (FUSL) with non-IID data. However, the performance of existing FUSL methods suffers from insufficient representations, i.e., (1) representation collapse entanglement among local and global models, and (2) inconsistent representation spaces among local models. The former indicates that representation collapse in local model will subsequently impact the global model and other local models. The latter means that clients model data representation with inconsistent parameters due to the deficiency of supervision signals. In this work, we propose FedU2 which enhances generating uniform and unified representation in FUSL with non-IID data. Specifically, FedU2 consists of flexible uniform regularizer (FUR) and efficient unified aggregator (EUA). FUR in each client avoids representation collapse via dispersing samples uniformly, and EUA in server promotes unified representation by constraining consistent client model updating. To extensively validate the performance of FedU2, we conduct both cross-device and cross-silo evaluation experiments on two benchmark datasets, i.e., CIFAR10 and CIFAR100.
  </details>
#### Data Valuation and Detections in Federated Learning

> Wenqian Li, Shuran Fu, Fengrui Zhang, Yan Pang

* Paper: https://arxiv.org/abs/2311.05304
* Code: https://github.com/muz1lee/MOTdata
* <details>
  <summary>Abstract (Click to expand):</summary>
      Federated Learning (FL) enables collaborative model training while preserving the privacy of raw data. A challenge in this framework is the fair and efficient valuation of data, which is crucial for incentivizing clients to contribute high-quality data in the FL task. In scenarios involving numerous data clients within FL, it is often the case that only a subset of clients and datasets are pertinent to a specific learning task, while others might have either a negative or negligible impact on the model training process. This paper introduces a novel privacy-preserving method for evaluating client contributions and selecting relevant datasets without a pre-specified training algorithm in an FL task. Our proposed approach FedBary, utilizes Wasserstein distance within the federated context, offering a new solution for data valuation in the FL framework. This method ensures transparent data valuation and efficient computation of the Wasserstein barycenter and reduces the dependence on validation datasets. Through extensive empirical experiments and theoretical analyses, we demonstrate the potential of this data valuation method as a promising avenue for FL research.
  </details>
#### FedHCA$^2$: Towards Hetero-Client Federated Multi-Task Learning

> Yuxiang Lu, Suizhi Huang, Yuwen Yang, Shalayiding Sirejiding, Yue Ding, Hongtao Lu

* Paper: https://arxiv.org/abs/2311.13250
* Code: https://github.com/innovator-zero/FedHCA2
* <details>
  <summary>Abstract (Click to expand):</summary>
      Federated Learning (FL) enables joint training across distributed clients using their local data privately. Federated Multi-Task Learning (FMTL) builds on FL to handle multiple tasks, assuming model congruity that identical model architecture is deployed in each client. To relax this assumption and thus extend real-world applicability, we introduce a novel problem setting, Hetero-Client Federated Multi-Task Learning (HC-FMTL), to accommodate diverse task setups. The main challenge of HC-FMTL is the model incongruity issue that invalidates conventional aggregation methods. It also escalates the difficulties in accurate model aggregation to deal with data and task heterogeneity inherent in FMTL. To address these challenges, we propose the FedHCA2 framework, which allows for federated training of personalized models by modeling relationships among heterogeneous clients. Drawing on our theoretical insights into the difference between multi-task and federated optimization, we propose the Hyper Conflict-Averse Aggregation scheme to mitigate conflicts during encoder updates. Additionally, inspired by task interaction in MTL, the Hyper Cross Attention Aggregation scheme uses layer-wise cross attention to enhance decoder interactions while alleviating model incongruity. Moreover, we employ learnable Hyper Aggregation Weights for each client to customize personalized parameter updates. Extensive experiments demonstrate the superior performance of FedHCA2 in various HC-FMTL scenarios compared to representative methods. Our code will be made publicly available.
  </details>
#### Federated Generalized Category Discovery

> Nan Pu, Zhun Zhong, Xinyuan Ji, Nicu Sebe

* Paper: https://arxiv.org/abs/2305.14107
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
      Generalized category discovery (GCD) aims at grouping unlabeled samples from known and unknown classes, given labeled data of known classes. To meet the recent decentralization trend in the community, we introduce a practical yet challenging task, namely Federated GCD (Fed-GCD), where the training data are distributively stored in local clients and cannot be shared among clients. The goal of Fed-GCD is to train a generic GCD model by client collaboration under the privacy-protected constraint. The Fed-GCD leads to two challenges: 1) representation degradation caused by training each client model with fewer data than centralized GCD learning, and 2) highly heterogeneous label spaces across different clients. To this end, we propose a novel Associated Gaussian Contrastive Learning (AGCL) framework based on learnable GMMs, which consists of a Client Semantics Association (CSA) and a global-local GMM Contrastive Learning (GCL). On the server, CSA aggregates the heterogeneous categories of local-client GMMs to generate a global GMM containing more comprehensive category knowledge. On each client, GCL builds class-level contrastive learning with both local and global GMMs. The local GCL learns robust representation with limited local data. The global GCL encourages the model to produce more discriminative representation with the comprehensive category relationships that may not exist in local data. We build a benchmark based on six visual datasets to facilitate the study of Fed-GCD. Extensive experiments show that our AGCL outperforms the FedAvg-based baseline on all datasets.
  </details>  
#### FLHetBench: Benchmarking Device and State Heterogeneity in Federated Learning

> Junyuan Zhang, Shuang Zeng, Miao Zhang, Runxi Wang, Feifei Wang, Yuyin Zhou, Paul Pu Liang, Liangqiong Qu

* Paper: 
* Project: https://carkham.github.io/FL_Het_Bench/
* <details>
  <summary>Abstract (Click to expand):</summary>
      Federated learning (FL) is a powerful technology that enables collaborative training of machine learning models without sharing private data among clients. The fundamental challenge in FL lies in learning over extremely heterogeneous data distributions, device capacities, and device state availabilities, all of which adversely impact performance and communication efficiency. While data heterogeneity has been well-studied in the literature, this paper introduces FLHetBench, the first FL benchmark targeted toward understanding device and state heterogeneity. FLHetBench comprises two new sampling methods to generate real-world device and state databases with varying heterogeneity and new metrics for quantifying the success of FL methods under these real-world constraints. Using FLHetBench, we conduct a comprehensive evaluation of existing methods and find that they struggle under these settings, which inspires us to propose BiasPrompt+, a new method employing staleness-aware aggregation and fast weights to tackle these new heterogeneity challenges. Experiments on various FL tasks and datasets validate the effectiveness of our BiasPrompt+ method and highlight the value of FLHetBench in fostering the development of more efficient and robust FL solutions under real-world device and state constraints.
  </details>  
#### An Upload-Efficient Scheme for Transferring Knowledge From a Server-Side Pre-trained Generator to Clients in Heterogeneous Federated Learning

> Jianqing Zhang, Yang Liu, Yang Hua, Jian Cao

* Paper: https://arxiv.org/abs/2403.15760
* Code: https://github.com/TsingZ0/FedKTL
* <details>
  <summary>Abstract (Click to expand):</summary>
      Heterogeneous Federated Learning (HtFL) enables collaborative learning on multiple clients with different model architectures while preserving privacy. Despite recent research progress, knowledge sharing in HtFL is still difficult due to data and model heterogeneity. To tackle this issue, we leverage the knowledge stored in pre-trained generators and propose a new upload-efficient knowledge transfer scheme called Federated Knowledge-Transfer Loop (FedKTL). Our FedKTL can produce client-task-related prototypical image-vector pairs via the generator's inference on the server. With these pairs, each client can transfer pre-existing knowledge from the generator to its local model through an additional supervised local task. We conduct extensive experiments on four datasets under two types of data heterogeneity with 14 kinds of models including CNNs and ViTs. Results show that our upload-efficient FedKTL surpasses seven state-of-the-art methods by up to 7.31% in accuracy. Moreover, our knowledge transfer scheme is applicable in scenarios with only one edge client. Code: this https URL
  </details>  
#### Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts

> Jiayi Chen, Benteng Ma, Hengfei Cui, Yong Xia

* Paper: https://arxiv.org/abs/2312.02567
* Code: https://github.com/JiayiChen815/FEAL
* <details>
  <summary>Abstract (Click to expand):</summary>
      Federated learning facilitates the collaborative learning of a global model across multiple distributed medical institutions without centralizing data. Nevertheless, the expensive cost of annotation on local clients remains an obstacle to effectively utilizing local data. To mitigate this issue, federated active learning methods suggest leveraging local and global model predictions to select a relatively small amount of informative local data for annotation. However, existing methods mainly focus on all local data sampled from the same domain, making them unreliable in realistic medical scenarios with domain shifts among different clients. In this paper, we make the first attempt to assess the informativeness of local data derived from diverse domains and propose a novel methodology termed Federated Evidential Active Learning (FEAL) to calibrate the data evaluation under domain shift. Specifically, we introduce a Dirichlet prior distribution in both local and global models to treat the prediction as a distribution over the probability simplex and capture both aleatoric and epistemic uncertainties by using the Dirichlet-based evidential model. Then we employ the epistemic uncertainty to calibrate the aleatoric uncertainty. Afterward, we design a diversity relaxation strategy to reduce data redundancy and maintain data diversity. Extensive experiments and analysis on five real multi-center medical image datasets demonstrate the superiority of FEAL over the state-of-the-art active learning methods in federated scenarios with domain shifts. The code will be available at this https URL.
  </details>  
#### Revamping Federated Learning Security from a Defender's Perspective: A Unified Defense with Homomorphic Encrypted Data Space

> Naveen Kumar Kummari, Reshmi Mitra, Krishna Mohan Chalavadi

* Paper: 
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
  </details>  
#### FedSOL: Stabilized Orthogonal Learning with Proximal Restrictions in Federated Learning

> Gihun Lee, Minchan Jeong, Sangmook Kim, Jaehoon Oh, Se-Young Yun

* Paper: https://arxiv.org/abs/2308.12532
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
      Federated Learning (FL) aggregates locally trained models from individual clients to construct a global model. While FL enables learning a model with data privacy, it often suffers from significant performance degradation when clients have heterogeneous data distributions. This data heterogeneity causes the model to forget the global knowledge acquired from previously sampled clients after being trained on local datasets. Although the introduction of proximal objectives in local updates helps to preserve global knowledge, it can also hinder local learning by interfering with local objectives. To address this problem, we propose a novel method, Federated Stabilized Orthogonal Learning (FedSOL), which adopts an orthogonal learning strategy to balance the two conflicting objectives. FedSOL is designed to identify gradients of local objectives that are inherently orthogonal to directions affecting the proximal objective. Specifically, FedSOL targets parameter regions where learning on the local objective is minimally influenced by proximal weight perturbations. Our experiments demonstrate that FedSOL consistently achieves state-of-the-art performance across various scenarios.
  </details>  
#### Global and Local Prompts Cooperation via Optimal Transport for Federated Learning

> Hongxia Li, Wei Huang, Jingya Wang, Ye Shi

* Paper: https://arxiv.org/abs/2403.00041
* Code: https://github.com/HongxiaLee/FedOTP
* <details>
  <summary>Abstract (Click to expand):</summary>
      Prompt learning in pretrained visual-language models has shown remarkable flexibility across various downstream tasks. Leveraging its inherent lightweight nature, recent research attempted to integrate the powerful pretrained models into federated learning frameworks to simultaneously reduce communication costs and promote local training on insufficient data. Despite these efforts, current federated prompt learning methods lack specialized designs to systematically address severe data heterogeneities, e.g., data distribution with both label and feature shifts involved. To address this challenge, we present Federated Prompts Cooperation via Optimal Transport (FedOTP), which introduces efficient collaborative prompt learning strategies to capture diverse category traits on a per-client basis. Specifically, for each client, we learn a global prompt to extract consensus knowledge among clients, and a local prompt to capture client-specific category characteristics. Unbalanced Optimal Transport is then employed to align local visual features with these prompts, striking a balance between global consensus and local personalization. By relaxing one of the equality constraints, FedOTP enables prompts to focus solely on the core regions of image patches. Extensive experiments on datasets with various types of heterogeneities have demonstrated that our FedOTP outperforms the state-of-the-art methods.
  </details>  
#### FedMef: Towards Memory-efficient Federated Dynamic Pruning

> Hong Huang, Weiming Zhuang, Chen Chen, Lingjuan Lyu

* Paper: https://arxiv.org/abs/2403.14737
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
      Federated learning (FL) promotes decentralized training while prioritizing data confidentiality. However, its application on resource-constrained devices is challenging due to the high demand for computation and memory resources to train deep learning models. Neural network pruning techniques, such as dynamic pruning, could enhance model efficiency, but directly adopting them in FL still poses substantial challenges, including post-pruning performance degradation, high activation memory usage, etc. To address these challenges, we propose FedMef, a novel and memory-efficient federated dynamic pruning framework. FedMef comprises two key components. First, we introduce the budget-aware extrusion that maintains pruning efficiency while preserving post-pruning performance by salvaging crucial information from parameters marked for pruning within a given budget. Second, we propose scaled activation pruning to effectively reduce activation memory footprints, which is particularly beneficial for deploying FL to memory-limited devices. Extensive experiments demonstrate the effectiveness of our proposed FedMef. In particular, it achieves a significant reduction of 28.5% in memory footprint compared to state-of-the-art methods while obtaining superior accuracy.
  </details>  
#### Leak and Learn: An Attacker's Cookbook to Train Using Leaked Data from Federated Learning

> Joshua C. Zhao, Ahaan Dabholkar, Atul Sharma, Saurabh Bagchi

* Paper: https://arxiv.org/abs/2403.18144
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
      Federated learning is a decentralized learning paradigm introduced to preserve privacy of client data. Despite this, prior work has shown that an attacker at the server can still reconstruct the private training data using only the client updates. These attacks are known as data reconstruction attacks and fall into two major categories: gradient inversion (GI) and linear layer leakage attacks (LLL). However, despite demonstrating the effectiveness of these attacks in breaching privacy, prior work has not investigated the usefulness of the reconstructed data for downstream tasks. In this work, we explore data reconstruction attacks through the lens of training and improving models with leaked data. We demonstrate the effectiveness of both GI and LLL attacks in maliciously training models using the leaked data more accurately than a benign federated learning strategy. Counter-intuitively, this bump in training quality can occur despite limited reconstruction quality or a small total number of leaked images. Finally, we show the limitations of these attacks for downstream training, individually for GI attacks and for LLL attacks.
  </details>  
#### Relaxed Contrastive Learning for Federated Learning

> Seonguk Seo, Jinkyu Kim, Geeho Kim, Bohyung Han

* Paper: https://arxiv.org/abs/2401.04928
* Code: https://github.com/skynbe/FedRCL
* <details>
  <summary>Abstract (Click to expand):</summary>
      We propose a novel contrastive learning framework to effectively address the challenges of data heterogeneity in federated learning. We first analyze the inconsistency of gradient updates across clients during local training and establish its dependence on the distribution of feature representations, leading to the derivation of the supervised contrastive learning (SCL) objective to mitigate local deviations. In addition, we show that a naïve adoption of SCL in federated learning leads to representation collapse, resulting in slow convergence and limited performance gains. To address this issue, we introduce a relaxed contrastive learning loss that imposes a divergence penalty on excessively similar sample pairs within each class. This strategy prevents collapsed representations and enhances feature transferability, facilitating collaborative training and leading to significant performance improvements. Our framework outperforms all existing federated learning approaches by huge margins on the standard benchmarks through extensive experimental results.
  </details>  
#### FedUV: Uniformity and Variance for Heterogeneous Federated Learning

> Ha Min Son, Moon-Hyun Kim, Tai-Myoung Chung, Chao Huang, Xin Liu

* Paper: https://arxiv.org/abs/2402.18372
* Code: https://github.com/sonhamin/FedUV
* <details>
  <summary>Abstract (Click to expand):</summary>
      Federated learning is a promising framework to train neural networks with widely distributed data. However, performance degrades heavily with heterogeneously distributed data. Recent work has shown this is due to the final layer of the network being most prone to local bias, some finding success freezing the final layer as an orthogonal classifier. We investigate the training dynamics of the classifier by applying SVD to the weights motivated by the observation that freezing weights results in constant singular values. We find that there are differences when training in IID and non-IID settings. Based on this finding, we introduce two regularization terms for local training to continuously emulate IID settings: (1) variance in the dimension-wise probability distribution of the classifier and (2) hyperspherical uniformity of representations of the encoder. These regularizations promote local models to act as if it were in an IID setting regardless of the local data distribution, thus offsetting proneness to bias while being flexible to the data. On extensive experiments in both label-shift and feature-shift settings, we verify that our method achieves highest performance by a large margin especially in highly non-IID cases in addition to being scalable to larger models and datasets.
  </details>  
#### Adaptive Hyper-graph Aggregation for Modality-Agnostic Federated Learning

> Fan Qi, Shuai Li

* Paper: 
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
  </details>  
#### Device-Wise Federated Network Pruning

> Shangqian Gao, Junyi Li, Zeyu Zhang, Yanfu Zhang, Weidong Cai, Heng Huang

* Paper: 
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
  </details>  
#### An Aggregation-Free Federated Learning for Tackling Data Heterogeneity

> Yuan Wang, Huazhu Fu, Renuga Kanagavelu, Qingsong Wei, Yong Liu, Rick Siow Mong Goh

* Paper: https://arxiv.org/abs/2404.18962
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
      The performance of Federated Learning (FL) hinges on the effectiveness of utilizing knowledge from distributed datasets. Traditional FL methods adopt an aggregate-then-adapt framework, where clients update local models based on a global model aggregated by the server from the previous training round. This process can cause client drift, especially with significant cross-client data heterogeneity, impacting model performance and convergence of the FL algorithm. To address these challenges, we introduce FedAF, a novel aggregation-free FL algorithm. In this framework, clients collaboratively learn condensed data by leveraging peer knowledge, the server subsequently trains the global model using the condensed data and soft labels received from the clients. FedAF inherently avoids the issue of client drift, enhances the quality of condensed data amid notable data heterogeneity, and improves the global model performance. Extensive numerical studies on several popular benchmark datasets show FedAF surpasses various state-of-the-art FL algorithms in handling label-skew and feature-skew data heterogeneity, leading to superior global model accuracy and faster convergence.
  </details>  
#### Decentralized Directed Collaboration for Personalized Federated Learning

> Yingqi Liu, Yifan Shi, Qinglun Li, Baoyuan Wu, Xueqian Wang, Li Shen

* Paper: https://arxiv.org/abs/2405.17876
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
      Personalized Federated Learning (PFL) is proposed to find the greatest personalized models for each client. To avoid the central failure and communication bottleneck in the server-based FL, we concentrate on the Decentralized Personalized Federated Learning (DPFL) that performs distributed model training in a Peer-to-Peer (P2P) manner. Most personalized works in DPFL are based on undirected and symmetric topologies, however, the data, computation and communication resources heterogeneity result in large variances in the personalized models, which lead the undirected aggregation to suboptimal personalized performance and unguaranteed convergence. To address these issues, we propose a directed collaboration DPFL framework by incorporating stochastic gradient push and partial model personalized, called \textbf{D}ecentralized \textbf{Fed}erated \textbf{P}artial \textbf{G}radient \textbf{P}ush (\textbf{DFedPGP}). It personalizes the linear classifier in the modern deep model to customize the local solution and learns a consensus representation in a fully decentralized manner. Clients only share gradients with a subset of neighbors based on the directed and asymmetric topologies, which guarantees flexible choices for resource efficiency and better convergence. Theoretically, we show that the proposed DFedPGP achieves a superior convergence rate of O(1/√T) in the general non-convex setting, and prove the tighter connectivity among clients will speed up the convergence. The proposed method achieves state-of-the-art (SOTA) accuracy in both data and computation heterogeneity scenarios, demonstrating the efficiency of the directed collaboration and partial gradient push.
  </details>  
#### PerAda: Parameter-Efficient Federated Learning Personalization with Generalization Guarantees

> Chulin Xie, De-An Huang, Wenda Chu, Daguang Xu, Chaowei Xiao, Bo Li, Anima Anandkumar

* Paper: https://arxiv.org/abs/2302.06637
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
      Personalized Federated Learning (pFL) has emerged as a promising solution to tackle data heterogeneity across clients in FL. However, existing pFL methods either (1) introduce high communication and computation costs or (2) overfit to local data, which can be limited in scope, and are vulnerable to evolved test samples with natural shifts. In this paper, we propose PerAda, a parameter-efficient pFL framework that reduces communication and computational costs and exhibits superior generalization performance, especially under test-time distribution shifts. PerAda reduces the costs by leveraging the power of pretrained models and only updates and communicates a small number of additional parameters from adapters. PerAda has good generalization since it regularizes each client's personalized adapter with a global adapter, while the global adapter uses knowledge distillation to aggregate generalized information from all clients. Theoretically, we provide generalization bounds to explain why PerAda improves generalization, and we prove its convergence to stationary points under non-convex settings. Empirically, PerAda demonstrates competitive personalized performance (+4.85% on CheXpert) and enables better out-of-distribution generalization (+5.23% on CIFAR-10-C) on different datasets across natural and medical domains compared with baselines, while only updating 12.6% of parameters per model based on the adapter. Our code is available at this https URL.
  </details>  
#### DiPrompT: Disentangled Prompt Tuning for Multiple Latent Domain Generalization in Federated Learning

> Sikai Bai, Jie Zhang, Shuaicheng Li, Song Guo, Jingcai Guo, Jun Hou, Tao Han, Xiaocheng Lu

* Paper: https://arxiv.org/abs/2403.08506
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
      Federated learning (FL) has emerged as a powerful paradigm for learning from decentralized data, and federated domain generalization further considers the test dataset (target domain) is absent from the decentralized training data (source domains). However, most existing FL methods assume that domain labels are provided during training, and their evaluation imposes explicit constraints on the number of domains, which must strictly match the number of clients. Because of the underutilization of numerous edge devices and additional cross-client domain annotations in the real world, such restrictions may be impractical and involve potential privacy leaks. In this paper, we propose an efficient and novel approach, called Disentangled Prompt Tuning (DiPrompT), a method that tackles the above restrictions by learning adaptive prompts for domain generalization in a distributed manner. Specifically, we first design two types of prompts, i.e., global prompt to capture general knowledge across all clients and domain prompts to capture domain-specific knowledge. They eliminate the restriction on the one-to-one mapping between source domains and local clients. Furthermore, a dynamic query metric is introduced to automatically search the suitable domain label for each sample, which includes two-substep text-image alignments based on prompt tuning without labor-intensive annotation. Extensive experiments on multiple datasets demonstrate that our DiPrompT achieves superior domain generalization performance over state-of-the-art FL methods when domain labels are not provided, and even outperforms many centralized learning methods using domain labels.
  </details>  
#### Towards Efficient Replay in Federated Incremental Learning

> Yichen Li, Qunwei Li, Haozhao Wang, Ruixuan Li, Wenliang Zhong, Guannan Zhang

* Paper: https://arxiv.org/abs/2403.05890
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
     In Federated Learning (FL), the data in each client is typically assumed fixed or static. However, data often comes in an incremental manner in real-world applications, where the data domain may increase dynamically. In this work, we study catastrophic forgetting with data heterogeneity in Federated Incremental Learning (FIL) scenarios where edge clients may lack enough storage space to retain full data. We propose to employ a simple, generic framework for FIL named Re-Fed, which can coordinate each client to cache important samples for replay. More specifically, when a new task arrives, each client first caches selected previous samples based on their global and local importance. Then, the client trains the local model with both the cached samples and the samples from the new task. Theoretically, we analyze the ability of Re-Fed to discover important samples for replay thus alleviating the catastrophic forgetting problem. Moreover, we empirically show that Re-Fed achieves competitive performance compared to state-of-the-art methods. 
  </details>  
#### Text-Enhanced Data-free Approach for Federated Class-Incremental Learning

> Minh-Tuan Tran, Trung Le, Xuan-May Le, Mehrtash Harandi, Dinh Phung

* Paper: https://arxiv.org/abs/2403.14101
* Code: https://github.com/tmtuan1307/lander
* <details>
  <summary>Abstract (Click to expand):</summary>
      Federated Class-Incremental Learning (FCIL) is an underexplored yet pivotal issue, involving the dynamic addition of new classes in the context of federated learning. In this field, Data-Free Knowledge Transfer (DFKT) plays a crucial role in addressing catastrophic forgetting and data privacy problems. However, prior approaches lack the crucial synergy between DFKT and the model training phases, causing DFKT to encounter difficulties in generating high-quality data from a non-anchored latent space of the old task model. In this paper, we introduce LANDER (Label Text Centered Data-Free Knowledge Transfer) to address this issue by utilizing label text embeddings (LTE) produced by pretrained language models. Specifically, during the model training phase, our approach treats LTE as anchor points and constrains the feature embeddings of corresponding training samples around them, enriching the surrounding area with more meaningful information. In the DFKT phase, by using these LTE anchors, LANDER can synthesize more meaningful samples, thereby effectively addressing the forgetting problem. Additionally, instead of tightly constraining embeddings toward the anchor, the Bounding Loss is introduced to encourage sample embeddings to remain flexible within a defined radius. This approach preserves the natural differences in sample embeddings and mitigates the embedding overlap caused by heterogeneous federated settings. Extensive experiments conducted on CIFAR100, Tiny-ImageNet, and ImageNet demonstrate that LANDER significantly outperforms previous methods and achieves state-of-the-art performance in FCIL. The code is available at this https URL.
  </details>  
#### Communication-Efficient Federated Learning with Accelerated Client Gradient

> Geeho Kim, Jinkyu Kim, Bohyung Han

* Paper: https://arxiv.org/abs/2201.03172
* Code: https://github.com/geehokim/FedACG
* <details>
  <summary>Abstract (Click to expand):</summary>
      Federated learning often suffers from slow and unstable convergence due to the heterogeneous characteristics of participating client datasets. Such a tendency is aggravated when the client participation ratio is low since the information collected from the clients has large variations. To address this challenge, we propose a simple but effective federated learning framework, which improves the consistency across clients and facilitates the convergence of the server model. This is achieved by making the server broadcast a global model with a lookahead gradient. This strategy enables the proposed approach to convey the projected global update information to participants effectively without additional client memory and extra communication costs. We also regularize local updates by aligning each client with the overshot global model to reduce bias and improve the stability of our algorithm. We provide the theoretical convergence rate of our algorithm and demonstrate remarkable performance gains in terms of accuracy and communication efficiency compared to the state-of-the-art methods, especially with low client participation rates. The source code is available at our project page.
  </details>  
#### Efficiently Assemble Normalization Layers and Regularization for Federated Domain Generalization

> Khiem Le, Long Ho, Cuong Do, Danh Le-Phuoc, Kok-Seng Wong

* Paper: https://arxiv.org/abs/2403.15605
* Code: https://github.com/lhkhiem28/gPerXAN
* <details>
  <summary>Abstract (Click to expand):</summary>
      Domain shift is a formidable issue in Machine Learning that causes a model to suffer from performance degradation when tested on unseen domains. Federated Domain Generalization (FedDG) attempts to train a global model using collaborative clients in a privacy-preserving manner that can generalize well to unseen clients possibly with domain shift. However, most existing FedDG methods either cause additional privacy risks of data leakage or induce significant costs in client communication and computation, which are major concerns in the Federated Learning paradigm. To circumvent these challenges, here we introduce a novel architectural method for FedDG, namely gPerXAN, which relies on a normalization scheme working with a guiding regularizer. In particular, we carefully design Personalized eXplicitly Assembled Normalization to enforce client models selectively filtering domain-specific features that are biased towards local data while retaining discrimination of those features. Then, we incorporate a simple yet effective regularizer to guide these models in directly capturing domain-invariant representations that the global model's classifier can leverage. Extensive experimental results on two benchmark datasets, i.e., PACS and Office-Home, and a real-world medical dataset, Camelyon17, indicate that our proposed method outperforms other existing methods in addressing this particular problem.
  </details>  
#### FedSelect: Personalized Federated Learning with Customized Selection of Parameters for Fine-Tuning

> Rishub Tamirisa, Chulin Xie, Wenxuan Bao, Andy Zhou, Ron Arel, Aviv Shamsian

* Paper: 
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
  </details>  
#### Traceable Federated Continual Learning

> Qiang Wang, Bingyan Liu, Yawen Li

* Paper: 
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
  </details>  
#### Byzantine-robust Decentralized Federated Learning via Dual-domain Clustering and Trust Bootstrapping

> Peng Sun, Xinyang Liu, Zhibo Wang, Bo Liu

* Paper: 
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
  </details>  
#### Unlocking the Potential of Prompt-Tuning in Bridging Generalized and Personalized Federated Learning

> Wenlong Deng, Christos Thrampoulidis, Xiaoxiao Li

* Paper: https://arxiv.org/abs/2310.18285
* Code: https://github.com/ubc-tea/SGPT
* <details>
  <summary>Abstract (Click to expand):</summary>
      Vision Transformers (ViT) and Visual Prompt Tuning (VPT) achieve state-of-the-art performance with improved efficiency in various computer vision tasks. This suggests a promising paradigm shift of adapting pre-trained ViT models to Federated Learning (FL) settings. However, the challenge of data heterogeneity among FL clients presents a significant hurdle in effectively deploying ViT models. Existing Generalized FL (GFL) and Personalized FL (PFL) methods have limitations in balancing performance across both global and local data distributions. In this paper, we present a novel algorithm, SGPT, that integrates GFL and PFL approaches by employing a unique combination of both shared and group-specific prompts. This design enables SGPT to capture both common and group-specific features. A key feature of SGPT is its prompt selection module, which facilitates the training of a single global model capable of automatically adapting to diverse local client data distributions without the need for local fine-tuning. To effectively train the prompts, we utilize block coordinate descent (BCD), learning from common feature information (shared prompts), and then more specialized knowledge (group prompts) iteratively. Theoretically, we justify that learning the proposed prompts can reduce the gap between global and local performance. Empirically, we conduct experiments on both label and feature heterogeneity settings in comparison with state-of-the-art baselines, along with extensive ablation studies, to substantiate the superior performance of SGPT.
  </details>  
#### Mixed-Precision Quantization for Federated Learning on Resource-Constrained Heterogeneous Devices

> Huancheng Chen, Haris Vikalo

* Paper: https://arxiv.org/abs/2311.18129
* Code: 
* <details>
  <summary>Abstract (Click to expand):</summary>
      While federated learning (FL) systems often utilize quantization to battle communication and computational bottlenecks, they have heretofore been limited to deploying fixed-precision quantization schemes. Meanwhile, the concept of mixed-precision quantization (MPQ), where different layers of a deep learning model are assigned varying bit-width, remains unexplored in the FL settings. We present a novel FL algorithm, FedMPQ, which introduces mixed-precision quantization to resource-heterogeneous FL systems. Specifically, local models, quantized so as to satisfy bit-width constraint, are trained by optimizing an objective function that includes a regularization term which promotes reduction of precision in some of the layers without significant performance degradation. The server collects local model updates, de-quantizes them into full-precision models, and then aggregates them into a global model. To initialize the next round of local training, the server relies on the information learned in the previous training round to customize bit-width assignments of the models delivered to different clients. In extensive benchmarking experiments on several model architectures and different datasets in both iid and non-iid settings, FedMPQ outperformed the baseline FL schemes that utilize fixed-precision quantization while incurring only a minor computational overhead on the participating devices.
  </details>
  



## CVPR2024 FL Poster Schedule

#### 6.19 PM

| Paper ID | Paper Title                                                  |
| -------- | ------------------------------------------------------------ |
| 4722     | FedHCA$^2$: Towards Hetero-Client Federated Multi-Task  Learning |
| 10342    | FedUV: Uniformity and Variance for Heterogeneous  Federated Learning |
| 14579    | Efficiently Assemble Normalization Layers and  Regularization for Federated Domain Generalization |
| 15929    | Unlocking the Potential of Prompt-Tuning in Bridging  Generalized and Personalized Federated Learning |
| 17278    | Mixed-Precision Quantization for Federated Learning on  Resource-Constrained Heterogeneous Devices |

#### 6.20 AM

| Paper ID | Paper Title                                                  |
| -------- | ------------------------------------------------------------ |
| 1898     | FedSOL: Stabilized Orthogonal Learning with Proximal  Restrictions in Federated Learning |
| 2006     | FedAS: Bridging Inconsistency in Personalized  Federated Learning |
| 3595     | Data Valuation and Detections in Federated Learning          |
| 4377     | Fair Federated Learning under Domain Skew with Local  Consistency and Domain Diversity |
| 4992     | FLHetBench: Benchmarking Device and State  Heterogeneity in Federated Learning |
| 5850     | An Upload-Efficient Scheme for Transferring Knowledge  From a Server-Side Pre-trained Generator to Clients in Heterogeneous  Federated Learning |
| 6502     | Think Twice Before Selection: Federated Evidential  Active Learning for Medical Image Analysis with Domain Shifts |
| 6869     | Global and Local Prompts Cooperation via Optimal  Transport for Federated Learning |
| 8638     | Leak and Learn: An Attacker's Cookbook to Train Using  Leaked Data from Federated Learning |
| 9311     | Relaxed Contrastive Learning for Federated Learning          |
| 11497    | Adaptive Hyper-graph Aggregation for Modality-Agnostic  Federated Learning |
| 13071    | Device-Wise Federated Network Pruning                        |
| 13797    | Towards Efficient Replay in Federated Incremental  Learning  |
| 14237    | Communication-Efficient Federated Learning with  Accelerated Client Gradient |
| 15265    | Traceable Federated Continual Learning                       |

#### 6.21 AM

| Paper ID | Paper Title                                                  |
| -------- | ------------------------------------------------------------ |
| 2768     | Federated Online Adaptation for Deep Stereo                  |
| 2987     | Rethinking the Representation in Federated  Unsupervised Learning with Non-IID Data |
| 13579    | Directed Decentralized Collaboration for Personalized  Federated Learning |
| 13614    | PerAda: Parameter-Efficient Federated Learning  Personalization with Generalization Guarantees |
| 14025    | Text-Enhanced Data-free Approach for Federated  Class-Incremental Learning |
| 15081    | FedSelect: Personalized Federated Learning with  Customized Selection of Parameters for Fine-Tuning |

#### 6.21 PM

| Paper ID | Paper Title                                                  |
| -------- | ------------------------------------------------------------ |
| 4916     | Federated Generalized Category Discovery                     |
| 6841     | Revamping Federated Learning Security from a  Defender's Perspective: A Unified Defense with Homomorphic Encrypted Data  Space |
| 7130     | FedMef: Towards Memory-efficient Federated Dynamic  Pruning  |
| 12291    | An Aggregation-Free Federated Learning for Tackling  Data Heterogeneity |
| 13655    | DiPrompT: Disentangled Prompt Tuning for Multiple  Latent Domain Generalization in Federated Learning |
| 15367    | Byzantine-robust Decentralized Federated Learning via  Dual-domain Clustering and Trust Bootstrapping |