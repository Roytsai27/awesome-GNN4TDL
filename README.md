<div align="center">
    <h1>Graph Neural Networks for Tabular Data Learning (GNN4TDL)</h1>
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg"/></a>
    <a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square"/></a>
</div>

## Table of contents
- [✨ News](#--news)
- [Intro: Graph Neural Networks for Tabular Data Learning](#graph-neural-networks-for-tabular-data-learning)
- [Homogeneous Instance Graphs](#homogeneous-instance-graph-gnns-for-tdl)
- [Homogeneous Feature Graphs](#homogeneous-feature-graph-gnns-for-tdl)
- [Bipartite Graphs](#bipartite-graph-gnns-for-tdl)
- [Heterogeneous Graphs](#heterogeneous-graph-gnns-for-tdl)
- [Hypergraph GNNs](#hypergraph-gnns-for-tdl)
- [Knowledge Graphs](#knowledge-graph-gnns-for-tdl)

👉 If you notice any errors or have suggestions, feel free to share them with us!

⭐️ If you find this resource helpful, please consider to star this repository and cite our [survey paper](https://arxiv.org/abs/2401.02143):
```
@article{li2024graph,
  title={Graph neural networks for tabular data learning: A survey with taxonomy and directions},
  author={Li, Cheng-Te and Tsai, Yu-Che and Chen, Chih-Yao and Liao, Jay Chiehen},
  journal={ACM Computing Surveys},
  year={2024},
  publisher={ACM New York, NY}
}
```

## ✨ News
* [2023-12-16] We have released this repository that collects the resources related to GNNs for tabular data learning (GNN4TDL).

## Intro: Graph Neural Networks for Tabular Data Learning
<p align="center"><img src="https://github-production-user-asset-6210df.s3.amazonaws.com/38278579/291000976-5ab868e1-56ce-47e7-9ea9-cb17bc54f952.png" width="65%" height="65%"></p>

The deep learning-based approaches to Tabular Data Learning (TDL), classification and regression, have shown competing performance, compared to their conventional counterparts. However, the latent correlation among data instances and feature values is less modeled in deep neural TDL. Recently, graph neural networks (GNNs), which can enable modeling relations and interactions between different tabular data elements, has received tremendous attention across application domains including TDL. It turns out creating proper graph structures from the input tabular dataset, along with GNN learning, can improve the TDL performance. In this survey, we systematically review the methodologies of designing and applying GNNs for TDL (GNN4TDL). The topics to be covered include: (1) foundations and overview of GNN-based TDL methods; (2) a comprehensive taxonomy of constructing graph structures and representation learning in GNN-based TDL methods; (3) how to apply GNN to various TDL application scenarios and tasks; (4) limitations in current research and future directions.

![taxnomy2-1](https://github.com/Roytsai27/awesome-GNN4TDL/assets/38278579/adc91fd0-0e74-42aa-ad58-076edc3417bf)

This survey presents an in-depth exploration into the application of GNNs in tabular data learning. It starts by establishing the fundamental problem statement and introduces various graph types used to represent tabular data. The survey is structured around a detailed GNN-based learning pipeline, encompassing phases like **Graph Formulation**, where tabular elements are converted into graph nodes; **Graph Construction**, focusing on establishing connections within these elements; **Representation Learning**, highlighting how GNNs process these structures to learn data instance features; and **Training Plans**, discussing the integration of auxiliary tasks for enhanced predictive outcomes.

## Homogeneous Instance Graphs

|Year|Title|Venue|Paper|Code|
| :- | :- | :- | :- | :- |
|2023|EGG-GAE: Scalable Graph Neural Networks for Tabular Data Imputation|ICML 2023|[Link](https://arxiv.org/pdf/2210.10446.pdf)|[Link](https://github.com/levtelyatnikov/EGG_GAE)|
|2023|Graph Neural Networks for Missing Value Classification in a Task-driven Metric Space|TKDE 2023|[Link](https://doi.org/10.1109/TKDE.2022.3198689)||
|2023|Look Around! A Neighbor Relation Graph Learning Framework for Real Estate Appraisal|AAAI 2023|[Link](https://arxiv.org/pdf/2212.12190.pdf)||
|2023|Homophily-Enhanced Self-Supervision for Graph Structure Learning: Insights and Directions|TNNLS 2023|[Link](https://doi.org/10.1109/TNNLS.2023.3257325)|[Link](https://github.com/LirongWu/Homophily-Enhanced-Self-supervision)|
|2023|TabGSL: Graph Structure Learning for Tabular Data Prediction|arXiv 2023|[Link](https://arxiv.org/pdf/2305.15843.pdf)||
|2023|Oversmoothing Relief Graph Convolutional Network-Based Fault Diagnosis Method with Application to the Rectifier of High-Speed Trains|IEEE TII 2023|[Link](https://doi.org/10.1109/TII.2022.3167522)||
|2022|Self-Adaptation Graph Attention Network via Meta-Learning for Machinery Fault Diagnosis with Few Labeled Data|IEEE TIM 2022|[Link](https://doi.org/10.1109/TIM.2022.3181894)||
|2023|A Graph Neural Network-based Bearing Fault Detection Method|Scientific Reports 2023|[Link](https://doi.org/10.1038/s41598-023-32369-y)||
|2023|Latent Graphs for Semi-Supervised Learning on Biomedical Tabular Data|IJCLR 2023|[Link](https://arxiv.org/pdf/2309.15757.pdf)||
|2023|Interpretable Graph Neural Networks for Tabular Data|arXiv 2023|[Link](https://arxiv.org/pdf/2308.08945.pdf)||
|2022|Differentiable graph module (dgm) for graph convolutional networks|IEEE TPAMI 2022|[Link](https://arxiv.org/pdf/2002.04999.pdf)|[Link](https://github.com/lcosmo/DGM\_pytorch)|
|2022|Towards Unsupervised Deep Graph Structure Learning|WWW 2022|[Link](https://arxiv.org/pdf/2201.06367.pdf)|[Link](https://github.com/GRAND-Lab/SUBLIME)|
|2022|LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks|AAAI 2022|[Link](https://arxiv.org/pdf/2112.05355.pdf)|[Link](https://github.com/agoodge/LUNAR)|
|2022|Graph Regularized Autoencoder and its Application in Unsupervised Anomaly Detection|IEEE TPAMI 2022|[Link](https://arxiv.org/pdf/2010.15949.pdf)||
|2022|Deep Unsupervised Active Learning on Learnable Graphs|IEEE TNNLS 2022|[Link](https://arxiv.org/pdf/2111.04286.pdf)||
|2022|Graph autoencoder-based unsupervised outlier detection|Information Sciences 2022|[Link](https://doi.org/10.1016/j.ins.2022.06.039)|[Link](https://github.com/duxusheng-xju/GAE-Unsupervised-Outlier-Detection)|
|2022|Structure-Aware Siamese Graph Neural Networks for Encounter-Level Patient Similarity Learning|Journal of Biomedical Informatics 2022|[Link](https://doi.org/10.1016/j.jbi.2022.104027)||
|2022|GEDI: A Graph-based End-to-end Data Imputation Framework|arXiv 2022|[Link](https://arxiv.org/pdf/2208.06573.pdf)||
|2021|Self-supervision improves structure learning for graph neural networks.|NeurIPS 2021|[Link](https://arxiv.org/pdf/2102.05034.pdf)|[Link](https://github.com/BorealisAI/SLAPS-GNN)|
|2021|Predicting Patient Outcomes with Graph Representation Learning|DLG-AAAI 2021|[Link](https://arxiv.org/pdf/2101.03940.pdf)|[Link](https://github.com/EmmaRocheteau/eICU-GNN-LSTM)|
|2021|A Weighted Patient Network-based Framework for Predicting Chronic Diseases Using Graph Neural Networks|Scientific Reports 2021|[Link](https://doi.org/10.1038/s41598-021-01964-2)||
|2021|k-Nearest Neighbor Learning with Graph Neural Networks|Mathematics 2021|[Link](https://www.mdpi.com/2227-7390/9/8/830/pdf)||
|2020|Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings|NeurIPS 2020|[Link](https://arxiv.org/pdf/2006.13009.pdf)|[Link](https://github.com/hugochan/IDGL)|
|2020|Missing Data Imputation with Adversarially-trained Graph Convolutional Networks|Neural Networks 2020|[Link](https://arxiv.org/pdf/1905.01907.pdf)|[Link](https://github.com/spindro/GINN)|
|2019|Learning Discrete Structures for Graph Neural Networks|ICML 2019|[Link](https://arxiv.org/pdf/1903.11960.pdf)|[Link](https://github.com/lucfra/LDS-GNN)|

## Homogeneous Feature Graphs


|Year|Title|Venue|Paper|Code|
| :- | :- | :- | :- | :- |
|2024|Graph Neural Network Contextual Embedding for Deep Learning on Tabular Data|Neural Networks 2024|[Link](https://doi.org/10.1016/j.neunet.2024.106180)|[Link](https://github.com/MatteoSalvatori/INCE)|
|2023|T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction|AAAI 2023|[Link](https://arxiv.org/pdf/2211.16887.pdf)|[Link](https://github.com/jyansir/t2g-former)|
|2023|Deep Tabular Data Modeling With Dual-Route Structure-Adaptive Graph Networks|IEEE TKDE 2023|[Link](https://ieeexplore.ieee.org/document/10054100)||
|2023|Causality-based CTR Prediction using Graph Neural Networks|Information Processing & Management 2023|[Link](https://doi.org/10.1016/j.ipm.2022.103137)||
|2023|FT-GAT: Graph Neural Network for Predicting Spontaneous Breathing Trial Success in Patients with Mechanical Ventilation|Computer Methods and Programs in Biomedicine 2023|[Link](https://pubmed.ncbi.nlm.nih.gov/37336152/)||
|2022|Table2Graph: Transforming Tabular Data to Unifed Weighted Graph|IJCAI 2022|[Link](https://www.ijcai.org/proceedings/2022/0336.pdf)||
|2022|Local Contrastive Feature Learning for Tabular Data|CIKM 2022|[Link](https://arxiv.org/pdf/2211.10549.pdf)||
|2021|FIVES: Feature Interaction Via Edge Search for Large-Scale Tabular Data|KDD 2021|[Link](https://arxiv.org/pdf/2007.14573.pdf)||
|2021|TabularNet: A Neural Network Architecture for Understanding Semantic Structures of Tabular Data|KDD 2021|[Link](https://arxiv.org/pdf/2106.03096.pdf)||
|2021|GCN-Int: A Click-Through Rate Prediction Model Based on Graph Convolutional Network Interaction|IEEE Access 2021|[Link](https://ieeexplore.ieee.org/document/9552883)||
|2019|Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction|CIKM 2019|[Link](https://arxiv.org/pdf/1910.05552.pdf)|[Link](https://github.com/CRIPAC-DIG/Fi_GNN)|


## Bipartite Graphs


|Year|Title|Venue|Paper|Code|
| :- | :- | :- | :- | :- |
|2023|Data Imputation with Iterative Graph Reconstruction|AAAI 2023|[Link](https://arxiv.org/pdf/2212.02810.pdf)||
|2022|Learning Enhanced Representations for Tabular Data via Neighborhood Propagation|NeurIPS 2022|[Link](https://arxiv.org/pdf/2206.06587.pdf)|[Link](https://github.com/KounianhuaDu/PET)|
|2022|Relational Multi-Task Learning: Modeling Relations between Data and Tasks|ICLR 2022|[Link](https://arxiv.org/pdf/2303.07666.pdf)|[Link](https://github.com/snap-stanford/GraphGym)|
|2022|Predicting the Survival of Cancer Patients with Multimodal Graph Neural Network|IEEE TCBB 2022|[Link](https://doi.org/10.1109/TCBB.2021.3083566)||
|2021|Towards Open-World Feature Extrapolation: An Inductive Graph Learning Approach|NeurIPS 2021|[Link](https://arxiv.org/pdf/2110.04514.pdf)|[Link](https://github.com/qitianwu/FATE)|
|2021|MedGraph: Structural and Temporal Representation Learning of Electronic Medical Records|ECAI 2021|[Link](https://arxiv.org/pdf/1912.03703.pdf)|[Link](https://github.com/bhagya-hettige/MedGraph)|
|2021|Disease Prediction via Graph Neural Networks|IEEE JBHI 2021|[Link](https://ieeexplore.ieee.org/document/9122573)|[Link](https://github.com/zhchs/Disease-Prediction-via-GCN)|
|2020|Handling Missing Data with Graph Representation Learning|NeurIPS 2020|[Link](https://arxiv.org/pdf/2010.16418.pdf)|[Link](https://github.com/maxiaoba/GRAPE)|
|2019|Large-Scale Heterogeneous Feature Embedding|AAAI 2019|[Link](https://ojs.aaai.org/index.php/AAAI/article/view/4276)|[Link](https://github.com/DEEP-PolyU/FeatWalk_AAAI19)|


## Heterogeneous Graphs


|Year|Title|Venue|Paper|Code|
| :- | :- | :- | :- | :- |
|2023|Relational Deep Learning: Graph Representation Learning on Relational Databases|arXiv 2023|[Link](https://arxiv.org/pdf/2312.04615.pdf)|[Link](https://github.com/snap-stanford/relbench)|
|2023|GCondNet: A Novel Method for Improving Neural Networks on Small High-Dimensional Tabular Data|NeurIPS 2023 TRL Workshop|[Link](https://arxiv.org/pdf/2211.06302.pdf)||
|2023|Lifelong Property Price Prediction: A Case Study for the Toronto Real Estate Market|IEEE TKDE 2023|[Link](https://arxiv.org/pdf/2008.05880.pdf)|[Link](https://github.com/RingBDStack/LUCE)|
|2023|GraphFC: Customs Fraud Detection with Label Scarcity|CIKM 2023|[Link](https://arxiv.org/pdf/2305.11377.pdf)|[Link](https://github.com/k-s-b/gnn_wco)|
|2023|GFS: Graph-based Feature Synthesis for Prediction over Relational Databases|arXiv 2023|[Link](https://arxiv.org/pdf/2312.02037.pdf)||
|2022|GCF-RD: A Graph-based Contrastive Framework for Semi-Supervised Learning on Relational Databases|ACM CIKM 2022|[Link](https://doi.org/10.1145/3511808.3557331)|[Link](https://github.com/ChenRunjin/GCF-RD)|
|2022|Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks|ACM TOIS 2022|[Link](https://arxiv.org/pdf/2104.07886.pdf)|[Link](https://github.com/safe-graph/RioGNN)|
|2022|H2-Fdetector: A GNN-based Fraud Detector with Homophilic and Heterophilic Connections|WWW 2022|[Link](https://doi.org/10.1145/3485447.3512195)||
|2022|xFraud: Explainable Fraud Transaction Detection|VLDB 2022|[Link](https://arxiv.org/pdf/2011.12193.pdf)|[Link](https://github.com/eBay/xFraud)|
|2022|AUC-oriented Graph Neural Network for Fraud Detection|WWW 2022|[Link](https://doi.org/10.1145/3485447.3512178)||
|2022|Hierarchical Multi-Modal Fusion on Dynamic Heterogeneous Graph for Health Insurance Fraud Detection|IEEE ICME 2022|[Link](https://doi.org/10.1109/ICME52920.2022.9859871)||
|2021|Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection|WWW 2021|[Link](https://doi.org/10.1145/3442381.3449989)|[Link](https://github.com/PonderLY/PC-GNN)|
|2021|Learning Graph Meta Embeddings for Cold-Start Ads in Click-Through Rate Prediction|SIGIR 2021|[Link](https://arxiv.org/pdf/2105.08909.pdf)|[Link](https://github.com/oywtece/gme)|
|2021|Modeling Heterogeneous Graph Network on Fraud Detection: A Community-based Framework with Attention Mechanism|CIKM 2021|[Link](https://doi.org/10.1145/3459637.3482277)||
|2021|Towards Consumer Loan Fraud Detection: Graph Neural Networks with Role-Constrained Conditional Random Field|AAAI 2021|[Link](https://doi.org/10.1609/aaai.v35i5.16582)||
|2021|Intention-Aware Heterogeneous Graph Attention Networks for Fraud Transactions Detection|KDD 2021|[Link](https://doi.org/10.1145/3447548.3467142)||
|2021|TabGNN: Multiplex Graph Neural Network for Tabular Data Prediction|DLP-KDD 2021|[Link](https://arxiv.org/pdf/2108.09127.pdf)|[Link](https://github.com/LARS-research/TabGNN)|
|2020|Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters|CIKM 2020|[Link](https://arxiv.org/pdf/2008.08692.pdf)|[Link](https://github.com/YingtongDou/CARE-GNN)|
|2020|Loan Default Analysis with Multiplex Graph Learning|ACM CIKM 2020|[Link](https://doi.org/10.1145/3340531.3412724)||
|2020|Financial Risk Analysis for SMEs with Graph-based Supply Chain Mining|IJCAI 2020|[Link](https://www.ijcai.org/proceedings/2020/0643.pdf)||
|2020|Heterogeneous Similarity Graph Neural Network on Electronic Health Records|IEEE BigData 2020|[Link](https://arxiv.org/pdf/2101.06800.pdf)||
|2020|Supervised Learning on Relational Databases with Graph Neural Networks|ICLR 2020|[Link](https://arxiv.org/pdf/2002.02046.pdf)|[Link](https://github.com/mwcvitkovic/Supervised-Learning-on-Relational-Databases-with-GNNs)|
|2020|Learning the Graphical Structure of Electronic Health Records with Graph Convolutional Transformer|AAAI 2020|[Link](https://arxiv.org/pdf/1906.04716.pdf)|[Link](https://github.com/Google-Health/records-research/tree/master/graph-convolutional-transformer)|


## Hypergraphs
|Year|Title|Venue|Paper|Code|
| :- | :- | :- | :- | :- |
|2023|HYTREL: Hypergraph-enhanced Tabular Data Representation Learning|NeurIPS 2023|[Link](https://arxiv.org/pdf/2307.08623.pdf)|[Link](https://github.com/awslabs/hypergraph-tabular-lm)|
|2022|Learning Enhanced Representations for Tabular Data via Neighborhood Propagation|NeurIPS 2022|[Link](https://arxiv.org/pdf/2206.06587.pdf)|[Link](https://github.com/KounianhuaDu/PET)|
|2022|Hypergraph Contrastive Learning for Electronic Health Records|SDM 2022|[Link](https://doi.org/10.1137/1.9781611977172.15)||

## Knowledge Graphs
|Year|Title|Venue|Paper|Code|
| :- | :- | :- | :- | :- |
|2023|High dimensional, tabular deep learning with an auxiliary knowledge graph|NeurIPS 2023|[Link](https://arxiv.org/pdf/2306.04766.pdf)|[link](https://github.com/snap-stanford/plato)|

