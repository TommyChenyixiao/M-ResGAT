# M-ResGAT: Multi-hop Motif-based Residual Graph Attention Networks

This repository contains the implementation of M-ResGAT, a novel graph neural network architecture that enhances ResGAT with multi-hop attention mechanisms and motif-based structural information for improved node classification in citation networks.

## Overview

M-ResGAT extends the traditional Graph Attention Network (GAT) by incorporating:
- Multi-hop attention mechanisms to capture broader network context
- Motif-based structural patterns for enhanced graph representation
- Residual connections for better gradient flow
- Hybrid information matrices combining first-order and higher-order relationships

## Model Architecture

The key components of our architecture include:
- Motif-based hybrid information integration
- Residual connections with layer normalization

## Requirements

```bash
# Core dependencies
torch>=1.9.0
torch-geometric>=2.0.0
torch-scatter>=2.0.9
torch-sparse>=0.6.12
numpy>=1.21.0
scikit-learn>=0.24.2
matplotlib>=3.4.3
PyYAML>=5.1
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/M-ResGAT.git
cd M-ResGAT

# Create a conda environment
conda create -n resgat python=3.8
conda activate resgat

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train on a specific dataset
python train.py --datasets Cora_ML

# Train on multiple datasets
python train.py --datasets Cora_ML CiteSeer

# Train specific models
python train.py --datasets Cora_ML --models GAT ResGAT M-ResGAT

```

### Configuration

Modify `config` in train.py to adjust model parameters:
- Model architecture (hidden channels, layers, heads)
- Training parameters (learning rate, weight decay)
- Multi-hop settings (number of hops, combination method)
- Motif attention parameters (beta coefficient)

## Project Structure

```
M-ResGAT/
├── data/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── gcn.py
│   ├── graphsage.py
│   ├── gat.py
│   ├── resgat.py
│   ├── mresgat.py
│   └── M_resgat.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── visualization.py
├── requirements.txt
└── train.py
```

## Evaluation Metrics

The model's performance is evaluated using:
- Classification Accuracy
- Macro F1-Score
- Node Classification AUC

## Results


## Citation
- Veličković, Petar, Cucurull, Guillem, Casanova, Arantxa, Romero, Adriana, Liò, Pietro, & Bengio, Yoshua. (2017). [Graph Attention Networks](https://doi.org/10.48550/arXiv.1710.10903). *arXiv*. doi:10.48550/arXiv.1710.10903.

- Debackere, Wolfgang, & Glänzel, Koenraad. (2022). [Various aspects of interdisciplinarity in research and how to quantify and measure those](https://ideas.repec.org/a/spr/scient/v127y2022i9d10.1007_s11192-021-04133-4.html). *Scientometrics*, 127(9), 5551–5569. doi:10.1007/s11192-021-04133-4.

- Nguyen-Vo, Thanh-Hoang, Do, Trang T. T., & Nguyen, Binh P. (2024). [ResGAT: Residual Graph Attention Networks for molecular property prediction](https://doi.org/10.1007/s12293-024-00423-5). *Memetic Computing*, 16(3), 491–503. doi:10.1007/s12293-024-00423-5.

- Wang, Guangtao, Ying, Rex, Huang, Jing, & Leskovec, Jure. (2020). [Multi-hop Attention Graph Neural Network](https://doi.org/10.48550/arXiv.2009.14332). *arXiv*. doi:10.48550/arXiv.2009.14332.

- Huang, Xuejian, Wu, Zhibin, Wang, Gensheng, Li, Zhipeng, Luo, Yuansheng, & Wu, Xiaofang. (2024). [ResGAT: an improved graph neural network based on multi-head attention mechanism and residual network for paper classification](https://doi.org/10.1007/s11192-023-04898-w). *Scientometrics*, 129(2), 1015–1036. doi:10.1007/s11192-023-04898-w.

- Ye, Huang ZhengWei, Min JinTao, Yang YanNi, Huang Jin, & Tian. (2022). [Recommendation method for academic journal submission based on doc2vec and XGBoost](https://ideas.repec.org/a/spr/scient/v127y2022i5d10.1007_s11192-022-04354-1.html). *Scientometrics*, 127(5), 2381–2394. doi:10.1007/s11192-022-04354-1.

- Zhang, Lin, Sun, Beibei, Shu, Fei, & Huang, Ying. (2022). [Comparing paper level classifications across different methods and systems: an investigation of Nature publications](https://doi.org/10.1007/s11192-022-04352-3). *Scientometrics*, 127(12), 7633–7651. doi:10.1007/s11192-022-04352-3.

- Zhu, Yanqiao, Xu, Yichen, Yu, Feng, Liu, Qiang, Wu, Shu, & Wang, Liang. (2020). [Deep Graph Contrastive Representation Learning](https://doi.org/10.48550/arXiv.2006.04131). *arXiv*. doi:10.48550/arXiv.2006.04131.

- You, Yuning, Chen, Tianlong, Sui, Yongduo, Chen, Ting, Wang, Zhangyang, & Shen, Yang. (2020). [Graph Contrastive Learning with Augmentations](https://www.semanticscholar.org/paper/Graph-Contrastive-Learning-with-Augmentations-You-Chen/2a9fbca9dc6badbeedc591ad829c5c6e0f950fd6). *Neural Information Processing Systems*.

- Bojchevski, Aleksandar, & Günnemann, Stephan. (2017). [Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking](https://doi.org/10.48550/arXiv.1707.03815). *arXiv*. doi:10.48550/arXiv.1707.03815.

- Wu, Z., & Hao, J. (2024). [Electrical transport properties in group-V elemental ultrathin 2D layers](https://doi.org/10.1038/s41699-020-0139-x). *npj 2D Materials and Applications*, 4(1). doi:10.1038/s41699-020-0139-x.

- Kipf, Thomas N., & Welling, Max. (2016). [Semi-Supervised Classification with Graph Convolutional Networks](https://doi.org/10.48550/arXiv.1609.02907). *arXiv*. doi:10.48550/arXiv.1609.02907.

- Hamilton, William L., Ying, Rex, & Leskovec, Jure. (2017). [Inductive Representation Learning on Large Graphs](https://doi.org/10.48550/arXiv.1706.02216). *arXiv*. doi:10.48550/arXiv.1706.02216.

- Verma, Atul Kumar, Saxena, Rahul, Jadeja, Mahipal, Bhateja, Vikrant, & Lin, Jerry Chun-Wei. (2023). [Bet-GAT: An Efficient Centrality-Based Graph Attention Model for Semi-Supervised Node Classification](https://doi.org/10.3390/app13020847). *Applied Sciences*, 13(2), 847. doi:10.3390/app13020847.

- Kim, Dongkwan, & Oh, Alice. (2019). [Supervised Graph Attention Network for Semi-Supervised Node Classification](https://www.semanticscholar.org/paper/Supervised-Graph-Attention-Network-for-Node-Kim-Oh/431f4d7bbfdbd8add524c7c7cb4b075ad12e5827).

- Li, Xiangci, & Ouyang, Jessica. (2024). [Related Work and Citation Text Generation: A Survey](https://doi.org/10.48550/arXiv.2404.11588). *arXiv*. doi:10.48550/arXiv.2404.11588.

- Zhang, Jiasheng, Chen, Jialin, Maatouk, Ali, Bui, Ngoc, Xie, Qianqian, Tassiulas, Leandros, Shao, Jie, Xu, Hua, & Ying, Rex. (2024). [LitFM: A Retrieval Augmented Structure-aware Foundation Model For Citation Graphs](https://doi.org/10.48550/arXiv.2409.12177). *arXiv*. doi:10.48550/arXiv.2409.12177.

- Wang, Lei, Li, Zheng-Wei, You, Zhu-Hong, Huang, De-Shuang, & Wong, Leon. (2023). [MAGCDA: A Multi-Hop Attention Graph Neural Networks Method for CircRNA-Disease Association Prediction](https://doi.org/10.1109/JBHI.2023.3346821). *IEEE Journal of Biomedical and Health Informatics*, 28(3), 1752–1761. doi:10.1109/JBHI.2023.3346821.

- Deng, Ping, & Huang, Yong. (2024). [Edge-featured multi-hop attention graph neural network for intrusion detection system](https://doi.org/10.1016/j.cose.2024.104132). *Computers & Security*, 148, 104132. doi:10.1016/j.cose.2024.104132.

- Gong, Chuanyang, Wei, Zhihua, Wang, Xinpeng, Wang, Rui, Li, Yu, & Zhu, Ping. (2024). [Subgraph-Based Attention Network for Multi-Hop Question Answering](https://doi.org/10.1109/IJCNN60899.2024.10650851). *2024 International Joint Conference on Neural Networks (IJCNN)*. doi:10.1109/IJCNN60899.2024.10650851.

- Sheng, Jinfang, Zhang, Yufeng, Wang, Bin, & Chang, Yaoxing. (2024). [MGATs: Motif-Based Graph Attention Networks](https://doi.org/10.3390/math12020293). *Mathematics*, 12(2), 293. doi:10.3390/math12020293.

- Sankar, Aravind, Zhang, Xinyang, & Chang, Kevin Chen-Chuan. (2017). [Motif-based Convolutional Neural Network on Graphs](https://doi.org/10.48550/arXiv.1711.05697). *arXiv*. doi:10.48550/arXiv.1711.05697.

- Peng, Hao, Li, Jianxin, Gong, Qiran, Wang, Senzhang, Ning, Yuanxing, & Yu, Philip S. (2018). [Graph Convolutional Neural Networks via Motif-based Attention](https://doi.org/10.48550/arXiv.1811.08270). *arXiv*. doi:10.48550/arXiv.1811.08270.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
