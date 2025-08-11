# StructureExtractor-Pretrain
预训练GDM-Net的 StructureExtractor



StructureExtractor-Pretrain/
├── README.md                           # 项目介绍、安装指南、使用方法
├── requirements.txt                    # Python 依赖包列表
├── config/                             # 配置文件目录
│   ├── default_config.yaml             # 默认配置文件
│   └── # 其他特定配置 (如 for_training.yaml, for_inference.yaml)
├── src/                                # 源代码主目录
│   ├── __init__.py
│   ├── data/                           # 数据处理相关代码
│   │   ├── __init__.py
│   │   ├── dataset.py                  # ReDocRED 数据集类 (ReDocREDDataset)
│   │   ├── collator.py                 # 自定义数据整理器 (DataCollator)
│   │   └── preprocess.py               # (可选) 数据预处理脚本/函数
│   ├── model/                          # 模型相关代码 (核心是 StructureExtractor)
│   │   ├── __init__.py
│   │   ├── extractor.py                # 您设计的 StructureExtractor 模型 (可复用或微调 GDM-Net 中的版本)
│   │   └── # 其他模型组件 (如果需要，如简化版的 GraphWriter 用于验证)
│   ├── train/                          # 训练相关代码
│   │   ├── __init__.py
│   │   ├── trainer.py                  # 训练器类 (PretrainTrainer)，封装训练循环
│   │   └── train.py                    # 训练脚本入口 (main function)
│   ├── evaluate/                       # 评估相关代码
│   │   ├── __init__.py
│   │   └── evaluate.py                 # 评估脚本/函数
│   └── utils/                          # 工具函数
│       ├── __init__.py
│       ├── config.py                   # 配置加载和验证
│       ├── logger.py                   # 日志配置
│       └── # 其他通用工具 (如 metrics 计算)
├── scripts/                            # 实用脚本目录
│   ├── download_redocred.sh            # 下载 ReDocRED 数据的脚本 (如果需要)
│   └── # 其他脚本 (如 run_training.sh, run_evaluation.sh)
├── notebooks/                          # (可选) 探索性数据分析或模型实验的 Jupyter Notebooks
│   └── # ...
├── checkpoints/                        # (初始为空) 用于存放训练好的模型检查点
│   └── # ...
├── logs/                               # (初始为空) 用于存放训练日志
│   └── # ...
└── data/                               # (初始为空或符号链接) 指向 ReDocRED 数据集的实际存放位置
    ├── train.json                      # ReDocRED 训练集
    ├── dev.json                        # ReDocRED 验证集
    └── # ... (test.json 等)
