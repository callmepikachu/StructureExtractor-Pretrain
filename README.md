# StructureExtractor-Pretrain

预训练GDM-Net的 StructureExtractor，融合了 ContinualGNN 和 StreamE 思想的长文档图构建模型。

## 项目结构

```
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
│   │   ├── train.py                    # 训练脚本入口 (main function)
│   │   └── train_eval.py               # 训练和评估一体化脚本
│   ├── evaluate/                       # 评估相关代码
│   │   ├── __init__.py
│   │   ├── evaluate.py                 # 评估函数
│   │   └── eval.py                     # 评估脚本入口
│   └── utils/                          # 工具函数
│       ├── __init__.py
│       ├── config.py                   # 配置加载和验证
│       ├── logger.py                   # 日志配置
│       └── # 其他通用工具 (如 metrics 计算)
├── scripts/                            # 实用脚本目录
│   ├── download_redocred.sh            # 下载 ReDocRED 数据的脚本
│   └── run_pipeline.sh                 # 运行完整训练评估流程的脚本
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
```

## 安装指南

1. 克隆项目仓库:
   ```
   git clone <repository-url>
   cd StructureExtractor-Pretrain
   ```

2. 创建虚拟环境 (推荐):
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\\Scripts\\activate  # Windows
   ```

3. 安装依赖:
   ```
   pip install -r requirements.txt
   ```

## 使用方法

### 1. 下载数据集
```
bash scripts/download_redocred.sh
```

### 2. 运行完整训练评估流程
```
bash scripts/run_pipeline.sh
```

### 3. 单独训练模型
```
python src/train/train.py --config config/default_config.yaml --train-data data/train_revised.json --dev-data data/dev_revised.json --output-dir output
```

### 4. 评估模型
```
python src/evaluate/eval.py \\
    --config config/default_config.yaml \\
    --checkpoint output/checkpoints/best_model.pt \\
    --test-data data/dev.json \\
    --output-dir output
```

### 5. 数据预处理（可选，推荐用于大数据集）
```
python scripts/preprocess_data.py \\
    --config config/default_config.yaml \\
    --input-data data/train_revised.json \\
    --output-dir data/preprocessed_train
```

## 配置说明

配置文件 `config/default_config.yaml` 包含模型、数据、训练和评估的所有参数。主要配置项包括：

- `model`: 模型架构参数
- `data`: 数据处理参数
- `training`: 训练超参数
- `evaluation`: 评估参数
- `infrastructure`: 基础设施配置 (设备、日志等)

### 数据加载优化

为了处理大型数据集，我们提供了多种数据加载优化方案：

1. **懒加载**：按需处理数据chunk，减少初始内存占用
2. **内存高效加载器**：使用LRU缓存机制，平衡内存使用和性能
3. **预处理脚本**：预先处理数据，避免训练时重复计算

详细信息请参见 `数据加载优化方案.md` 文件。

## 技术特点

本项目实现了以下关键技术：

1. **ContinualGNN**: 持续学习机制，防止灾难性遗忘
2. **StreamE**: 流式图构建和增量更新
3. **双重视角知识巩固**: 结合数据回放和模型正则化
4. **分层-重要性采样**: 优化记忆库更新策略
5. **近似影响检测**: 高效检测文档变化对知识图谱的影响

## 许可证

本项目基于 MIT 许可证开源。
