"""
StructureExtractor-Pretrain 的训练器。
实现了结合 ContinualGNN 和 StreamE 优化的训练循环，
专为长文档图建模流水线而设计。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Set
import logging
import os
from tqdm import tqdm
import json
import numpy as np

# 类型占位符
ContinualGNNUpdater = object

# --- 导入项目模块 ---
from src.model.extractor import StructureExtractor  # 流水线协调器
# ContinualGNNUpdater 已经集成在 StructureExtractor 中
from src.data.dataset import ReDocREDDataset  # 数据集类
from src.evaluate.evaluate_adapted import evaluate_model  # 图评估函数
# GT 图构建函数已经在 evaluate_model 中实现

logger = logging.getLogger(__name__)


class PretrainTrainer:
    """
    用于预训练 StructureExtractor 模型的训练器类，使用持续学习 (Continual Learning)。
    此训练器专为处理文档序列并增量更新知识图谱的流水线而设计。
    """

    def __init__(self,
                 model: StructureExtractor, # 流水线协调器
                 train_dataset: ReDocREDDataset,
                 dev_dataset: Optional[ReDocREDDataset],
                 config: Dict[str, Any]):
        """
        初始化训练器。

        Args:
            model: StructureExtractor 流水线协调器。
            train_dataset: 训练数据集 (文档)。
            dev_dataset: 可选的验证数据集 (文档)。
            config: 训练配置字典。
        """
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.config = config

        # --- 训练配置 ---
        self.training_config = config['training']
        self.num_epochs = self.training_config['num_epochs']
        self.gradient_clip_norm = self.training_config.get('gradient_clip_norm', 1.0)
        self.save_steps = self.training_config.get('save_steps', 1000)
        self.eval_steps = self.training_config.get('eval_steps', 500)
        self.logging_steps = self.training_config.get('logging_steps', 100)

        # --- 早停配置 ---
        self.early_stopping_patience = self.training_config.get('early_stopping_patience', 3)
        self.early_stopping_metric = self.training_config.get('early_stopping_metric', 'graph_similarity')
        # 'min' 或 'max'，取决于指标是越小越好还是越大越好
        self.early_stopping_mode = self.training_config.get('early_stopping_mode', 'max')
        self.early_stopping_threshold = self.training_config.get('early_stopping_threshold', 1e-4)

        # --- 基础设施配置 ---
        self.infrastructure_config = config['infrastructure']
        self.device = self._setup_device()
        self.use_mixed_precision = self.infrastructure_config.get('mixed_precision', False)
        self.output_dir = self.infrastructure_config.get('output_dir', './outputs')

        # --- 设置组件 ---
        # 1. 将模型组件移至设备 (如果适用)
        #    对于流水线协调器，这可能不直接适用。
        #    self.model.to(self.device) # 可能无法直接在流水线协调器上工作

        # 2. 从流水线中获取需要训练的 GNN 模型
        #    这需要您的 StructureExtractor 暴露该模型。
        self.gnn_model = self._get_trainable_gnn_model()
        if self.gnn_model is None:
            raise ValueError("无法从流水线中获取可训练的 GNN 模型。")

        # 3. 为 GNN 模型初始化优化器
        self.optimizer = self._create_optimizer()
        if self.optimizer is None:
            raise ValueError("创建优化器失败。")

        # 4. 初始化调度器 (可选)
        self.scheduler = self._create_scheduler() # 可以为 None

        # 5. 初始化持续学习更新器
        #    这是整合 ContinualGNN 思想的核心。
        self.continual_updater = self._create_continual_updater()
        if self.continual_updater is None:
            raise ValueError("创建 ContinualGNN 更新器失败。")

        # 6. 初始化数据加载器
        #    关键：文档流的 batch_size 通常为 1。
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=1, # 一次处理一个文档
            shuffle=True,
            collate_fn=self._document_collate_fn # 为单个文档自定义 collate
        )

        if dev_dataset:
            self.dev_dataloader = DataLoader(
                dev_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=self._document_collate_fn
            )
        else:
            self.dev_dataloader = None

        # --- 跟踪变量 ---
        self.global_step = 0
        # 根据模式初始化最佳评估指标
        self.best_eval_metric = float('-inf') if self.early_stopping_mode == 'max' else float('inf')
        self.patience_counter = 0
        self.early_stopping_triggered = False

        # --- 日志设置 ---
        self._setup_logging()

        logger.info("PretrainTrainer 初始化成功。")

    def _setup_device(self) -> torch.device:
        """设置训练设备。"""
        device_config = self.infrastructure_config.get('device', 'auto')
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config.startswith('cuda'):
            device = torch.device(device_config)
        else:
            device = torch.device('cpu')
        logger.info(f"使用设备: {device}")
        return device

    def _get_trainable_gnn_model(self):
        """
        从 StructureExtractor 流水线中检索可训练的 GNN 模型。
        StructureExtractor 本身就是可训练的模型。
        """
        # StructureExtractor 本身就是可训练的模型
        return self.model

    def _create_optimizer(self) -> Optional[optim.Optimizer]:
        """为 GNN 模型创建优化器。"""
        if not self.gnn_model:
            return None
        try:
            optimizer_name = self.training_config.get('optimizer', 'AdamW').lower()
            learning_rate = float(self.training_config.get('learning_rate', 1e-4))
            weight_decay = float(self.training_config.get('weight_decay', 0.01))

            # 仅获取需要梯度的 GNN 模型参数
            gnn_params = [p for p in self.gnn_model.parameters() if p.requires_grad]

            if optimizer_name == 'adamw':
                optimizer = optim.AdamW(gnn_params, lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_name == 'adam':
                optimizer = optim.Adam(gnn_params, lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_name == 'sgd':
                optimizer = optim.SGD(gnn_params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
            else:
                raise ValueError(f"不支持的优化器: {optimizer_name}")
            logger.info(f"使用优化器: {optimizer_name}，学习率: {learning_rate}")
            return optimizer
        except Exception as e:
            logger.error(f"创建优化器时出错: {e}")
            return None

    def _create_scheduler(self):
        """创建学习率调度器 (可选)。"""
        # 占位符: 暂时不使用调度器
        return None

    def _create_continual_updater(self) -> Optional[ContinualGNNUpdater]:
        """
        创建 ContinualGNN 更新器实例。
        在当前实现中，持续学习逻辑已经集成在 StructureExtractor 中。
        """
        # 持续学习逻辑已经集成在 StructureExtractor 中
        logger.info("使用集成在 StructureExtractor 中的 ContinualGNN 更新器。")
        return None  # 不需要单独的更新器实例

    def _setup_logging(self):
        """设置日志记录和实验跟踪 (例如, wandb)。"""
        # 假设基本的文件日志已经设置好
        # 在此处添加更复杂的日志/跟踪 (例如, wandb)
        pass

    def _document_collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        用于文档流的自定义 collate 函数。
        由于 batch_size=1，简单地返回单个文档字典。
        """
        # batch 是一个包含一个文档字典的列表
        return batch[0] # 直接返回文档数据

    def train(self):
        """
        主训练循环。
        """
        logger.info("开始训练循环...")
        os.makedirs(self.output_dir, exist_ok=True)

        for epoch in range(self.num_epochs):
            if self.early_stopping_triggered:
                logger.info("触发早停。停止训练。")
                break

            logger.info(f"开始第 {epoch + 1}/{self.num_epochs} 轮")
            train_metrics = self._train_epoch(epoch)

            # 如果有验证数据集则进行评估
            if self.dev_dataloader:
                eval_metrics = self._evaluate()
                logger.info(f"第 {epoch + 1} 轮 - 训练指标: {train_metrics}")
                logger.info(f"第 {epoch + 1} 轮 - 评估指标: {eval_metrics}")

                # 根据评估指标检查早停
                self._check_early_stopping(eval_metrics)
            else:
                logger.info(f"第 {epoch + 1} 轮 - 训练指标: {train_metrics}")

            # 在每轮结束时保存检查点
            self._save_checkpoint(epoch, is_best=False, is_epoch_end=True)

        logger.info("训练完成。")
        # 保存最终模型
        self._save_checkpoint(self.num_epochs - 1, is_best=False, is_final=True)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一轮 (按顺序处理文档)。

        Args:
            epoch: 当前轮数。

        Returns:
            该轮的平均训练指标。
        """
        # 如果组件有 train 方法，则设置为训练模式
        # self.model.train() # 可能不适用
        if hasattr(self.gnn_model, 'train'):
             self.gnn_model.train()

        # 初始化指标累加器
        epoch_metrics = {"avg_cl_loss": 0.0, "documents_processed": 0}
        num_documents = 0

        # 如果请求且可用，则使用混合精度缩放器
        scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision and self.device.type == 'cuda' else None

        progress_bar = tqdm(self.train_dataloader, desc=f"第 {epoch + 1} 轮")

        for doc_idx, document_data in enumerate(progress_bar):
            # document_data 是数据集中的单个文档字典

            try:
                # --- 核心步骤 1: 运行 StructureExtractor 流水线 ---
                # 这应该处理文档，更新图数据库，
                # 执行实体链接等，并返回 CL 所需的必要信息。
                # 流水线本身可能不执行 GNN 参数更新，
                # 但会准备增量图和可能的初始 GNN 表示。
                
                # 获取文档文本
                document_text = document_data.get('text', '')
                if not document_text:
                    logger.warning(f"文档 {doc_idx} 没有文本内容。跳过 CL 更新。")
                    continue
                    
                # 处理文档并获取输出
                pipeline_outputs = self.model.forward(document_text)
                
                # 构建增量图
                entities = pipeline_outputs.get('entities', [])
                relations = pipeline_outputs.get('relations', [])
                delta_graph = self.model.build_incremental_graph(entities, relations)
                
                if not delta_graph or (len(delta_graph.get('nodes', [])) == 0 and len(delta_graph.get('edges', [])) == 0):
                     logger.warning(f"文档 {doc_idx} 未产生增量图。跳过 CL 更新。")
                     continue

                # --- 核心步骤 2: 执行持续学习更新 ---
                # 检测受影响的节点
                influenced_nodes = self.model.detect_influenced_nodes(delta_graph)
                
                # 更新图和模型
                self.model.update_graph_and_model(delta_graph, list(influenced_nodes))
                
                # 合并增量图到主图
                self.model.merge_delta_graph_into_main_graph(delta_graph)
                
                # 更新内存缓冲区
                self.model.update_memory_buffer(influenced_nodes)
                
                # 计算损失（用于跟踪）
                # 在当前实现中，损失已经在 update_graph_and_model 中处理
                # 我们可以通过计算预测图和真实图之间的差异来近似计算损失
                cl_loss_val = 0.0  # 简化处理，实际损失已在模型内部计算

                # --- 更新跟踪 ---
                epoch_metrics["avg_cl_loss"] += cl_loss_val
                epoch_metrics["documents_processed"] += 1
                num_documents += 1
                self.global_step += 1

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{cl_loss_val:.4f}",
                    'avg_loss': f"{epoch_metrics['avg_cl_loss'] / num_documents:.4f}"
                })

                # --- 日志记录 ---
                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_metrics['avg_cl_loss'] / num_documents
                    logger.info(f"步骤 {self.global_step} - CL 损失: {cl_loss_val:.4f}, 平均损失: {avg_loss:.4f}")
                    # 如果配置了外部跟踪器 (wandb 等) 则记录

                # --- 评估 ---
                if self.dev_dataloader and self.global_step % self.eval_steps == 0:
                    eval_metrics = self._evaluate()
                    logger.info(f"步骤 {self.global_step} - 中间评估指标: {eval_metrics}")
                    # 记录评估指标
                    self._check_early_stopping(eval_metrics) # 轮内也检查早停

                # --- 检查点 ---
                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(epoch, step=self.global_step, is_best=False)

            except Exception as e:
                logger.error(f"在第 {epoch} 轮处理文档 {doc_idx} 时出错: {e}", exc_info=True)
                # 根据错误严重程度，您可以选择继续或中断
                continue # 继续处理下一个文档

        # 计算最终轮平均值
        if num_documents > 0:
            epoch_metrics["avg_cl_loss"] /= num_documents
        else:
            epoch_metrics["avg_cl_loss"] = 0.0

        # 如果 eval 改变了模式，则设置回训练模式
        if hasattr(self.gnn_model, 'train'):
             self.gnn_model.train()

        return epoch_metrics

    def _evaluate(self) -> Dict[str, float]:
        """
        在验证集上使用基于图的指标评估模型。

        Returns:
            评估指标字典。
        """
        if not self.dev_dataloader:
            return {}

        # 设置组件为评估模式
        # self.model.eval() # 可能不适用
        if hasattr(self.gnn_model, 'eval'):
             self.gnn_model.eval()

        logger.info("开始评估...")
        # 使用您现有的 evaluate_model 函数
        eval_results = evaluate_model(
            model=self.model, # 传递流水线协调器
            dataset=self.dev_dataset,
            config=self.config
        )
        logger.info("评估完成。")

        # 设置组件回训练模式
        # self.model.train() # 可能不适用
        if hasattr(self.gnn_model, 'train'):
             self.gnn_model.train()

        return eval_results

    def _check_early_stopping(self, current_metrics: Dict[str, float]):
        """
        根据评估指标检查早停条件是否满足。

        Args:
            current_metrics: 当前评估指标字典。
        """
        current_metric_val = current_metrics.get(self.early_stopping_metric, None)
        if current_metric_val is None:
            logger.warning(f"早停指标 '{self.early_stopping_metric}' 在评估结果中未找到。")
            return

        is_better = False
        # 根据模式判断是否更好
        if self.early_stopping_mode == 'max':
            is_better = current_metric_val > self.best_eval_metric + self.early_stopping_threshold
        else: # 'min'
            is_better = current_metric_val < self.best_eval_metric - self.early_stopping_threshold

        if is_better:
            self.best_eval_metric = current_metric_val
            self.patience_counter = 0
            logger.info(f"新的最佳 {self.early_stopping_metric}: {self.best_eval_metric:.4f}。保存最佳模型。")
            self._save_checkpoint(None, is_best=True)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"触发早停，因为 {self.early_stopping_metric} 在 {self.patience_counter} 次评估后没有改善。"
                )
                self.early_stopping_triggered = True

    def _save_checkpoint(self, epoch: Optional[int], step: Optional[int] = None,
                         is_best: bool = False, is_epoch_end: bool = False, is_final: bool = False):
        """
        保存模型检查点。

        Args:
            epoch: 当前轮数。
            step: 当前全局步骤数。
            is_best: 是否是基于验证指标的最佳模型。
            is_epoch_end: 是否是轮结束时的检查点。
            is_final: 是否是训练结束后的最终模型。
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            checkpoint = {
                'config': self.config,
                'epoch': epoch,
                'step': self.global_step,
                'best_eval_metric': self.best_eval_metric,
                'patience_counter': self.patience_counter,
                # 保存模型状态
                'model_state_dict': self.model.state_dict() if self.model else None,
                # 保存优化器状态
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                # 如果存在，保存调度器状态
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                # 保存图结构
                'graph_state': {
                    'nodes': dict(self.model.graph.graph.nodes(data=True)),
                    'edges': list(self.model.graph.graph.edges(data=True)),
                    'node_embeddings': self.model.graph.node_embeddings,
                    'node_attributes': self.model.graph.node_attributes
                } if hasattr(self.model, 'graph') else None,
                # 保存内存缓冲区
                'memory_buffer_state': {
                    'nodes': self.model.memory_buffer.nodes,
                    'cluster_counts': dict(self.model.memory_buffer.cluster_counts)
                } if hasattr(self.model, 'memory_buffer') else None
            }

            # 确定文件名
            if is_best:
                checkpoint_path = os.path.join(self.output_dir, 'best_model.pt')
            elif is_final:
                checkpoint_path = os.path.join(self.output_dir, 'final_model.pt')
            elif is_epoch_end:
                checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
            elif step:
                checkpoint_path = os.path.join(self.output_dir, f'checkpoint_step_{step}.pt')
            else:
                checkpoint_path = os.path.join(self.output_dir, f'checkpoint.pt') # 备用

            torch.save(checkpoint, checkpoint_path)
            logger.info(f"检查点已保存到 {checkpoint_path}")

            # 同时保存配置以便轻松重新加载
            if is_best or is_final:
                 config_path = os.path.join(self.output_dir, 'config.json')
                 with open(config_path, 'w') as f:
                      json.dump(self.config, f, indent=4)
                 logger.info(f"配置已保存到 {config_path}")

        except Exception as e:
            logger.error(f"保存检查点失败: {e}")


# --- 未实现的关键部分 ---

"""
1.  **`StructureExtractor` 模型类**:
    *   **缺失内容**: 将 `StructureExtractor` 实现为流水线协调器。
    *   **位置**: 贯穿 `PretrainTrainer`，尤其是在 `__init__`, `_get_trainable_gnn_model`, `_train_epoch` (调用 `self.model.process_document`)。
    *   **操作**: 您需要实现这个类，协调 NER、RE、实体链接、图数据库更新等模块。

2.  **`ContinualGNNUpdater` 类**:
    *   **缺失内容**: 持续学习的核心逻辑：受影响节点检测 (使用引理 4.1)、记忆缓冲区管理 (分层重要性采样)、Fisher 信息计算、以及组合损失函数 (`L = L_new + L_data + L_model`) 的优化。
    *   **位置**: `_create_continual_updater`, `_train_epoch` (调用 `self.continual_updater.update`)。
    *   **操作**: 根据 ContinualGNN 论文第 4 节实现这个类。这是最实质性的部分。

3.  **`StructureExtractor` 内的集成**:
    *   **缺失内容**: `StructureExtractor` 需要方法来暴露其 GNN 模型 (`get_gnn_component`)、图数据库接口 (`get_graph_database`)，以及处理文档并返回 `delta_graph` 的方法 (`process_document`)。
    *   **位置**: `_get_trainable_gnn_model`, `_train_epoch` (调用 `self.model.process_document` 和 `self.model.get_graph_database`)。
    *   **操作**: 将这些方法添加到您的 `StructureExtractor` 实现中。

4.  **数据集 (`ReDocREDDataset`)**:
    *   **缺失内容**: 实际的数据集类，用于加载和提供预期格式 (例如 DocRED 格式) 的文档数据。
    *   **位置**: `__init__`, `_document_collate_fn`。
    *   **操作**: 实现或使用一个与您的数据格式兼容的现有数据集类。

5.  **评估 (`evaluate_model`)**:
    *   **缺失内容**: 函数，用于获取训练好的流水线并根据真实情况评估其图构建质量。
    *   **位置**: `_evaluate`。
    *   **操作**: 实现或最终确定您一直在研究的 `evaluate.py` 脚本，确保它能正确比较预测图和真实图。

6.  **图数据库交互**:
    *   **缺失内容**: 与 Neo4j 或内存图表示交互的具体代码。训练器和更新器需要查询和修改图。
    *   **位置**: `ContinualGNNUpdater` (用于受影响节点检测、获取邻居) 和 `StructureExtractor` (用于更新) 隐式需要它。调用如 `self.model.get_graph_database()` 暗示其存在。
    *   **操作**: 实现与您选择的图存储解决方案交互的实用程序或类。

7.  **混合精度和调度器细节**:
    *   **缺失内容**: 虽然结构存在，但 `scaler` 和 `scheduler` 在 `ContinualGNNUpdater.update` 调用中的确切集成需要完善。
    *   **位置**: `_train_epoch`。
    *   **操作**: 确保 `update` 方法正确使用传递的 `scaler` 进行 `autocast` 和 `scale.step()`/`scale.update()`，并在适用时调用 `scheduler.step()`。

"""

if __name__ == "__main__":
    # 这部分仅用于演示/测试结构。
    # 实际使用需要创建真实组件的实例。
    print("这是一个 PretrainTrainer 的框架。关键组件需要被实现。")