# evaluate.py

"""
Evaluation utilities for StructureExtractor-Pretrain.
Implements knowledge graph construction evaluation metrics and procedures.
This evaluator assesses the quality of the final knowledge graph built by the pipeline,
not individual model logits.

NOTE: This is the original evaluation script that has been adapted for the project.
The adapted version is in `evaluate_adapted.py` and should be used instead.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Set
import logging
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# --- 假设的外部模块导入 ---
# 这些模块需要您根据实际项目实现
from your_pipeline_module import build_knowledge_graph_from_document # 核心流水线函数
from your_data_utils import DocGraphDataset # 自定义数据集类

logger = logging.getLogger(__name__)

def evaluate_model(pipeline, # 流水线入口点，可以是模型配置或None
                   dataset: DocGraphDataset, 
                   config: Dict[str, Any],
                   batch_size: int = 1) -> Dict[str, float]:
    """
    评估 StructureExtractor-Pretrain 流水线构建知识图谱的质量。
    
    Args:
        pipeline: 流水线的入口点或配置 (在此评估中可能不直接使用模型参数)
        dataset: 包含文档和其真实图谱的 Dataset (例如 DocGraphDataset)
        config: 配置字典
        batch_size: 批处理大小（通常为1，因为每个文档独立处理）
        
    Returns:
        字典，包含图谱级别的评估指标
    """
    logger.info("Starting evaluation of StructureExtractor-Pretrain pipeline...")
    
    # 初始化累加器，用于计算平均指标
    total_metrics = defaultdict(float)
    num_documents = 0
    successful_documents = 0

    # 创建数据加载器
    # 注意：batch_size=1 是推荐设置，因为每个文档是独立处理的单元
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        # 使用默认的 collate_fn，它会将一个 batch (大小为1) 的样本列表合并
        # 例如，[sample] -> sample
    )

    evaluation_config = config.get('evaluation', {})
    metrics_to_compute = evaluation_config.get('metrics', [
        'entity_precision', 'entity_recall', 'entity_f1',
        'relation_precision', 'relation_recall', 'relation_f1',
        'graph_similarity'
    ])

    with tqdm(dataloader, desc="Evaluating Documents") as pbar:
        for batch_idx, batch in enumerate(pbar):
            # 从 batch 中提取单个文档数据
            # 假设 batch 是一个字典，包含 'document_data' 键
            # 或者如果 batch_size=1, batch 本身可能就是文档数据字典
            if isinstance(batch, list) and len(batch) == 1:
                document_data = batch[0]
            elif isinstance(batch, dict):
                document_data = batch
            else:
                logger.error(f"Unexpected batch format at index {batch_idx}: {type(batch)}")
                continue

            try:
                # --- 核心步骤 1: 使用您的流水线构建预测图谱 ---
                # 这会调用您系统中的所有组件（NER, RE, 实体链接, 图数据库更新等）
                predicted_graph = build_knowledge_graph_from_document(document_data)
                logger.debug(f"Built predicted graph for document {batch_idx}")

                # --- 核心步骤 2: 从真实数据中提取真实图谱 ---
                # 将您的数据格式转换为一个真实的图谱对象
                ground_truth_graph = build_ground_truth_graph(document_data)
                logger.debug(f"Built ground truth graph for document {batch_idx}")

                # --- 核心步骤 3: 评估两个图谱 ---
                # 调用图谱比较函数，计算指标
                metrics = evaluate_graphs(predicted_graph, ground_truth_graph, metrics_to_compute)
                logger.debug(f"Computed metrics for document {batch_idx}: {metrics}")

                # 累加指标
                for key, value in metrics.items():
                    total_metrics[key] += value
                successful_documents += 1

            except Exception as e:
                logger.error(f"Error processing document at index {batch_idx}: {e}", exc_info=True)
                # 可以选择跳过错误文档或停止评估
                # continue # 跳过，继续下一个
                # break # 遇到错误则停止评估

            num_documents += 1
            # 更新进度条后缀，显示当前平均指标
            if successful_documents > 0:
                avg_metrics = {k: v / successful_documents for k, v in total_metrics.items()}
                pbar.set_postfix(avg_metrics)

    # --- 计算最终平均值 ---
    final_results = {}
    if successful_documents > 0:
        for key in total_metrics:
            final_results[key] = total_metrics[key] / successful_documents
    else:
        logger.warning("No documents were successfully processed for evaluation.")
        # 返回0.0作为所有指标的默认值
        for metric in metrics_to_compute:
            final_results[metric] = 0.0

    final_results['successful_documents'] = successful_documents
    final_results['total_documents'] = num_documents

    logger.info(f"Evaluation completed. Successful: {successful_documents}/{num_documents}")
    logger.info(f"Final Results: {final_results}")
    
    return final_results


def build_ground_truth_graph(document_data: Dict) -> Dict:
    """
    根据您提供的数据格式，构建一个真实的知识图谱。
    输出格式应与 build_knowledge_graph_from_document 的输出一致。
    
    数据格式:
    {
      'title',
      'sents': [[word in sent 0], [word in sent 1], ...]
      'vertexSet': [
                      [
                        { 'name': mention_name, 
                          'sent_id': mention in which sentence, 
                          'pos': position of mention in a sentence, 
                          'type': NER_type}
                        {another mention}
                      ], 
                      [another entity]
                   ]
      'labels':   [
                    {
                      'h': idx of head entity in vertexSet,
                      't': idx of tail entity in vertexSet,
                      'r': relation,
                      'evidence': evidence sentences' id
                    }
                  ]
    }
    
    Args:
        document_data: 一个文档的数据，格式如上所示
        
    Returns:
        一个字典，表示真实的知识图谱，例如：
        {
          "nodes": [{"id": "ent_0", "name": "Alice", "type": "Person"}, ...],
          "edges": [{"source": "ent_0", "target": "ent_1", "type": "WORKS_FOR"}, ...]
        }
    """
    nodes = []
    node_id_to_vertex_idx = {} # 辅助映射，用于后续构建边
    edges = []
    
    # 1. 从 vertexSet 构建节点
    for entity_idx, mentions in enumerate(document_data['vertexSet']):
        # 为每个实体创建一个节点
        # 使用第一个提及的名字作为节点名（或采用其他策略，如主要提及）
        primary_mention = mentions[0] # 简化处理，取第一个
        node_name = primary_mention['name']
        node_type = primary_mention['type'] # 假设同一实体的所有提及类型相同
        
        node_id = f"gt_ent_{entity_idx}" # 使用 gt_ 前缀区分 ground truth ID
        node = {
            "id": node_id,
            "name": node_name,
            "type": node_type,
            # 可以选择性地保留提及信息用于更复杂的评估
            # "mentions": mentions 
        }
        nodes.append(node)
        node_id_to_vertex_idx[entity_idx] = node_id # 建立映射
    
    # 2. 从 labels 构建边
    for rel_idx, rel in enumerate(document_data['labels']):
        head_vertex_idx = rel['h']
        tail_vertex_idx = rel['t']
        relation_type = rel['r']
        
        # 使用映射获取节点ID
        head_id = node_id_to_vertex_idx.get(head_vertex_idx)
        tail_id = node_id_to_vertex_idx.get(tail_vertex_idx)
        
        # 只有当头尾实体都存在时才添加边
        if head_id is not None and tail_id is not None:
            edge_id = f"gt_rel_{rel_idx}"
            edge = {
                "id": edge_id, # 为边也分配ID，便于评估
                "source": head_id,
                "target": tail_id,
                "type": relation_type,
                "evidence": rel.get('evidence', [])
            }
            edges.append(edge)
        else:
            logger.warning(f"Relation {rel} references non-existent entity index. Skipping.")
    
    logger.debug(f"Built ground truth graph with {len(nodes)} nodes and {len(edges)} edges.")
    return {"nodes": nodes, "edges": edges, "type": "ground_truth"}


def evaluate_graphs(predicted_graph: Dict, ground_truth_graph: Dict, metrics_to_compute: List[str]) -> Dict[str, float]:
    """
    比较预测图谱和真实图谱，计算评估指标。
    
    Args:
        predicted_graph: 模型预测的图谱 (来自 build_knowledge_graph_from_document)
        ground_truth_graph: 真实的图谱 (来自 build_ground_truth_graph)
        metrics_to_compute: 需要计算的指标列表
        
    Returns:
        包含评估指标的字典
    """
    # --- 1. 提取节点和边的集合用于比较 ---
    
    # 节点比较：基于ID (假设ID系统能正确对齐预测和真实的实体)
    # 在更复杂的场景下，可能需要基于名称或嵌入进行实体对齐
    pred_nodes_dict = {node['id']: node for node in predicted_graph.get('nodes', [])}
    gt_nodes_dict = {node['id'].replace('gt_', ''): node for node in ground_truth_graph.get('nodes', [])} # 简化对齐
    
    # 为了精确比较，我们只比较ID能对上的节点（忽略预测中多出来的或缺少的）
    # 这里采用一种更宽松的比较方式：预测ID去掉可能的前缀后与真实ID比较
    aligned_pred_nodes = set()
    aligned_gt_nodes = set()
    for pred_id in pred_nodes_dict:
        # 假设预测ID格式为 'pred_ent_X' 或 'ent_X'，真实ID为 'gt_ent_X'
        # 简单处理：去掉前缀 'pred_' 和 'gt_'
        clean_pred_id = pred_id.replace('pred_', '').replace('gt_', '')
        if clean_pred_id in gt_nodes_dict:
             aligned_pred_nodes.add(pred_id)
             aligned_gt_nodes.add(gt_nodes_dict[clean_pred_id]['id']) # 'gt_ent_X'

    # 边比较：基于 (source, target, type) 三元组
    def _get_edge_key(edge):
        # 标准化边的表示，用于比较
        # 注意：根据任务定义，关系可能是有向的
        return (edge['source'], edge['target'], edge['type'])

    pred_edges = predicted_graph.get('edges', [])
    gt_edges = ground_truth_graph.get('edges', [])
    
    # 同样，对边的source和target ID进行标准化处理
    standardized_pred_edges = set()
    for e in pred_edges:
        # 假设预测边的source/target ID也需要标准化
        std_src = e['source'].replace('pred_', '').replace('gt_', '')
        std_tgt = e['target'].replace('pred_', '').replace('gt_', '')
        standardized_pred_edges.add((std_src, std_tgt, e['type']))

    standardized_gt_edges = set()
    for e in gt_edges:
        std_src = e['source'].replace('gt_', '')
        std_tgt = e['target'].replace('gt_', '')
        standardized_gt_edges.add((std_src, std_tgt, e['type']))

    # --- 2. 计算实体（节点）级别的指标 ---
    tp_nodes = len(aligned_pred_nodes) # 简化处理，认为对齐的都是对的
    # 更精确的方法需要比较对齐节点的属性（如name, type）
    # 这里我们假设ID对齐即为正确
    fp_nodes = len(pred_nodes_dict) - len(aligned_pred_nodes)
    fn_nodes = len(gt_nodes_dict) - len(aligned_gt_nodes)
    
    precision_nodes = tp_nodes / (tp_nodes + fp_nodes) if (tp_nodes + fp_nodes) > 0 else 0.0
    recall_nodes = tp_nodes / (tp_nodes + fn_nodes) if (tp_nodes + fn_nodes) > 0 else 0.0
    f1_nodes = 2 * (precision_nodes * recall_nodes) / (precision_nodes + recall_nodes) if (precision_nodes + recall_nodes) > 0 else 0.0
    
    # --- 3. 计算关系（边）级别的指标 ---
    tp_edges = len(standardized_pred_edges & standardized_gt_edges)
    fp_edges = len(standardized_pred_edges - standardized_gt_edges)
    fn_edges = len(standardized_gt_edges - standardized_pred_edges)
    
    precision_rels = tp_edges / (tp_edges + fp_edges) if (tp_edges + fp_edges) > 0 else 0.0
    recall_rels = tp_edges / (tp_edges + fn_edges) if (tp_edges + fn_edges) > 0 else 0.0
    f1_rels = 2 * (precision_rels * recall_rels) / (precision_rels + recall_rels) if (precision_rels + recall_rels) > 0 else 0.0
    
    # --- 4. 计算图相似度 (示例: 加权F1) ---
    graph_sim = 0.5 * f1_nodes + 0.5 * f1_rels

    # --- 5. 组装并返回结果 ---
    all_metrics = {
        'entity_precision': precision_nodes,
        'entity_recall': recall_nodes,
        'entity_f1': f1_nodes,
        'relation_precision': precision_rels,
        'relation_recall': recall_rels,
        'relation_f1': f1_rels,
        'graph_similarity': graph_sim,
        # 可以添加更多指标，如实体类型准确率、关系类型准确率等
    }
    
    # 只返回配置中要求的指标
    final_metrics = {k: v for k, v in all_metrics.items() if k in metrics_to_compute}
    
    logger.debug(f"Graph evaluation results: {final_metrics}")
    return final_metrics

# --- 旧的、不适用的代码可以删除 ---
# 以下函数是为旧的 logits 评估设计的，与新的图谱评估范式不匹配，应删除
#
# def _evaluation_collate_fn(batch: List[Dict]) -> Dict[str, Any]: ...
# def _compute_evaluation_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor: ...
# def _create_entity_targets(...): ...
# def _create_relation_targets(...): ...
# --- End of Old Code ---