"""
StructureExtractor: Core model implementing ContinualGNN and StreamE optimization techniques
for long document graph construction and knowledge extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data, Batch
from transformers import AutoModel, AutoTokenizer
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import copy


class MemoryBuffer:
    """Memory buffer for storing important nodes using hierarchical-importance sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.nodes = {}  # node_id -> (embedding, importance, cluster)
        self.cluster_counts = defaultdict(int)
        
    def add_with_reservoir_sampling(self, node_id: str, embedding: torch.Tensor, 
                                  importance: float, cluster: str):
        """Add node using reservoir sampling with importance weighting."""
        if len(self.nodes) < self.capacity:
            self.nodes[node_id] = (embedding, importance, cluster)
            self.cluster_counts[cluster] += 1
        else:
            # Reservoir sampling with importance bias
            min_importance_node = min(self.nodes.keys(), 
                                    key=lambda x: self.nodes[x][1])
            min_importance = self.nodes[min_importance_node][1]
            
            if importance > min_importance:
                old_cluster = self.nodes[min_importance_node][2]
                self.cluster_counts[old_cluster] -= 1
                del self.nodes[min_importance_node]
                
                self.nodes[node_id] = (embedding, importance, cluster)
                self.cluster_counts[cluster] += 1
    
    def get_nodes(self) -> List[str]:
        """Get all node IDs in memory buffer."""
        return list(self.nodes.keys())
    
    def get_node_data(self, node_id: str) -> Tuple[torch.Tensor, float, str]:
        """Get node embedding, importance, and cluster."""
        return self.nodes[node_id]


class FisherMatrix:
    """Fisher Information Matrix for EWC regularization."""
    
    def __init__(self, model_params: Dict[str, torch.Tensor]):
        self.fisher_info = {}
        for name, param in model_params.items():
            self.fisher_info[name] = torch.zeros_like(param)
    
    def update(self, model: nn.Module, data_loader, device: str):
        """Update Fisher information matrix using current data."""
        model.eval()
        self.fisher_info = {name: torch.zeros_like(param) 
                           for name, param in model.named_parameters()}
        
        for batch in data_loader:
            model.zero_grad()
            output = model(batch)
            loss = F.cross_entropy(output.logits, batch.labels)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.fisher_info[name] += param.grad.data ** 2
        
        # Normalize by number of samples
        num_samples = len(data_loader.dataset)
        for name in self.fisher_info:
            self.fisher_info[name] /= num_samples


class GraphStructure:
    """Graph structure management using NetworkX."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_embeddings = {}
        self.node_attributes = {}
        
    def add_node(self, node_id: str, attributes: Dict[str, Any] = None, 
                 embedding: torch.Tensor = None):
        """Add node to graph."""
        self.graph.add_node(node_id)
        if attributes:
            self.node_attributes[node_id] = attributes
        if embedding is not None:
            self.node_embeddings[node_id] = embedding
    
    def add_edge(self, head_id: str, tail_id: str, relation_type: str):
        """Add edge to graph."""
        self.graph.add_edge(head_id, tail_id, type=relation_type)
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return self.graph.has_node(node_id)
    
    def get_neighbors(self, node_id: str, hops: int = 1) -> Set[str]:
        """Get k-hop neighbors of a node."""
        if not self.has_node(node_id):
            return set()
        
        neighbors = set([node_id])
        current_level = set([node_id])
        
        for _ in range(hops):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.neighbors(node))
                next_level.update(self.graph.predecessors(node))
            current_level = next_level - neighbors
            neighbors.update(current_level)
        
        return neighbors - set([node_id])
    
    def update_node_embedding(self, node_id: str, embedding: torch.Tensor):
        """Update node embedding."""
        self.node_embeddings[node_id] = embedding
    
    def get_node_embedding(self, node_id: str) -> Optional[torch.Tensor]:
        """Get node embedding."""
        return self.node_embeddings.get(node_id)


class StructureExtractor(nn.Module):
    """
    StructureExtractor model with ContinualGNN and StreamE optimization.
    Implements the complete framework from the technical specification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.hidden_dim = config['model']['hidden_dim']
        self.gnn_hidden_dim = config['model']['gnn_hidden_dim']
        self.num_layers = config['model']['num_layers']
        
        # ContinualGNN parameters
        self.memory_capacity = config['model']['memory_capacity']
        self.regularization_weight = config['model']['regularization_weight']
        self.influence_threshold = config['model']['influence_threshold']
        self.detection_hop = config['model']['detection_hop']
        
        # Initialize components
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GATConv(self.hidden_dim if i == 0 else self.gnn_hidden_dim, 
                   self.gnn_hidden_dim, heads=4, concat=False)
            for i in range(config['model']['gnn_layers'])
        ])
        
        # Output layers
        self.entity_classifier = nn.Linear(self.gnn_hidden_dim, config['data']['max_entities'])
        self.relation_classifier = nn.Linear(self.gnn_hidden_dim * 2, config['data']['max_relations'])
        
        # Memory and regularization components
        self.memory_buffer = MemoryBuffer(self.memory_capacity)
        self.fisher_matrix = None
        self.previous_params = None
        
        # Graph structure
        self.graph = GraphStructure()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_entities_and_relations(self, text_chunk: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relations from text chunk using pretrained models.
        Returns: (entities, relations)
        """
        # Tokenize input text
        inputs = self.tokenizer(text_chunk, return_tensors='pt',
                               max_length=512, truncation=True, padding=True)

        # Get contextual embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            hidden_states = outputs.last_hidden_state

        # Simple entity extraction (in practice, use specialized NER models)
        entities = []
        relations = []

        # Placeholder implementation - replace with actual NER/RE models
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Mock entity extraction
        for i, token in enumerate(tokens):
            if token.startswith('##'):
                continue
            if token.isupper() or token.istitle():
                entities.append({
                    'id': f'entity_{len(entities)}',
                    'text': token,
                    'type': 'ENTITY',
                    'start': i,
                    'end': i + 1,
                    'embedding': hidden_states[0, i].clone()
                })

        # Mock relation extraction
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if j - i <= 5:  # Only consider nearby entities
                    relations.append({
                        'head': entities[i]['id'],
                        'tail': entities[j]['id'],
                        'type': 'RELATED_TO',
                        'confidence': 0.8
                    })

        return entities, relations

    def build_incremental_graph(self, entities: List[Dict], relations: List[Dict]) -> Dict:
        """
        Build incremental graph ΔG_k from extracted entities and relations.
        Returns: delta_graph = {'nodes': set, 'edges': set, 'attributes': dict}
        """
        delta_nodes = set()
        delta_edges = set()
        delta_attributes = {}

        # Process entities
        for entity in entities:
            # Entity linking (simplified)
            linked_entity = self.entity_linking(entity)

            if linked_entity['is_new']:
                delta_nodes.add(linked_entity['id'])
                delta_attributes[linked_entity['id']] = {
                    'text': entity['text'],
                    'type': entity['type'],
                    'embedding': entity['embedding']
                }
            else:
                # Update existing entity attributes if needed
                old_attrs = self.graph.node_attributes.get(linked_entity['id'], {})
                new_attrs = self._update_attributes(old_attrs, entity)
                if old_attrs != new_attrs:
                    delta_attributes[linked_entity['id']] = new_attrs

        # Process relations
        for relation in relations:
            head_id = relation['head']
            tail_id = relation['tail']
            rel_type = relation['type']

            delta_edges.add((head_id, rel_type, tail_id))

        return {
            'nodes': delta_nodes,
            'edges': delta_edges,
            'attributes': delta_attributes
        }

    def entity_linking(self, candidate_entity: Dict) -> Dict:
        """
        Perform entity linking to determine if entity is new or existing.
        Returns: {'id': str, 'is_new': bool}
        """
        # Simplified entity linking - in practice use BLINK or similar
        entity_text = candidate_entity['text'].lower()

        # Check if similar entity exists in graph
        for node_id in self.graph.graph.nodes():
            node_attrs = self.graph.node_attributes.get(node_id, {})
            if 'text' in node_attrs:
                if node_attrs['text'].lower() == entity_text:
                    return {'id': node_id, 'is_new': False}

        # Create new entity ID
        new_id = f"entity_{len(self.graph.graph.nodes())}"
        return {'id': new_id, 'is_new': True}

    def _update_attributes(self, old_attrs: Dict, entity: Dict) -> Dict:
        """Update entity attributes with new information."""
        new_attrs = old_attrs.copy()

        # Update embedding with moving average
        if 'embedding' in old_attrs and 'embedding' in entity:
            alpha = 0.1  # Update rate
            new_attrs['embedding'] = (1 - alpha) * old_attrs['embedding'] + alpha * entity['embedding']
        elif 'embedding' in entity:
            new_attrs['embedding'] = entity['embedding']

        # Update other attributes
        for key in ['text', 'type']:
            if key in entity:
                new_attrs[key] = entity[key]

        return new_attrs

    def detect_influenced_nodes(self, delta_graph: Dict) -> Set[str]:
        """
        Detect nodes influenced by the incremental graph changes.
        Implements ContinualGNN's approximate influence detection algorithm.
        """
        influenced_nodes = set()

        # Process attribute changes
        for node_id, new_attrs in delta_graph['attributes'].items():
            if node_id in self.graph.node_attributes:
                old_attrs = self.graph.node_attributes[node_id]

                # Calculate attribute change vector
                if 'embedding' in old_attrs and 'embedding' in new_attrs:
                    delta_x = new_attrs['embedding'] - old_attrs['embedding']

                    # Get k-hop neighbors
                    neighbors = self.graph.get_neighbors(node_id, self.detection_hop)

                    # Approximate influence calculation
                    for neighbor_id in neighbors:
                        # Simplified influence score based on graph distance
                        distance = self._graph_distance(node_id, neighbor_id)
                        if distance > 0:
                            influence_score = torch.norm(delta_x) / (distance ** 2)
                            if influence_score > self.influence_threshold:
                                influenced_nodes.add(neighbor_id)

        # Process structural changes (new nodes and edges)
        for node_id in delta_graph['nodes']:
            # New nodes influence their immediate neighbors
            if node_id in self.graph.graph.nodes():
                neighbors = self.graph.get_neighbors(node_id, 1)
                influenced_nodes.update(neighbors)

        for head_id, rel_type, tail_id in delta_graph['edges']:
            # New edges influence both endpoints and their neighbors
            for node_id in [head_id, tail_id]:
                if node_id in self.graph.graph.nodes():
                    neighbors = self.graph.get_neighbors(node_id, 1)
                    influenced_nodes.update(neighbors)
                    influenced_nodes.add(node_id)

        return influenced_nodes

    def _graph_distance(self, node1: str, node2: str) -> int:
        """Calculate shortest path distance between two nodes."""
        try:
            return nx.shortest_path_length(self.graph.graph, node1, node2)
        except nx.NetworkXNoPath:
            return float('inf')

    def update_graph_and_model(self, delta_graph: Dict, influenced_nodes: Set[str]):
        """
        Core update function implementing dual-perspective knowledge consolidation.
        Combines data replay and model regularization.
        """
        # Get replay nodes from memory buffer
        replay_nodes = self.memory_buffer.get_nodes()

        # Update Fisher information matrix if needed
        if self.fisher_matrix is None:
            self.fisher_matrix = FisherMatrix({name: param for name, param in self.named_parameters()})

        # Define loss function components
        def compute_total_loss():
            total_loss = 0.0

            # 1. New knowledge loss (on influenced nodes)
            if influenced_nodes:
                new_loss = self._compute_node_loss(influenced_nodes)
                total_loss += new_loss

            # 2. Historical knowledge loss (data replay)
            if replay_nodes:
                replay_loss = self._compute_node_loss(replay_nodes)
                total_loss += replay_loss

            # 3. Model regularization loss (EWC)
            if self.previous_params is not None:
                reg_loss = self._compute_regularization_loss()
                total_loss += self.regularization_weight * reg_loss

            return total_loss

        # Optimize the model
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        for epoch in range(5):  # Few optimization steps
            optimizer.zero_grad()
            loss = compute_total_loss()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

        # Update node embeddings for influenced nodes
        self._update_node_embeddings(influenced_nodes)

        # Save current parameters for next iteration
        self.previous_params = {name: param.clone().detach()
                              for name, param in self.named_parameters()}

    def _compute_node_loss(self, node_ids: List[str]) -> torch.Tensor:
        """Compute loss for a set of nodes."""
        if not node_ids:
            return torch.tensor(0.0, requires_grad=True)

        total_loss = torch.tensor(0.0, requires_grad=True)

        for node_id in node_ids:
            if node_id in self.graph.node_embeddings:
                # Get node embedding and compute a simple reconstruction loss
                embedding = self.graph.node_embeddings[node_id]

                # Simple autoencoder-style loss (placeholder)
                reconstructed = self.entity_classifier(embedding)
                target = torch.zeros_like(reconstructed)  # Placeholder target

                loss = F.mse_loss(reconstructed, target)
                total_loss = total_loss + loss

        return total_loss

    def _compute_regularization_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        reg_loss = torch.tensor(0.0)

        for name, param in self.named_parameters():
            if name in self.previous_params and name in self.fisher_matrix.fisher_info:
                fisher_info = self.fisher_matrix.fisher_info[name]
                prev_param = self.previous_params[name]
                reg_loss += (fisher_info * (param - prev_param) ** 2).sum()

        return reg_loss

    def _update_node_embeddings(self, node_ids: Set[str]):
        """Update embeddings for specified nodes using current model."""
        for node_id in node_ids:
            if node_id in self.graph.graph.nodes():
                # Get node features and run through GNN
                node_features = self._get_node_features(node_id)
                if node_features is not None:
                    new_embedding = self._forward_gnn(node_features, node_id)
                    self.graph.update_node_embedding(node_id, new_embedding)

    def _get_node_features(self, node_id: str) -> Optional[torch.Tensor]:
        """Get features for a node."""
        if node_id in self.graph.node_attributes:
            attrs = self.graph.node_attributes[node_id]
            if 'embedding' in attrs:
                return attrs['embedding']
        return None

    def _forward_gnn(self, node_features: torch.Tensor, node_id: str) -> torch.Tensor:
        """Forward pass through GNN for a single node."""
        # Create a subgraph around the node for GNN processing
        neighbors = self.graph.get_neighbors(node_id, 2)
        subgraph_nodes = list(neighbors) + [node_id]

        # Get features for all nodes in subgraph
        x = []
        node_mapping = {}
        for i, nid in enumerate(subgraph_nodes):
            features = self._get_node_features(nid)
            if features is not None:
                x.append(features)
                node_mapping[nid] = i
            else:
                x.append(torch.zeros(self.hidden_dim))
                node_mapping[nid] = i

        if not x:
            return torch.zeros(self.gnn_hidden_dim)

        x = torch.stack(x)

        # Create edge index for subgraph
        edge_index = []
        for edge in self.graph.graph.edges():
            if edge[0] in node_mapping and edge[1] in node_mapping:
                edge_index.append([node_mapping[edge[0]], node_mapping[edge[1]]])

        if not edge_index:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index).t().contiguous()

        # Forward through GNN layers
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, edge_index))

        # Return embedding for target node
        target_idx = node_mapping[node_id]
        return x[target_idx]

    def merge_delta_graph_into_main_graph(self, delta_graph: Dict):
        """Merge incremental graph into main graph structure."""
        # Add new nodes
        for node_id in delta_graph['nodes']:
            if not self.graph.has_node(node_id):
                attrs = delta_graph['attributes'].get(node_id, {})
                embedding = attrs.get('embedding')
                self.graph.add_node(node_id, attrs, embedding)

        # Add new edges
        for head_id, rel_type, tail_id in delta_graph['edges']:
            self.graph.add_edge(head_id, tail_id, rel_type)

        # Update node attributes
        for node_id, attrs in delta_graph['attributes'].items():
            if self.graph.has_node(node_id):
                self.graph.node_attributes[node_id].update(attrs)
                if 'embedding' in attrs:
                    self.graph.update_node_embedding(node_id, attrs['embedding'])

    def update_memory_buffer(self, influenced_nodes: Set[str]):
        """Update memory buffer using hierarchical-importance sampling."""
        for node_id in influenced_nodes:
            if node_id in self.graph.node_embeddings:
                # Calculate node importance
                importance = self._calculate_node_importance(node_id)

                # Get node cluster (simplified)
                cluster = self._get_node_cluster(node_id)

                # Add to memory buffer
                embedding = self.graph.node_embeddings[node_id]
                self.memory_buffer.add_with_reservoir_sampling(
                    node_id, embedding, importance, cluster
                )

    def _calculate_node_importance(self, node_id: str) -> float:
        """Calculate node importance based on neighborhood diversity."""
        neighbors = list(self.graph.get_neighbors(node_id, 1))
        if not neighbors:
            return 0.0

        # Simple importance: ratio of neighbors with different types
        node_type = self.graph.node_attributes.get(node_id, {}).get('type', 'UNKNOWN')
        different_count = 0

        for neighbor_id in neighbors:
            neighbor_type = self.graph.node_attributes.get(neighbor_id, {}).get('type', 'UNKNOWN')
            if neighbor_type != node_type:
                different_count += 1

        return different_count / len(neighbors)

    def _get_node_cluster(self, node_id: str) -> str:
        """Get cluster/community for a node (simplified)."""
        node_type = self.graph.node_attributes.get(node_id, {}).get('type', 'UNKNOWN')
        return node_type

    def process_document_stream(self, document_chunks: List[str]) -> GraphStructure:
        """
        Main function: Process document stream and build knowledge graph.
        Implements the complete framework from the technical specification.
        """
        for k, chunk in enumerate(document_chunks):
            print(f"Processing chunk {k+1}/{len(document_chunks)}")

            # 1. Extract entities and relations
            entities, relations = self.extract_entities_and_relations(chunk)

            # 2. Build incremental graph
            delta_graph = self.build_incremental_graph(entities, relations)

            # 3. Detect influenced nodes
            influenced_nodes = self.detect_influenced_nodes(delta_graph)

            # 4. Update graph and model with dual-perspective consolidation
            self.update_graph_and_model(delta_graph, influenced_nodes)

            # 5. Merge incremental graph into main graph
            self.merge_delta_graph_into_main_graph(delta_graph)

            # 6. Update memory buffer
            self.update_memory_buffer(influenced_nodes)

        return self.graph

    def forward(self, input_text: str) -> Dict[str, torch.Tensor]:
        """Forward pass for training/inference."""
        # Extract entities and relations
        entities, relations = self.extract_entities_and_relations(input_text)

        # Build graph representation
        delta_graph = self.build_incremental_graph(entities, relations)

        # Get entity and relation predictions
        entity_logits = []
        relation_logits = []

        for entity in entities:
            if 'embedding' in entity:
                entity_pred = self.entity_classifier(entity['embedding'])
                entity_logits.append(entity_pred)

        for relation in relations:
            head_emb = self._get_entity_embedding(relation['head'], entities)
            tail_emb = self._get_entity_embedding(relation['tail'], entities)
            if head_emb is not None and tail_emb is not None:
                rel_input = torch.cat([head_emb, tail_emb], dim=-1)
                rel_pred = self.relation_classifier(rel_input)
                relation_logits.append(rel_pred)

        return {
            'entity_logits': torch.stack(entity_logits) if entity_logits else torch.empty(0),
            'relation_logits': torch.stack(relation_logits) if relation_logits else torch.empty(0),
            'entities': entities,
            'relations': relations
        }

    def _get_entity_embedding(self, entity_id: str, entities: List[Dict]) -> Optional[torch.Tensor]:
        """Get embedding for entity by ID."""
        for entity in entities:
            if entity['id'] == entity_id and 'embedding' in entity:
                return entity['embedding']
        return None
