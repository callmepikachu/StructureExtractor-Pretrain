"""
ReDocRED Dataset implementation for StructureExtractor-Pretrain.
Handles loading and processing of ReDocRED data for document-level relation extraction.
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ReDocREDDataset(Dataset):
    """
    ReDocRED dataset for document-level relation extraction.
    Processes documents into chunks suitable for incremental graph construction.
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer_name: str = "bert-base-uncased",
                 max_seq_length: int = 512,
                 max_entities: int = 100,
                 max_relations: int = 50,
                 chunk_size: int = 3,  # Number of sentences per chunk
                 overlap_size: int = 1):  # Overlap between chunks
        
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load and process data
        self.data = self._load_data()
        self.processed_data = self._process_data()
        
        logger.info(f"Loaded {len(self.processed_data)} document chunks from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load ReDocRED data from JSON file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            return []
    
    def _process_data(self) -> List[Dict[str, Any]]:
        """Process raw ReDocRED data into chunks suitable for training."""
        processed_chunks = []
        
        for doc_idx, doc in enumerate(self.data):
            try:
                # Extract document information
                doc_id = doc.get('title', f'doc_{doc_idx}')
                sentences = doc.get('sents', [])
                entities = doc.get('vertexSet', [])
                relations = doc.get('labels', [])
                
                # Create chunks from sentences
                chunks = self._create_chunks(sentences)
                
                # Process each chunk
                for chunk_idx, chunk_sentences in enumerate(chunks):
                    chunk_data = self._process_chunk(
                        doc_id, chunk_idx, chunk_sentences, entities, relations
                    )
                    if chunk_data:
                        processed_chunks.append(chunk_data)
                        
            except Exception as e:
                logger.warning(f"Error processing document {doc_idx}: {e}")
                continue
        
        return processed_chunks
    
    def _create_chunks(self, sentences: List[List[str]]) -> List[List[List[str]]]:
        """Create overlapping chunks from document sentences."""
        if not sentences:
            return []
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(sentences):
            end_idx = min(start_idx + self.chunk_size, len(sentences))
            chunk = sentences[start_idx:end_idx]
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.overlap_size
            if start_idx >= len(sentences):
                break
        
        return chunks
    
    def _process_chunk(self, 
                      doc_id: str, 
                      chunk_idx: int,
                      chunk_sentences: List[List[str]], 
                      entities: List[List[Dict]], 
                      relations: List[Dict]) -> Optional[Dict[str, Any]]:
        """Process a single chunk into training format."""
        
        # Flatten sentences into text
        chunk_text = ""
        sentence_offsets = []
        current_offset = 0
        
        for sent_idx, sentence in enumerate(chunk_sentences):
            sentence_text = " ".join(sentence) + " "
            sentence_offsets.append((current_offset, current_offset + len(sentence_text)))
            chunk_text += sentence_text
            current_offset += len(sentence_text)
        
        # Tokenize chunk
        tokenized = self.tokenizer(
            chunk_text.strip(),
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Extract entities in this chunk
        chunk_entities = self._extract_chunk_entities(
            entities, chunk_sentences, sentence_offsets
        )
        
        # Extract relations in this chunk
        chunk_relations = self._extract_chunk_relations(
            relations, chunk_entities
        )
        
        return {
            'doc_id': doc_id,
            'chunk_idx': chunk_idx,
            'text': chunk_text.strip(),
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'offset_mapping': tokenized['offset_mapping'].squeeze(0),
            'entities': chunk_entities,
            'relations': chunk_relations,
            'sentence_offsets': sentence_offsets
        }
    
    def _extract_chunk_entities(self, 
                               entities: List[List[Dict]], 
                               chunk_sentences: List[List[str]],
                               sentence_offsets: List[Tuple[int, int]]) -> List[Dict]:
        """Extract entities that appear in the current chunk."""
        chunk_entities = []
        
        for entity_idx, entity_mentions in enumerate(entities):
            for mention in entity_mentions:
                sent_id = mention.get('sent_id', -1)
                
                # Check if entity is in current chunk
                if 0 <= sent_id < len(chunk_sentences):
                    pos = mention.get('pos', [])
                    if len(pos) >= 2:
                        # Calculate character positions
                        sent_start, sent_end = sentence_offsets[sent_id]
                        
                        # Get mention text
                        sentence = chunk_sentences[sent_id]
                        mention_tokens = sentence[pos[0]:pos[1]]
                        mention_text = " ".join(mention_tokens)
                        
                        chunk_entities.append({
                            'id': f'entity_{entity_idx}',
                            'text': mention_text,
                            'type': mention.get('type', 'ENTITY'),
                            'sent_id': sent_id,
                            'pos': pos,
                            'char_start': sent_start,
                            'char_end': sent_end
                        })
        
        return chunk_entities[:self.max_entities]  # Limit number of entities
    
    def _extract_chunk_relations(self, 
                                relations: List[Dict], 
                                chunk_entities: List[Dict]) -> List[Dict]:
        """Extract relations between entities in the current chunk."""
        chunk_relations = []
        entity_ids = {entity['id'] for entity in chunk_entities}
        
        for relation in relations:
            head_id = f"entity_{relation.get('h', -1)}"
            tail_id = f"entity_{relation.get('t', -1)}"
            
            # Only include relations where both entities are in chunk
            if head_id in entity_ids and tail_id in entity_ids:
                chunk_relations.append({
                    'head': head_id,
                    'tail': tail_id,
                    'type': relation.get('r', 'UNKNOWN'),
                    'evidence': relation.get('evidence', [])
                })
        
        return chunk_relations[:self.max_relations]  # Limit number of relations
    
    def __len__(self) -> int:
        """Return number of chunks in dataset."""
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single chunk by index."""
        if idx >= len(self.processed_data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.processed_data)}")
        
        return self.processed_data[idx]
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        return [chunk for chunk in self.processed_data if chunk['doc_id'] == doc_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_entities = sum(len(chunk['entities']) for chunk in self.processed_data)
        total_relations = sum(len(chunk['relations']) for chunk in self.processed_data)
        
        entity_types = defaultdict(int)
        relation_types = defaultdict(int)
        
        for chunk in self.processed_data:
            for entity in chunk['entities']:
                entity_types[entity['type']] += 1
            for relation in chunk['relations']:
                relation_types[relation['type']] += 1
        
        return {
            'num_chunks': len(self.processed_data),
            'total_entities': total_entities,
            'total_relations': total_relations,
            'avg_entities_per_chunk': total_entities / len(self.processed_data) if self.processed_data else 0,
            'avg_relations_per_chunk': total_relations / len(self.processed_data) if self.processed_data else 0,
            'entity_types': dict(entity_types),
            'relation_types': dict(relation_types)
        }
