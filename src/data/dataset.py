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
import time

logger = logging.getLogger(__name__)


class ReDocREDDataset(Dataset):
    """
    ReDocRED dataset for document-level relation extraction.
    Processes documents into chunks suitable for incremental graph construction.
    Implements lazy loading to reduce memory usage.
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer_name: str = "bert-base-uncased",
                 max_seq_length: int = 512,
                 max_entities: int = 100,
                 max_relations: int = 50,
                 chunk_size: int = 3,  # Number of sentences per chunk
                 overlap_size: int = 1,
                 max_documents: int = None):  # Limit number of documents to load
        
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load data index for lazy loading
        self.data = self._load_data()
        # Limit number of documents if specified
        if max_documents:
            self.data = self.data[:max_documents]
            logger.info(f"Limited to {len(self.data)} documents")
            
        # Create an index of all chunks without processing them
        self.chunk_index = self._create_chunk_index()
        
        logger.info(f"Indexed {len(self.chunk_index)} document chunks from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load ReDocRED data from JSON file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            return []
    
    def _create_chunk_index(self) -> List[Tuple[int, int, List[List[str]]]]:
        """Create an index of all chunks without processing them."""
        chunk_index = []
        t0 = time.time()

        for doc_idx, doc in enumerate(self.data):
            if doc_idx % 100 == 0:
                elapsed = time.time() - t0
                print(f"[DEBUG] 已处理 {doc_idx} 篇文档, 用时 {elapsed:.2f}s, chunk_index 长度 {len(chunk_index)}")
                t0 = time.time()

            try:
                # Extract document information
                sentences = doc.get('sents', [])
                if not sentences:
                    continue  # 跳过空文档

                # Create chunks from sentences
                chunks = self._create_chunks(sentences)
                
                # Add each chunk to index
                for chunk_idx, chunk_sentences in enumerate(chunks):
                    chunk_index.append((doc_idx, chunk_idx, chunk_sentences))
                        
            except Exception as e:
                logger.warning(f"Error indexing document {doc_idx}: {e}")
                continue
        
        return chunk_index
    
    def _create_chunks(self, sentences: List[List[str]]) -> List[List[List[str]]]:
        """Create overlapping chunks from document sentences."""
        if not sentences:
            return []
        
        chunks = []
        start_idx = 0
        num_sentences = len(sentences)
        
        while start_idx < num_sentences:
            end_idx = min(start_idx + self.chunk_size, num_sentences)
            chunk = sentences[start_idx:end_idx]
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx += self.chunk_size - self.overlap_size

        return chunks
    
    def _process_chunk(self, 
                      doc_idx: int,
                      chunk_idx: int,
                      chunk_sentences: List[List[str]]) -> Dict[str, Any]:
        """Process a single chunk into training format (lazy loading)."""
        
        # Get document data
        doc = self.data[doc_idx]
        doc_id = doc.get('title', f'doc_{doc_idx}')
        entities = doc.get('vertexSet', [])
        relations = doc.get('labels', [])
        
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
        return len(self.chunk_index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single chunk by index (lazy loading)."""
        if idx >= len(self.chunk_index):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.chunk_index)}")
        
        # Get chunk info from index
        doc_idx, chunk_idx, chunk_sentences = self.chunk_index[idx]
        
        # Process chunk on-demand
        return self._process_chunk(doc_idx, chunk_idx, chunk_sentences)
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        # This method is not compatible with lazy loading implementation
        # Would need to process all chunks to implement this
        raise NotImplementedError("get_document_chunks not implemented for lazy loading dataset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        # This method is not compatible with lazy loading implementation
        # Would need to process all chunks to implement this
        raise NotImplementedError("get_statistics not implemented for lazy loading dataset")
