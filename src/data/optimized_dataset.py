"""
Optimized ReDocRED Dataset implementation for StructureExtractor-Pretrain.
Supports loading preprocessed chunks from disk for efficient training.
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class OptimizedReDocREDDataset(Dataset):
    """
    Optimized ReDocRED dataset that loads preprocessed chunks from disk.
    Significantly reduces memory usage and loading time.
    """
    
    def __init__(self, 
                 data_dir: str,
                 tokenizer_name: str = "bert-base-uncased",
                 max_seq_length: int = 512):
        """
        Initialize optimized dataset.
        
        Args:
            data_dir: Directory containing preprocessed chunk files
            tokenizer_name: Name of tokenizer to use
            max_seq_length: Maximum sequence length for tokenization
        """
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load chunk index
        self.chunk_index = self._load_chunk_index()
        
        # Load first chunk to get structure info
        self._load_chunk_data(0)
        
        logger.info(f"Loaded optimized dataset with {len(self.chunk_index['chunk_files'])} chunk files")
    
    def _load_chunk_index(self) -> Dict[str, Any]:
        """Load chunk index from disk."""
        index_file = os.path.join(self.data_dir, "chunk_index.json")
        with open(index_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_chunk_data(self, chunk_file_idx: int) -> List[Dict[str, Any]]:
        """Load chunk data from a specific file."""
        if chunk_file_idx >= len(self.chunk_index['chunk_files']):
            raise IndexError(f"Chunk file index {chunk_file_idx} out of range")
        
        chunk_file = self.chunk_index['chunk_files'][chunk_file_idx]
        with open(chunk_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        """Return total number of chunks."""
        return self.chunk_index['total_chunks']
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single chunk by index."""
        # Calculate which file and position within file
        chunk_size = self.chunk_index['chunk_size']
        file_idx = idx // chunk_size
        position_idx = idx % chunk_size
        
        # Load chunk data if not already loaded or if different file
        if not hasattr(self, '_current_chunks') or getattr(self, '_current_file_idx', -1) != file_idx:
            self._current_chunks = self._load_chunk_data(file_idx)
            self._current_file_idx = file_idx
        
        # Return chunk data
        if position_idx >= len(self._current_chunks):
            raise IndexError(f"Position index {position_idx} out of range in file {file_idx}")
        
        return self._current_chunks[position_idx]


class MemoryEfficientReDocREDDataset(Dataset):
    """
    Memory-efficient ReDocRED dataset that loads data on-demand with caching.
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer_name: str = "bert-base-uncased",
                 max_seq_length: int = 512,
                 max_entities: int = 100,
                 max_relations: int = 50,
                 chunk_size: int = 3,
                 overlap_size: int = 1,
                 max_documents: int = None,
                 cache_size: int = 100):
        """
        Initialize memory-efficient dataset.
        
        Args:
            data_path: Path to ReDocRED data file
            tokenizer_name: Name of tokenizer to use
            max_seq_length: Maximum sequence length for tokenization
            max_entities: Maximum number of entities per chunk
            max_relations: Maximum number of relations per chunk
            chunk_size: Number of sentences per chunk
            overlap_size: Overlap between chunks
            max_documents: Maximum number of documents to load
            cache_size: Number of processed chunks to cache in memory
        """
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.cache_size = cache_size
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load data index
        self.data = self._load_data()
        # Limit number of documents if specified
        if max_documents:
            self.data = self.data[:max_documents]
            logger.info(f"Limited to {len(self.data)} documents")
            
        # Create an index of all chunks
        self.chunk_index = self._create_chunk_index()
        
        # Initialize cache
        self.chunk_cache = {}
        self.cache_order = []
        
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
        """Create an index of all chunks."""
        chunk_index = []
        
        for doc_idx, doc in enumerate(self.data):
            try:
                # Extract document information
                sentences = doc.get('sents', [])
                
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
                      doc_idx: int,
                      chunk_idx: int,
                      chunk_sentences: List[List[str]]) -> Dict[str, Any]:
        """Process a single chunk into training format."""
        
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
        """Get a single chunk by index with caching."""
        if idx >= len(self.chunk_index):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.chunk_index)}")
        
        # Check cache first
        if idx in self.chunk_cache:
            # Move to end of cache order (most recently used)
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            return self.chunk_cache[idx]
        
        # Get chunk info from index
        doc_idx, chunk_idx, chunk_sentences = self.chunk_index[idx]
        
        # Process chunk on-demand
        chunk_data = self._process_chunk(doc_idx, chunk_idx, chunk_sentences)
        
        # Add to cache
        self.chunk_cache[idx] = chunk_data
        self.cache_order.append(idx)
        
        # Remove oldest item if cache is full
        if len(self.chunk_cache) > self.cache_size:
            oldest_idx = self.cache_order.pop(0)
            del self.chunk_cache[oldest_idx]
        
        return chunk_data