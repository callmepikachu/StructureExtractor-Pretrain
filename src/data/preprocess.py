"""
Data preprocessing utilities for StructureExtractor-Pretrain.
Handles data cleaning, formatting, and preprocessing tasks.
"""

import json
import re
from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def preprocess_redocred_data(input_path: str, output_path: str, 
                           max_documents: int = None) -> Dict[str, Any]:
    """
    Preprocess ReDocRED data for training.
    
    Args:
        input_path: Path to input ReDocRED JSON file
        output_path: Path to output preprocessed JSON file
        max_documents: Maximum number of documents to process (for testing)
        
    Returns:
        Statistics about the preprocessing
    """
    logger.info(f"Loading ReDocRED data from {input_path}")
    
    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} documents")
    
    # Limit documents if requested
    if max_documents:
        data = data[:max_documents]
        logger.info(f"Limited to {len(data)} documents")
    
    # Process documents
    processed_data = []
    stats = {
        'total_documents': len(data),
        'total_sentences': 0,
        'total_entities': 0,
        'total_relations': 0,
        'entity_types': defaultdict(int),
        'relation_types': defaultdict(int)
    }
    
    for doc_idx, doc in enumerate(data):
        try:
            # Process document
            processed_doc = _process_document(doc, stats)
            if processed_doc:
                processed_data.append(processed_doc)
        except Exception as e:
            logger.warning(f"Error processing document {doc_idx}: {e}")
            continue
    
    # Save processed data
    logger.info(f"Saving processed data to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # Calculate averages
    stats['processed_documents'] = len(processed_data)
    stats['avg_sentences_per_doc'] = stats['total_sentences'] / len(processed_data) if processed_data else 0
    stats['avg_entities_per_doc'] = stats['total_entities'] / len(processed_data) if processed_data else 0
    stats['avg_relations_per_doc'] = stats['total_relations'] / len(processed_data) if processed_data else 0
    
    logger.info(f"Preprocessing completed. Statistics: {dict(stats)}")
    return stats


def _process_document(doc: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single document.
    
    Args:
        doc: Document dictionary
        stats: Statistics dictionary to update
        
    Returns:
        Processed document dictionary
    """
    # Extract document information
    doc_id = doc.get('title', f'doc_{hash(str(doc)) % 10000}')
    sentences = doc.get('sents', [])
    entities = doc.get('vertexSet', [])
    relations = doc.get('labels', [])
    
    # Update statistics
    stats['total_sentences'] += len(sentences)
    
    # Process entities
    processed_entities = _process_entities(entities)
    stats['total_entities'] += len(processed_entities)
    
    for entity in processed_entities:
        stats['entity_types'][entity.get('type', 'UNKNOWN')] += 1
    
    # Process relations
    processed_relations = _process_relations(relations)
    stats['total_relations'] += len(processed_relations)
    
    for relation in processed_relations:
        stats['relation_types'][relation.get('r', 'UNKNOWN')] += 1
    
    # Return processed document
    return {
        'title': doc_id,
        'sents': sentences,
        'vertexSet': processed_entities,
        'labels': processed_relations
    }


def _process_entities(entities: List[List[Dict]]) -> List[List[Dict]]:
    """
    Process entity vertex set.
    
    Args:
        entities: List of entity mentions
        
    Returns:
        Processed entities
    """
    processed_entities = []
    
    for entity_group in entities:
        processed_group = []
        for mention in entity_group:
            # Clean and standardize entity mention
            processed_mention = {
                'name': mention.get('name', mention.get('mention', '')),
                'sent_id': mention.get('sent_id', 0),
                'pos': mention.get('pos', [0, 0]),
                'type': mention.get('type', 'ENTITY'),
                'mention_id': mention.get('id', ''),
                'global_id': mention.get('global_id', '')
            }
            processed_group.append(processed_mention)
        processed_entities.append(processed_group)
    
    return processed_entities


def _process_relations(relations: List[Dict]) -> List[Dict]:
    """
    Process relation labels.
    
    Args:
        relations: List of relation dictionaries
        
    Returns:
        Processed relations
    """
    processed_relations = []
    
    for relation in relations:
        # Clean and standardize relation
        processed_relation = {
            'h': relation.get('h', 0),  # head entity index
            't': relation.get('t', 0),  # tail entity index
            'r': relation.get('r', 'RELATED_TO'),  # relation type
            'evidence': relation.get('evidence', []),  # evidence sentences
            'confidence': relation.get('confidence', 1.0)  # confidence score
        }
        processed_relations.append(processed_relation)
    
    return processed_relations


def create_entity_linking_candidates(entities: List[List[Dict]], 
                                   max_candidates: int = 1000) -> Dict[str, List[str]]:
    """
    Create candidate entity mentions for entity linking.
    
    Args:
        entities: List of entity groups
        max_candidates: Maximum number of candidates to generate
        
    Returns:
        Dictionary mapping entity names to candidate mentions
    """
    candidates = defaultdict(list)
    candidate_count = 0
    
    for entity_group in entities:
        for mention in entity_group:
            entity_name = mention.get('name', '')
            if entity_name and candidate_count < max_candidates:
                candidates[entity_name.lower()].append({
                    'mention': mention.get('mention', ''),
                    'type': mention.get('type', 'ENTITY'),
                    'sent_id': mention.get('sent_id', 0)
                })
                candidate_count += 1
    
    return dict(candidates)


def format_for_training(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format data for training with StructureExtractor.
    
    Args:
        data: List of processed documents
        
    Returns:
        List of training examples
    """
    training_examples = []
    
    for doc in data:
        # Each document can be split into chunks for incremental processing
        sentences = doc.get('sents', [])
        entities = doc.get('vertexSet', [])
        relations = doc.get('labels', [])
        
        # For simplicity, we create one example per document
        # In practice, you might split long documents into chunks
        example = {
            'doc_id': doc.get('title', ''),
            'text': ' '.join([' '.join(sent) for sent in sentences]),
            'entities': entities,
            'relations': relations
        }
        
        training_examples.append(example)
    
    return training_examples