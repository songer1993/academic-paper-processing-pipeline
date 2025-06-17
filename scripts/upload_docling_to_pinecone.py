#!/usr/bin/env python3
"""
Upload Docling documents to Pinecone using hierarchical chunking.
Based on the Docling Weaviate example but adapted for Pinecone.
Extracts metadata directly from Zotero filenames (Author et al. - Year - Title.pdf).
No filtering applied - all chunks preserved for maximum recall.
Requires only PINECONE_API_KEY in .env file.
"""

import os
import re
import json
import argparse
import logging
import statistics
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from pinecone import Pinecone

# Suppress noisy warnings and logs
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

from chunk_quality_classifier import create_optimized_classifier

from docling_core.types.doc import (
    DocItemLabel, 
    GroupLabel,
    DoclingDocument,
    NodeItem,
    TextItem,
    TableItem
)
from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.base import BaseChunker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress more verbose loggers
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

# Set LiteLLM to be completely silent
os.environ["LITELLM_LOG"] = "ERROR"

# Initialize the DSPy-based chunk classifier
quality_classifier = create_optimized_classifier()

# Load environment variables from .env file
if not load_dotenv():
    logger.warning("No .env file found. Make sure to set PINECONE_API_KEY in .env file.")

# Using DSPy-based chunk quality classifier with Gemini
logger.info("Using DSPy-based chunk quality classifier with Gemini")

def preprocess_chunk_text(text: str) -> str:
    """Clean and preprocess chunk text to improve vector quality."""
    if not text:
        return ""
        
    # Remove repeated headers and footers
    text = re.sub(r'(?m)^Author Manuscript\s*$', '', text)
    text = re.sub(r'(?m)^Page \d+\s*$', '', text)
    text = re.sub(r'(?m)^Author manuscript[^.]*\.\s*$', '', text)
    
    # Clean up special characters and formatting
    text = re.sub(r'\^(\d+)', r'[\1]', text)  # Convert superscripts to brackets
    text = re.sub(r'<!--.*?-->', '', text)  # Remove HTML comments
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Remove redundant section markers
    text = re.sub(r'(?m)^(Published in final edited form as:|Corresponding author:|Conflict of interest\s*\.)\s*$', '', text)
    
    return text.strip()

def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extract metadata from Zotero-style filename: 'Author et al. - Year - Title.pdf'"""
    # Remove .pdf extension if present
    name = filename.replace('.pdf', '').replace('.json', '')
    
    # Pattern: "Author et al. - Year - Title"
    # Split by " - " to get components
    parts = name.split(' - ')
    
    metadata = {
        'author': '',
        'year': '',
        'title': '',
        'citekey': ''
    }
    
    if len(parts) >= 3:
        metadata['author'] = parts[0].strip()
        metadata['year'] = parts[1].strip()
        metadata['title'] = ' - '.join(parts[2:]).strip()  # Join remaining parts as title
        
        # Create sophisticated citation key: firstAuthorTitleWordsYear
        # e.g., "De Zambotti et al. - 2024 - State of the science and recommendations" 
        # -> "dezambottiStateScienceRecommendations2024"
        
        # Extract first author family name
        author_parts = metadata['author'].split()
        if author_parts:
            # Handle cases like "De Zambotti" -> "dezambotti"
            first_author = author_parts[0].lower().replace(',', '')
            if len(author_parts) > 1 and author_parts[1] not in ['et', 'al.', 'al']:
                # Multi-word family name like "De Zambotti"
                first_author = ''.join(word.lower() for word in author_parts[:2] if word not in ['et', 'al.', 'al'])
        else:
            first_author = 'unknown'
        
        # Extract first 3 meaningful words from title
        title_words = [word.strip() for word in metadata['title'].split() if len(word.strip()) > 2]
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'with', 'from', 'using', 'based', 'research', 'study'}
        meaningful_words = [word for word in title_words if word.lower() not in stop_words][:3]
        
        # Capitalize first letter of each word and join
        title_part = ''.join(word.capitalize() for word in meaningful_words) if meaningful_words else 'Paper'
        
        metadata['citekey'] = f"{first_author}{title_part}{metadata['year']}"
        
        logger.debug(f"Extracted from filename '{filename}': {metadata}")
    else:
        logger.warning(f"Could not parse filename '{filename}' - expected format: 'Author et al. - Year - Title.pdf'")
        # Fallback: use filename as title and generate basic citekey
        metadata['title'] = name
        metadata['citekey'] = name.lower().replace(' ', '').replace('-', '')[:20]  # Truncate for safety
    
    return metadata



def load_docling_document(doc_path: Path) -> Optional[DoclingDocument]:
    """Load a Docling document from JSON file."""
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_dict = json.load(f)
        
        # Validate schema version
        if 'schema_name' not in doc_dict or doc_dict['schema_name'] != 'DoclingDocument':
            logger.error(f"Invalid document schema in {doc_path}")
            return None
            
        try:
            # Convert dict back to DoclingDocument with validation
            document = DoclingDocument.model_validate(doc_dict)
            logger.debug(f"Successfully loaded document: {doc_path}")
            logger.debug(f"Document version: {document.version if hasattr(document, 'version') else 'unknown'}")
            return document
        except Exception as validation_error:
            logger.error(f"Document validation error in {doc_path}: {validation_error}")
            return None
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in {doc_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading Docling document {doc_path}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def assess_chunk_relevance(chunk_text: str) -> bool:
    """Assess if a chunk is relevant enough to include."""
    # Skip empty chunks
    if not chunk_text.strip():
        return False
        
    # Skip chunks that are just headers/footers
    if re.match(r'^(Author Manuscript|Page \d+|Author manuscript)$', chunk_text.strip()):
        return False
        
    # Skip very short chunks (likely just formatting)
    if len(chunk_text.split()) < 5:
        return False
        
    return True

def setup_hierarchical_chunker() -> BaseChunker:
    """Setup hierarchical chunker similar to the Weaviate example."""
    chunker = HierarchicalChunker()
    return chunker

def chunk_docling_document(document: DoclingDocument, chunker: BaseChunker) -> List[Dict[str, Any]]:
    """Chunk a Docling document using hierarchical chunking with quality assessment."""
    try:
        # Ensure we have a DoclingDocument object
        if isinstance(document, dict):
            document = DoclingDocument.model_validate(document)
        elif hasattr(document, 'export_to_dict'):
            # If it's already a DoclingDocument but might be an older version
            document = DoclingDocument.model_validate(document.export_to_dict())
        
        # Initialize chunker with the document
        logger.info("Starting hierarchical chunking with quality assessment...")
        chunk_iter = chunker.chunk(document)
        chunks = []
        chunk_index = 0
        current_section = ""
        
        total_chunks = 0
        filtered_chunks = 0
        
        for chunk in chunk_iter:
            total_chunks += 1
            
            # Extract text content from chunk dict
            chunk_text = ""
            if isinstance(chunk, dict):
                chunk_text = chunk.get('text', '')
            elif hasattr(chunk, 'text'):
                chunk_text = chunk.text
                
            chunk_text = chunk_text.strip()
            chunk_text = preprocess_chunk_text(chunk_text)
            
            # Track section context from chunk metadata
            if isinstance(chunk, dict) and 'headings' in chunk:
                current_section = chunk['headings'][-1] if chunk['headings'] else current_section
            elif hasattr(chunk, 'headings') and chunk.headings:
                current_section = chunk.headings[-1]
            
            # Show progress for every 10 chunks
            if total_chunks % 10 == 0:
                logger.info(f"Processing chunk {total_chunks}...")
            
            # Assess chunk quality using DSPy classifier
            logger.debug(f"Assessing chunk {total_chunks}: {chunk_text[:100]}...")
            result = quality_classifier(chunk_text=chunk_text)
            logger.debug(f"Chunk {total_chunks} result: {result.classification} (confidence: {result.confidence:.2f})")
            
            if result.classification == "keep":
                # Extract metadata safely
                chunk_meta = {}
                if isinstance(chunk, dict):
                    chunk_meta = chunk.get('meta', {})
                elif hasattr(chunk, 'meta'):
                    chunk_meta = chunk.meta if isinstance(chunk.meta, dict) else {}
                
                chunk_data = {
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'section': current_section,
                    'meta': {
                        **chunk_meta,
                        'section_context': current_section,
                        'quality_confidence': result.confidence
                    },
                    'headings': (chunk['headings'] if isinstance(chunk, dict) and 'headings' in chunk 
                                else (chunk.headings if hasattr(chunk, 'headings') else []))
                }
                chunks.append(chunk_data)
                chunk_index += 1
            else:
                filtered_chunks += 1
        
        # Log quality metrics
        logger.info(f"Processed {total_chunks} chunks:")
        logger.info(f"- Kept {len(chunks)} high-quality chunks")
        logger.info(f"- Filtered {filtered_chunks} low-quality chunks")
        if chunks:
            logger.info(f"- Average quality confidence: {statistics.mean([c['meta']['quality_confidence'] for c in chunks]):.2f}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error chunking document: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

def extract_document_metadata(document: DoclingDocument) -> Dict[str, str]:
    """Extract metadata from Docling document with deduplication."""
    metadata = {}
    seen_values = set()
    
    # The document structure now uses children and $ref references
    if hasattr(document, 'body') and hasattr(document.body, 'children'):
        for child_ref in document.body.children:
            if hasattr(child_ref, '$ref'):
                # Follow the reference to get the actual item
                ref_path = child_ref['$ref'].split('/')
                if len(ref_path) >= 2:
                    collection = ref_path[1]  # e.g., 'texts', 'pictures'
                    index = int(ref_path[2])  # e.g., 0, 1, 2
                    
                    # Get the referenced item
                    if hasattr(document, collection):
                        items = getattr(document, collection)
                        if index < len(items):
                            item = items[index]
                            
                            # Check for title and abstract
                            if hasattr(item, 'label'):
                                if item.label == DocItemLabel.TITLE and hasattr(item, 'text'):
                                    value = preprocess_chunk_text(item.text.strip())
                                    if value and value not in seen_values:
                                        seen_values.add(value)
                                        metadata['extracted_title'] = value
                                elif item.label == DocItemLabel.ABSTRACT and hasattr(item, 'text'):
                                    value = preprocess_chunk_text(item.text.strip())
                                    if value and value not in seen_values:
                                        seen_values.add(value)
                                        metadata['extracted_abstract'] = value
    
    return metadata

def process_docling_file(doc_path: Path, pc: Pinecone, index, namespace: str, chunker: BaseChunker) -> bool:
    """Process a single Docling document and upload to Pinecone."""
    try:
        # Extract metadata directly from filename (Zotero format: "Author et al. - Year - Title.pdf/json")
        filename = doc_path.name
        file_metadata = extract_metadata_from_filename(filename)
        
        if not file_metadata['citekey']:
            logger.warning(f"Could not extract metadata from filename: {filename}")
            return False
        
        # Check if the first chunk already exists in Pinecone
        first_chunk_id = f"{file_metadata['citekey']}_chunk_0"
        try:
            fetch_response = index.fetch(ids=[first_chunk_id], namespace=namespace)
            if fetch_response.vectors and first_chunk_id in fetch_response.vectors:
                logger.info(f"Skipping {filename} (ID: {file_metadata['citekey']}) as its first chunk already exists in Pinecone.")
                return True
        except Exception as e:
            logger.warning(f"Could not verify existence of {first_chunk_id} (Error: {e}). Proceeding with processing.")
        
        # Load Docling document
        document = load_docling_document(doc_path)
        if not document:
            return False
        
        # Extract document-level metadata (title/abstract from document structure)
        doc_metadata = extract_document_metadata(document)
        
        # Chunk the document
        chunks = chunk_docling_document(document, chunker)
        if not chunks:
            logger.warning(f"No valid chunks generated for {filename}")
            return False
        
        # Create base metadata using filename extraction
        base_metadata = {
            'citekey': file_metadata['citekey'],
            'title': file_metadata['title'] or doc_metadata.get('extracted_title', ''),
            'author': file_metadata['author'],
            'year': file_metadata['year']
        }
        
        # Process each chunk
        vectors_to_upsert = []
        total_chunks = len(chunks)
        
        logger.info(f"Generating embeddings for {total_chunks} chunks...")
        for i, chunk_data in enumerate(chunks, 1):
            try:
                # Generate embedding using just the chunk text (following Weaviate example strategy)
                embedding = pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=[chunk_data['text']],
                    parameters={"input_type": "passage"}
                )
                
                # Create metadata for this chunk - only essential fields
                metadata = {
                    **base_metadata,
                    'text': chunk_data['text'],
                    'chunk_index': chunk_data['chunk_index']
                }
                
                # Create vector
                vector = {
                    "id": f"{file_metadata['citekey']}_chunk_{chunk_data['chunk_index']}",
                    "values": embedding[0]['values'],
                    "metadata": metadata
                }
                
                vectors_to_upsert.append(vector)
                
                # Log progress every 50 chunks
                if i % 50 == 0:
                    logger.info(f"Generated embeddings for {i}/{total_chunks} chunks...")
                    
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {i}/{total_chunks}: {e}")
                continue
        
        if not vectors_to_upsert:
            logger.error("No vectors generated for upload")
            return False
            
        # Batch upsert to Pinecone (upsert in batches of 100)
        logger.info(f"Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
        batch_size = 100
        total_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size
        
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            try:
                index.upsert(vectors=batch, namespace=namespace)
                current_batch = (i // batch_size) + 1
                logger.info(f"Uploaded batch {current_batch}/{total_batches} to Pinecone...")
            except Exception as e:
                logger.error(f"Error uploading batch {(i // batch_size) + 1}/{total_batches} to Pinecone: {e}")
                return False
        
        logger.info(f"Successfully processed and upserted {filename} (ID: {file_metadata['citekey']}, Chunks: {len(chunks)})")
        return True
        
    except Exception as e:
        logger.error(f"Error processing Docling file {doc_path}: {e}")
        return False

def find_docling_files(input_dir: Path) -> List[Path]:
    """Find all Docling JSON files in the directory and its subdirectories."""
    json_files = []
    for json_path in input_dir.rglob("*.json"):
        json_files.append(json_path)
    return json_files

def main():
    parser = argparse.ArgumentParser(description="""
    Upload Docling documents to Pinecone using hierarchical chunking.
    Based on the Docling Weaviate example but adapted for Pinecone.
    Extracts metadata directly from Zotero filenames (Author et al. - Year - Title.pdf).
    No filtering applied - all chunks preserved for maximum recall.
    Requires only PINECONE_API_KEY in .env file.
    """)
    parser.add_argument("--input_dir", type=str, required=True, 
                      help="Path to a single Docling JSON file or directory containing Docling files")
    parser.add_argument("--index_name", type=str, required=True, 
                      help="Pinecone index name")
    parser.add_argument("--namespace", type=str, default="", 
                      help="Optional Pinecone namespace")
    parser.add_argument("--debug", action="store_true", 
                      help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("docling").setLevel(logging.DEBUG)
    
    # Initialize Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY not found in environment variables. Set it in .env file.")
        return
    
    logger.info("Initializing Pinecone connection...")
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Test API key by listing indexes
        pc.list_indexes()
        logger.info("Successfully connected to Pinecone")
        
        # Get index
        logger.info(f"Connecting to index '{args.index_name}'...")
        index = pc.Index(args.index_name)
        
        # Test index connection
        index.describe_index_stats()
        logger.info(f"Successfully connected to index '{args.index_name}'")
        
    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {e}")
        return
    
    # Metadata is extracted directly from Zotero filenames - no BibTeX needed!
    logger.info("Extracting metadata directly from Zotero filenames")
    
    # Setup hierarchical chunker
    chunker = setup_hierarchical_chunker()
    
    # Handle both single file and directory input
    input_dir = Path(args.input_dir)
    if input_dir.is_file():
        if not input_dir.suffix.lower() == '.json':
            logger.error(f"Input file must be a JSON file: {input_dir}")
            return
        doc_files = [input_dir]
    else:
        # Find all Docling JSON files in directory
        doc_files = find_docling_files(input_dir)
    
    if not doc_files:
        logger.error(f"No Docling JSON files found at {input_dir}")
        return
    
    logger.info(f"Found {len(doc_files)} Docling document(s) to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for doc_file in doc_files:
        if process_docling_file(doc_file, pc, index, args.namespace, chunker):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Processing complete. Successfully processed: {successful}, Failed: {failed}")

if __name__ == "__main__":
    main() 