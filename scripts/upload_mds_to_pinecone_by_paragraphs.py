#!/usr/bin/env python3
"""
Upload markdown files to Pinecone, splitting content into 1-2 paragraph chunks.
Uses Gemini to filter out non-research content (acknowledgments, references, etc.).
Maintains bibliographic metadata from BibTeX entries.
Requires GEMINI_API_KEY and PINECONE_API_KEY in .env file.
"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import bibtexparser
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
if not load_dotenv():
    logger.error("No .env file found. This script requires GEMINI_API_KEY and PINECONE_API_KEY in .env file.")
    exit(1)

# Initialize Gemini
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not found in environment variables. Set it in .env file.")
        exit(1)

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    # Test the model with a simple prompt to ensure it's working
    test_response = model.generate_content("Respond with OK if you can read this.", 
        generation_config={"temperature": 0, "candidate_count": 1})
    if not test_response or test_response.text.strip() != "OK":
        raise Exception("Gemini model test failed")
    logger.info("Successfully initialized Gemini model")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {e}")
    logger.warning("Will proceed without content filtering")
    model = None

def normalize_filename(filename: str) -> str:
    """Normalize filename by removing special characters and converting to lowercase."""
    # Remove special characters and convert to lowercase
    normalized = re.sub(r'[^\w\s-]', '', filename.lower())
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()

def extract_filename_from_file_field(file_field: str) -> Optional[str]:
    """Extract the base filename without extension from BibTeX file field."""
    if not file_field:
        return None
    
    # The file field might contain multiple files separated by semicolons
    # We'll take the first one that matches our pattern
    file_paths = file_field.split(';')
    
    for path in file_paths:
        # Remove any curly braces and trim
        path = path.strip('{}').strip()
        
        # Extract the filename from the path
        match = re.search(r'/([^/]+)\.pdf$', path)
        if match:
            # Normalize the filename
            return normalize_filename(match.group(1))
    
    return None

def get_entry_type(bib_entry: Dict) -> str:
    """Get the type of BibTeX entry (article, inproceedings, book, etc.)."""
    # Get ENTRYTYPE field or default to 'unknown'
    return bib_entry.get('ENTRYTYPE', 'unknown').lower()

def extract_bib_metadata(bib_entry: Dict) -> Dict:
    """Extract common metadata fields from BibTeX entry."""
    # Common fields that should exist in most entry types
    metadata = {
        'entry_type': get_entry_type(bib_entry),  # Type of the entry
        'citation_key': bib_entry.get('ID', ''),  # Citation key from BibTeX
        'title': bib_entry.get('title', ''),      # Title exists in all types
        'author': bib_entry.get('author', ''),    # Author(s)
        'year': bib_entry.get('year', ''),        # Publication year
        'abstract': bib_entry.get('abstract', '') # Abstract (if available)
    }
    
    # Handle optional identifiers if they exist
    if 'doi' in bib_entry:
        metadata['doi'] = bib_entry['doi']
    if 'url' in bib_entry:
        metadata['url'] = bib_entry['url']
    if 'keywords' in bib_entry:
        metadata['keywords'] = bib_entry['keywords']
    
    # Add venue information based on entry type
    entry_type = metadata['entry_type']
    if entry_type == 'article':
        metadata['venue'] = bib_entry.get('journal', '')
    elif entry_type == 'inproceedings':
        metadata['venue'] = bib_entry.get('booktitle', '')
    elif entry_type == 'book':
        metadata['venue'] = bib_entry.get('publisher', '')
    elif entry_type == 'phdthesis' or entry_type == 'mastersthesis':
        metadata['venue'] = bib_entry.get('school', '')
    else:
        # For other types, try common venue fields
        for field in ['journal', 'booktitle', 'publisher', 'school']:
            if field in bib_entry:
                metadata['venue'] = bib_entry[field]
                break
    
    return metadata

def load_bibtex_entries(bibtex_file: Path) -> Dict[str, Dict]:
    """Load BibTeX entries and create a mapping of filenames to entries."""
    try:
        with open(bibtex_file, 'r', encoding='utf-8') as f:
            bib_database = bibtexparser.load(f)
        
        # Create a dictionary with normalized filenames as keys
        entries = {}
        for entry in bib_database.entries:
            if 'file' in entry and 'ID' in entry:  # Ensure both file and ID exist
                filename = extract_filename_from_file_field(entry['file'])
                if filename:
                    entries[filename] = entry
                    logger.debug(f"Mapped {filename} to BibTeX entry (ID: {entry['ID']}, Type: {entry.get('ENTRYTYPE', 'unknown')})")
        
        return entries
    except Exception as e:
        logger.error(f"Error loading BibTeX file: {e}")
        return {}

def find_markdown_files(markdown_dir: Path) -> List[Path]:
    """Find all markdown files in the directory and its subdirectories."""
    md_files = []
    for dirpath, dirnames, filenames in os.walk(markdown_dir):
        for filename in filenames:
            if filename.endswith('.md'):
                md_files.append(Path(dirpath) / filename)
    return md_files

def assess_chunk_relevance(chunk: str) -> bool:
    """Use Gemini to assess if a chunk contains meaningful research content."""
    # If model initialization failed, keep all chunks
    if model is None:
        return True
        
    try:
        prompt = """You are a research paper content filter. Your task is to determine if a text chunk contains meaningful research content.

Rules:
1. ONLY respond with a single word: either "KEEP" or "SKIP"
2. Respond "KEEP" if the chunk contains:
   - Research methodology
   - Experimental results
   - Data analysis
   - Theoretical framework
   - Literature findings
   - Technical discussion
   - Research conclusions
   - Abstract content
   - Research objectives
   - Problem statements

3. Respond "SKIP" if the chunk contains:
   - Acknowledgments
   - Author information
   - Funding statements
   - References/citations
   - Copyright notices
   - Journal submission info
   - Headers/footers
   - Table of contents
   - Section numbers/titles only
   - Figure/table captions only

Text chunk to assess:
{chunk}

Respond with ONLY one word (KEEP or SKIP):"""
        
        # Retry up to 3 times
        for attempt in range(3):
            try:
                response = model.generate_content(
                    prompt.format(chunk=chunk),
                    generation_config={"temperature": 0, "candidate_count": 1}
                )
                result = response.text.strip().upper()
                
                # Only accept exact matches
                if result not in ["KEEP", "SKIP"]:
                    logger.warning(f"Unexpected response from Gemini: {result}, defaulting to KEEP")
                    return True
                    
                return result == "KEEP"
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise e
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}/3): {e}")
                continue
        
    except Exception as e:
        logger.warning(f"Error assessing chunk relevance: {e}")
        # If assessment fails, default to keeping the chunk
        return True

def chunk_text(text: str) -> List[str]:
    """Split text into chunks of 1-2 paragraphs and assess relevance."""
    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n')]
    
    # Basic filtering first (length and text quality)
    filtered_paragraphs = []
    for p in paragraphs:
        if (len(p) >= 50 and                          # Length check
            not re.match(r'^[^a-zA-Z0-9]*$', p) and  # Skip decorative lines
            len(re.findall(r'[a-zA-Z]', p)) > len(p) * 0.4):  # At least 40% letters
            filtered_paragraphs.append(p)
    
    chunks = []
    current_chunk = []
    
    for paragraph in filtered_paragraphs:
        # Start new chunk if we already have 2 paragraphs
        if len(current_chunk) >= 2:
            chunk_text = '\n\n'.join(current_chunk)
            # Only keep chunk if Gemini assesses it as relevant
            if assess_chunk_relevance(chunk_text):
                chunks.append(chunk_text)
            current_chunk = []
        
        current_chunk.append(paragraph)
    
    # Add any remaining paragraphs
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        if assess_chunk_relevance(chunk_text):
            chunks.append(chunk_text)
    
    return chunks

def process_markdown_file(md_file: Path, bib_entries: Dict[str, Dict], pc: Pinecone, index, namespace: str) -> bool:
    """Process a single markdown file and upsert to Pinecone."""
    try:
        # Get the parent directory name (paper title) and normalize it
        dir_name = normalize_filename(md_file.parent.name)
        
        # Get BibTeX entry
        bib_entry = bib_entries.get(dir_name)
        if not bib_entry:
            logger.warning(f"No BibTeX entry found for {dir_name}")
            return False
        
        # Ensure we have a citation key (ID)
        if 'ID' not in bib_entry:
            logger.warning(f"No citation key found in BibTeX entry for {dir_name}")
            return False
        
        # Read markdown content
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {md_file}: {e}")
            return False
        
        try:
            # Split content into chunks
            chunks = chunk_text(content)
            logger.debug(f"Split {dir_name} into {len(chunks)} chunks")
            
            # Create base metadata that will be the same for all chunks
            base_metadata = {
                'citekey': bib_entry['ID'],  # Citation key from BibTeX
                'title': bib_entry.get('title', ''),
                'author': bib_entry.get('author', ''),
                'year': bib_entry.get('year', ''),
                'abstract': bib_entry.get('abstract', ''),
                'doi': bib_entry.get('doi', '')
            }
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Create enriched text for embedding by combining title, abstract, chunk position and content
                enriched_text = f"Title: {bib_entry.get('title', '')}\n\nAbstract: {bib_entry.get('abstract', '')}\n\nChunk {i+1} of {len(chunks)}:\n\n{chunk}"
                
                # Generate embedding using the enriched text
                embedding = pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=[enriched_text],
                    parameters={"input_type": "passage"}
                )
                
                # Create metadata for this chunk - keeping fields separate
                metadata = {
                    **base_metadata,
                    'text': chunk,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                
                # Create vector
                vector = {
                    "id": f"{bib_entry['ID']}_chunk_{i}",  # Unique ID for each chunk
                    "values": embedding[0]['values'],
                    "metadata": metadata
                }
                
                # Upsert to Pinecone
                index.upsert(
                    vectors=[vector],
                    namespace=namespace
                )
            
            logger.info(f"Successfully processed and upserted {dir_name} (ID: {bib_entry['ID']}, Chunks: {len(chunks)})")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {dir_name}: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Error processing markdown file {md_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="""
    Upload markdown files to Pinecone, splitting content into 1-2 paragraph chunks.
    Uses Gemini to filter out non-research content (acknowledgments, references, etc.).
    Maintains bibliographic metadata from BibTeX entries for each chunk.
    Requires GEMINI_API_KEY and PINECONE_API_KEY in .env file.
    """)
    parser.add_argument("--input_dir", type=str, required=True, 
                      help="Path to a single markdown file or directory containing markdown files")
    parser.add_argument("--bibtex_file", type=str, required=True, 
                      help="Path to BibTeX file with paper metadata")
    parser.add_argument("--index_name", type=str, required=True, 
                      help="Pinecone index name")
    parser.add_argument("--namespace", type=str, default="", 
                      help="Optional Pinecone namespace")
    parser.add_argument("--debug", action="store_true", 
                      help="Enable debug logging for chunk processing")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Initialize Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY not found in environment variables. Set it in .env file.")
        return
    
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Get index
    try:
        index = pc.Index(args.index_name)
    except Exception as e:
        logger.error(f"Error connecting to index {args.index_name}: {e}")
        return
    
    # Load BibTeX entries
    bibtex_file = Path(args.bibtex_file)
    bib_entries = load_bibtex_entries(bibtex_file)
    if not bib_entries:
        logger.error("No BibTeX entries loaded")
        return
    
    logger.info(f"Loaded {len(bib_entries)} BibTeX entries")
    
    # Handle both single file and directory input
    input_dir = Path(args.input_dir)
    if input_dir.is_file():
        if not input_dir.suffix.lower() == '.md':
            logger.error(f"Input file must be a markdown file: {input_dir}")
            return
        md_files = [input_dir]
    else:
        # Find all markdown files in directory
        md_files = find_markdown_files(input_dir)
    
    if not md_files:
        logger.error(f"No markdown files found at {input_dir}")
        return
    
    logger.info(f"Found {len(md_files)} markdown file(s) to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for md_file in md_files:
        if process_markdown_file(md_file, bib_entries, pc, index, args.namespace):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Processing complete. Successfully processed: {successful}, Failed: {failed}")

if __name__ == "__main__":
    main() 