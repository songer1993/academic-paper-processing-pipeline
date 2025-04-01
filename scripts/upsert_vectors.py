import os
from dotenv import load_dotenv
import markdown
from html2text import HTML2Text
import argparse
from typing import List, Tuple, Dict
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, VectorType
import time
import logging
import re
import threading
import random
import json

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Validate required environment variables
if not os.getenv("PINECONE_API_KEY"):
    logger.error("PINECONE_API_KEY not found in environment variables or .env file")
    raise ValueError("PINECONE_API_KEY is required")

class RateLimiter:
    """Rate limiter with token bucket algorithm."""
    def __init__(self, tokens_per_second: float, burst_limit: int):
        self.tokens_per_second = tokens_per_second
        self.burst_limit = burst_limit
        self.tokens = burst_limit
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens and return the time to wait.
        Returns the number of seconds to wait before proceeding.
        """
        with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.burst_limit,
                self.tokens + time_passed * self.tokens_per_second
            )
            self.last_update = now
            
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.tokens_per_second
                self.tokens = 0
                return wait_time
            
            self.tokens -= tokens
            return 0

class APIRateLimiters:
    """Manage rate limiters for different API services."""
    def __init__(self):
        # Pinecone limits: 100 requests per minute per project
        self.pinecone_limiter = RateLimiter(
            tokens_per_second=1.5,  # 90 requests per minute (conservative)
            burst_limit=10
        )
        
        # Gemini limits: 60 requests per minute
        self.gemini_limiter = RateLimiter(
            tokens_per_second=0.8,  # 48 requests per minute (conservative)
            burst_limit=5
        )
        
        # Embedding API limits: 100 requests per minute
        self.embedding_limiter = RateLimiter(
            tokens_per_second=1.5,  # 90 requests per minute (conservative)
            burst_limit=10
        )

def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0.8, 1.2)
    return delay * jitter

def detect_document_type(content: str, filename: str) -> str:
    """
    Detect whether the document is a book or academic paper based on content and structure.
    
    Args:
        content (str): Document content
        filename (str): Name of the file
        
    Returns:
        str: 'book' or 'paper'
    """
    # Count sections and estimate content length
    section_count = len(re.findall(r'^#+\s+', content, re.MULTILINE))
    content_length = len(content)
    
    # Look for book-specific indicators
    book_indicators = {
        'chapter': re.search(r'(?i)chapter\s+\d+|chapter\s+[ivxlcdm]+', content) is not None,
        'toc': re.search(r'(?i)^#+\s*(?:table\s+of\s+contents|contents)$', content, re.MULTILINE) is not None,
        'preface': re.search(r'(?i)^#+\s*(?:preface|foreword|introduction)$', content, re.MULTILINE) is not None,
        'appendix': re.search(r'(?i)^#+\s*appendix\s+[a-z]', content, re.MULTILINE) is not None,
        'index': re.search(r'(?i)^#+\s*(?:index|glossary)$', content, re.MULTILINE) is not None
    }
    
    # Score based on indicators
    book_score = sum(book_indicators.values())
    
    # Books typically have more sections and longer content
    if (book_score >= 2 or 
        (section_count > 20 and content_length > 100000) or
        any(term in filename.lower() for term in ['book', 'manual', 'guide', 'handbook'])):
        return 'book'
    
    return 'paper'

def get_section_type(title: str, doc_type: str = 'paper') -> str:
    """Determine the type of section based on its title."""
    title_lower = title.lower().strip()
    
    if doc_type == 'book':
        if re.match(r'^chapter\s+\d+|^[ivxlcdm]+\.\s|^\d+\.\s', title_lower):
            return 'chapter'
        elif 'appendix' in title_lower:
            return 'appendix'
        elif title_lower in ['glossary', 'index', 'table of contents']:
            return 'reference'
        elif title_lower in ['preface', 'foreword', 'introduction']:
            return 'front_matter'
        elif 'exercise' in title_lower or 'review' in title_lower:
            return 'practice'
            
    # Common section types for both books and papers
    if title_lower == 'abstract':
        return 'abstract'
    elif title_lower in ['introduction', 'background']:
        return 'introduction'
    elif title_lower in ['conclusion', 'conclusions', 'discussion', 'summary']:
        return 'conclusion'
    elif title_lower in ['method', 'methods', 'methodology']:
        return 'method'
    elif title_lower in ['result', 'results', 'evaluation', 'experiments']:
        return 'result'
    elif title_lower in ['references', 'bibliography']:
        return 'references'
        
    return 'body'

def parse_academic_markdown(markdown_content: str, doc_type: str = 'paper') -> List[Tuple[str, List[str]]]:
    """
    Parse an academic Markdown file into semantic chunks, optimized for Pinecone storage and retrieval.
    Handles both academic papers and books with appropriate chunking strategies.
    
    Args:
        markdown_content (str): The content of the academic Markdown file
        doc_type (str): Type of document ('paper' or 'book')
    
    Returns:
        List[Tuple[str, List[str]]]: List of (chunk_content, section_hierarchy) tuples
    """
    lines = markdown_content.splitlines()
    current_section_hierarchy = []
    current_content = []
    academic_sections = []
    current_header_level = 0
    in_equation = False
    equation_buffer = []
    
    # Adjust chunk sizes based on document type
    if doc_type == 'book':
        TARGET_CHUNK_SIZE = 4096    # ~1024 tokens for book sections
        MIN_CHUNK_SIZE = 1024       # ~256 tokens minimum
        MAX_CHUNK_SIZE = 8192       # ~2048 tokens maximum
    else:  # paper
        TARGET_CHUNK_SIZE = 2048    # ~512 tokens for technical content
        MIN_CHUNK_SIZE = 512        # ~128 tokens minimum
        MAX_CHUNK_SIZE = 4096       # ~1024 tokens maximum
    
    # Non-essential sections to skip or minimize
    NON_ESSENTIAL_SECTIONS = {
        'acknowledgments', 'acknowledgements', 'funding', 'conflicts of interest',
        'declaration of competing interest', 'author contributions', 'supplementary material',
        'copyright notice', 'financial disclosure'
    }
    
    # Book-specific sections to handle differently
    if doc_type == 'book':
        NON_ESSENTIAL_SECTIONS.update({
            'copyright page', 'dedication', 'about the author', 'colophon',
            'publisher information', 'printing history'
        })
    
    # Essential metadata sections to preserve
    ESSENTIAL_METADATA = {
        'abstract', 'introduction', 'methodology', 'methods', 'results', 'discussion',
        'conclusion', 'references', 'bibliography', 'citations'
    }
    
    # Add book-specific essential sections
    if doc_type == 'book':
        ESSENTIAL_METADATA.update({
            'chapter', 'appendix', 'glossary', 'index', 'table of contents',
            'preface', 'foreword', 'summary', 'review questions', 'exercises'
        })
    
    def is_non_essential_section(title: str) -> bool:
        """Check if a section title indicates non-essential content."""
        title_lower = title.lower().strip()
        return any(section in title_lower for section in NON_ESSENTIAL_SECTIONS)
    
    def is_chapter_header(line: str) -> bool:
        """Check if a line is a chapter header."""
        if not doc_type == 'book':
            return False
        line_lower = line.lower().strip('# ')
        return (line_lower.startswith('chapter ') or
                re.match(r'^[ivxlcdm]+\.\s', line_lower) or
                re.match(r'^\d+\.\s', line_lower))
    
    def get_header_info(line: str) -> Tuple[int, str]:
        """Extract header level and title from a markdown header line."""
        line = line.strip()
        header_level = len(line) - len(line.lstrip('#'))
        title = line[header_level:].strip()
        return header_level, title
    
    def should_split_chunk(content: List[str], force_split: bool = False) -> bool:
        """Determine if current chunk should be split based on size and semantics."""
        chunk_text = '\n'.join(content)
        chunk_size = len(chunk_text)
        
        # Always split if exceeding max size or forced
        if chunk_size > MAX_CHUNK_SIZE or force_split:
            return True
            
        # Don't split if under minimum size
        if chunk_size < MIN_CHUNK_SIZE:
            return False
            
        # If within target range, split on semantic boundaries
        if chunk_size >= TARGET_CHUNK_SIZE:
            return True
            
        return False
    
    def is_semantic_boundary(line: str) -> bool:
        """Check if a line represents a semantic boundary."""
        line_stripped = line.strip()
        
        # Headers are always boundaries
        if line_stripped.startswith('#'):
            return True
            
        # Chapter markers in books
        if doc_type == 'book' and is_chapter_header(line_stripped):
            return True
            
        # Empty lines might be paragraph boundaries
        if not line_stripped:
            return True
            
        # Equations are kept together
        if '$$' in line:
            return True
            
        # References section is a boundary
        if line_stripped.lower().startswith('reference'):
            return True
            
        # Book-specific boundaries
        if doc_type == 'book':
            # Exercise or review question boundaries
            if re.match(r'(?i)^\d+\.\s+exercise|review\s+question', line_stripped):
                return True
            # Appendix boundaries
            if re.match(r'(?i)^appendix\s+[a-z]', line_stripped):
                return True
                
        return False
    
    def process_current_content():
        """Process and split current content into chunks."""
        if not current_content:
            return
            
        # Skip processing if we're in a non-essential section and it's not a reference
        if (current_section_hierarchy and 
            is_non_essential_section(current_section_hierarchy[-1]) and 
            not any('reference' in s.lower() for s in current_section_hierarchy)):
            current_content.clear()
            return
            
        content_text = '\n'.join(current_content)
        
        # If content is small enough, keep it as one chunk
        if len(content_text) <= MAX_CHUNK_SIZE:
            if content_text.strip():
                # Ensure we have a valid hierarchy
                hierarchy = current_section_hierarchy if current_section_hierarchy else ["Main"]
                academic_sections.append((content_text.strip(), hierarchy))
            current_content.clear()
            return
        
        # Split large content into semantic chunks
        chunks = []
        current_chunk = []
        
        for line in current_content:
            current_chunk.append(line)
            
            if should_split_chunk(current_chunk) and is_semantic_boundary(line):
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = []
        
        # Add remaining content as final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
        
        # Add all chunks with their hierarchy
        for chunk in chunks:
            if chunk.strip():
                # Ensure we have a valid hierarchy
                hierarchy = current_section_hierarchy if current_section_hierarchy else ["Main"]
                academic_sections.append((chunk, hierarchy))
        
        current_content.clear()
    
    try:
        for line in lines:
            # Handle equations (preserve them in the content)
            if '$$' in line:
                if in_equation:
                    equation_buffer.append(line)
                    current_content.extend(equation_buffer)
                    equation_buffer.clear()
                    in_equation = False
                else:
                    in_equation = True
                    equation_buffer = [line]
                continue
            if in_equation:
                equation_buffer.append(line)
                continue
                
            # Handle headers
            if line.strip().startswith('#'):
                # Process previous section before starting new one
                process_current_content()
                
                # Parse header
                header_level, section_title = get_header_info(line)
                
                # Skip non-essential sections unless they contain references
                if (is_non_essential_section(section_title) and 
                    not 'reference' in section_title.lower()):
                    current_section_hierarchy = []
                    current_header_level = 0
                    continue
                
                # Update section hierarchy
                if header_level <= current_header_level:
                    # Trim hierarchy to appropriate level
                    current_section_hierarchy = current_section_hierarchy[:header_level - 1]
                current_section_hierarchy.append(section_title)
                current_header_level = header_level
                
                # Add header line to content
                current_content.append(line)
            else:
                # Skip content if we're in a non-essential section
                if (current_section_hierarchy and 
                    is_non_essential_section(current_section_hierarchy[-1]) and 
                    not any('reference' in s.lower() for s in current_section_hierarchy)):
                    continue
                
                # Handle regular content
                current_content.append(line)
                
                # Check if we should process the current chunk
                if should_split_chunk(current_content) and is_semantic_boundary(line):
                    process_current_content()
        
        # Process final section
        process_current_content()
        
        # Handle empty result
        if not academic_sections:
            logger.warning("No sections found in document, creating single section")
            academic_sections.append((markdown_content.strip(), ["Main"]))
        
        return academic_sections
        
    except Exception as e:
        logger.error(f"Error parsing markdown: {str(e)}")
        # Return whole document as one section if parsing fails
        return [(markdown_content.strip(), ["Main"])]

def convert_markdown_to_plaintext(markdown_text: str) -> str:
    """
    Convert academic Markdown text to clean plain text, removing formatting.
    
    Args:
        markdown_text (str): The academic Markdown text to convert.
    
    Returns:
        str: Clean plain text version of the input.
    """
    html = markdown.Markdown().convert(markdown_text)
    html_converter = HTML2Text()
    html_converter.ignore_links = True
    html_converter.ignore_images = True
    html_converter.ignore_tables = True
    return html_converter.handle(html).strip()

def get_embeddings_batch(texts: List[str], pc: Pinecone, model: str = "llama-text-embed-v2") -> List[Dict]:
    """
    Get embeddings for a batch of texts using Pinecone's inference API with improved rate limiting and error handling.
    
    Args:
        texts: List of texts to embed
        pc: Pinecone client instance
        model: Model name for embeddings
        
    Returns:
        List of embeddings or empty list on failure
    """
    if not texts:
        return []
        
    try:
        all_embeddings = []
        batch_size = 16  # Smaller batch size for better reliability
        rate_limiters = APIRateLimiters()
        
        # Process texts in batches with progress tracking
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processing embedding batch {batch_num}/{total_batches}")
            
            # Validate batch texts
            valid_texts = []
            for text in batch_texts:
                cleaned_text = clean_text_for_embedding(text)
                if cleaned_text:
                    valid_texts.append(cleaned_text)
                else:
                    logger.warning("Skipping invalid text in batch")
                    
            if not valid_texts:
                logger.warning(f"No valid texts in batch {batch_num}, skipping")
                continue
            
            # Apply rate limiting with dynamic backoff
            wait_time = rate_limiters.embedding_limiter.acquire()
            if wait_time > 0:
                logger.info(f"Rate limiting: waiting {wait_time:.1f}s before embedding batch")
                time.sleep(wait_time)
            
            # Retry logic with exponential backoff
            max_retries = 5
            base_delay = 2.0
            success = False
            
            for attempt in range(max_retries):
                try:
                    embeddings = pc.inference.embed(
                        model=model,
                        inputs=valid_texts,
                        parameters={
                            "input_type": "passage",
                            "truncate": "END"
                        }
                    )
                    
                    # Validate embeddings
                    if not embeddings or not all('values' in emb for emb in embeddings):
                        raise ValueError("Invalid embedding response")
                        
                    # Extract embedding values
                    batch_embeddings = [emb['values'] for emb in embeddings]
                    
                    # Validate embedding dimensions
                    expected_dim = 1024  # Expected dimension for llama-text-embed-v2
                    if not all(len(emb) == expected_dim for emb in batch_embeddings):
                        raise ValueError(f"Unexpected embedding dimensions")
                        
                    all_embeddings.extend(batch_embeddings)
                    success = True
                    break
                    
                except Exception as e:
                    error_msg = str(e)
                    if "rate limit" in error_msg.lower():
                        # Adjust rate limiter for next time
                        rate_limiters.embedding_limiter.tokens_per_second *= 0.8
                        
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to get embeddings after {max_retries} attempts: {error_msg}")
                        # Add empty embeddings to maintain order
                        all_embeddings.extend([[0.0] * expected_dim] * len(valid_texts))
                    else:
                        delay = exponential_backoff(attempt, base_delay)
                        logger.warning(f"Embedding attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                        logger.info(f"Retrying in {delay:.1f}s...")
                        time.sleep(delay)
            
            if success:
                # Add small delay between successful batches
                time.sleep(0.5)
            else:
                # Larger delay after failed batch
                time.sleep(2.0)
        
        return all_embeddings
        
    except Exception as e:
        logger.error(f"Unexpected error in get_embeddings_batch: {str(e)}")
        if hasattr(e, 'response'):
            logger.error(f"Response details: {e.response.text if hasattr(e.response, 'text') else str(e.response)}")
        return []

def clean_text_for_embedding(text: str) -> str:
    """
    Clean and validate text for embedding with improved handling of special cases.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text ready for embedding
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Remove common problematic patterns
    replacements = {
        '\u200b': '',  # Zero-width space
        '\ufeff': '',  # Zero-width no-break space
        '\xa0': ' ',   # Non-breaking space
        '\t': ' ',     # Tabs to spaces
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Handle equations and code blocks
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: f"[EQUATION: {m.group(1).strip()}]", text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', lambda m: f"[INLINE_EQUATION: {m.group(1).strip()}]", text)
    text = re.sub(r'```.*?```', lambda m: f"[CODE_BLOCK]", text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', lambda m: f"[CODE]", text)
    
    # Handle references and citations
    text = re.sub(r'\[(\d+)\]\s*([^\[]+)', lambda m: f"[REF-{m.group(1)}: {m.group(2).strip()}]", text)
    text = re.sub(r'\[@.*?\]', '[CITATION]', text)
    
    # Handle URLs and DOIs
    text = re.sub(r'(https?://\S+|doi:\S+)', '[URL]', text)
    
    # Clean whitespace while preserving structure
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    for p in paragraphs:
        # Normalize internal whitespace
        p = ' '.join(p.split())
        if p.strip():
            cleaned_paragraphs.append(p)
    text = '\n\n'.join(cleaned_paragraphs)
    
    # Remove any remaining control characters except newlines
    text = ''.join(char for char in text if char.isprintable() or char == '\n')
    
    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove other markdown formatting
    text = re.sub(r'[*_~]', '', text)
    
    # Validate final text
    if not text.strip():
        return ""
    
    # Ensure reasonable length
    if len(text) < 10:  # Too short to be meaningful
        return ""
        
    return text.strip()

def truncate_text(text: str, max_bytes: int = 32000) -> str:
    """
    Truncate text to stay under byte limit while preserving semantic meaning.
    
    Args:
        text: Text to truncate
        max_bytes: Maximum allowed bytes
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
        
    text_bytes = text.encode('utf-8')
    if len(text_bytes) <= max_bytes:
        return text
        
    # Try to find a good breakpoint
    truncated_length = max_bytes
    while truncated_length > 0:
        try:
            # Decode the truncated bytes
            partial_text = text_bytes[:truncated_length].decode('utf-8')
            
            # Find the last complete sentence
            sentences = re.split(r'(?<=[.!?])\s+', partial_text)
            if len(sentences) > 1:
                return ' '.join(sentences[:-1]) + '...'
                
            # If no sentence breaks, find last complete word
            words = partial_text.rsplit(' ', 1)
            if len(words) > 1:
                return words[0] + '...'
                
            # Last resort: just truncate at character boundary
            return partial_text + '...'
            
        except UnicodeDecodeError:
            truncated_length -= 1
            
    return ""

def is_ai_prompt_or_response(text: str) -> bool:
    """Detect if text is an AI prompt or response."""
    ai_indicators = [
        "I'm ready to process",
        "Please provide",
        "I will then apply",
        "Once you provide",
        "Here's how I would",
        "I will process",
        "Sample Input Text",
        "Processed Output Text",
        "Example of how I will",
        "Original Input:",
        "Processed Output:",
        "**Guidelines:**",
        "markdown```",
        "```markdown",
        "Explanation of Changes:",
        "template for processing"
    ]
    
    # Check for markdown code block markers
    if text.count("```") >= 2:
        return True
        
    # Check for AI conversation indicators
    text_lower = text.lower()
    if any(indicator.lower() in text_lower for indicator in ai_indicators):
        return True
        
    # Check for instructional/template content
    if "**" in text and any(word in text_lower for word in ["step", "guideline", "instruction", "template"]):
        return True
        
    return False

def clean_image_references(text: str) -> str:
    """Remove image references and clean up the text."""
    # Remove image markdown
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove image file references
    text = re.sub(r'_page_\d+_Picture_\d+\.jpeg', '', text)
    return text

def clean_section_title(title: str) -> str:
    """Clean section title of common artifacts."""
    # Remove markdown formatting
    title = re.sub(r'\*\*|\[|\]|`|#', '', title)
    # Remove image references
    title = clean_image_references(title)
    # Remove extra whitespace
    title = ' '.join(title.split())
    return title.strip()

def extract_document_metadata(sections: List[Tuple[str, List[str]]], doc_type: str, doc_id: str = "") -> Dict:
    """
    Extract only essential metadata.
    """
    return {'doc_id': doc_id}

def detect_content_features(text: str) -> Dict[str, bool]:
    """Detect special content features with error handling."""
    features = {
        'has_equations': False,
        'has_figures': False,
        'has_code': False
    }
    
    try:
        features['has_equations'] = bool(re.search(r'\$\$.*?\$\$|\$.*?\$', text, re.DOTALL))
    except Exception:
        pass
        
    try:
        features['has_figures'] = bool(re.search(r'(?i)figure|fig\.|table|tbl\.', text))
    except Exception:
        pass
        
    try:
        features['has_code'] = bool(re.search(r'```.*?```', text, re.DOTALL))
    except Exception:
        pass
    
    return features

def configure_gemini_model():
    """Configure Gemini model to only process text without adding content."""
    try:
        # Configure model with minimal output settings
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,  # No randomness
            top_p=0.0,  # No sampling
            top_k=1,  # Only most likely token
            candidate_count=1,  # Single response
            stop_sequences=["```", "#", "*", "[", "]"],  # Limited to 5 most important stop sequences
            max_output_tokens=32000  # Allow longer output for text cleaning
        )
        
        # Use default safety settings instead of explicitly disabling them
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",  # Using flash model for faster text processing
            generation_config=generation_config
        )
        return model
    except Exception as e:
        logger.error(f"Error configuring Gemini model: {e}")
        return None

def clean_text_with_gemini(text: str, model=None) -> str:
    """
    Clean text using Gemini model with focus on semantic preservation.
    Returns only the cleaned text without any additional content.
    
    Args:
        text: Text to clean
        model: Optional pre-configured Gemini model
        
    Returns:
        Cleaned text or empty string if cleaning fails
    """
    if not text or not isinstance(text, str):
        return ""
        
    try:
        if not model:
            model = configure_gemini_model()
            if not model:
                logger.warning("Failed to configure Gemini model, falling back to rule-based cleaning")
                return clean_text_for_embedding(text)
        
        # Configure prompt to force only cleaned text output
        prompt = """IMPORTANT: Your response must contain ONLY the cleaned text, without any explanations, markdown, or formatting.

Clean this academic text by:
1. Removing all markdown syntax (##, **, [], etc.)
2. Converting equations to plain text format
3. Standardizing citations and references to plain text
4. Preserving all technical terms and meaning
5. Maintaining paragraph structure
6. Removing redundant whitespace
7. Keeping only essential content

DO NOT:
- Add any explanations or comments
- Use any markdown or formatting
- Include any metadata or structure markers
- Add any new content
- Respond with anything except the cleaned text

Input text to clean:
"""
        
        # Get response with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt + text)
                if response and response.text:
                    cleaned = response.text.strip()
                    # Verify we got meaningful content back
                    if len(cleaned) < len(text) * 0.1:  # Too much was removed
                        logger.warning("Gemini cleaning removed too much content, retrying...")
                        continue
                    # Additional verification that we got only text
                    if '```' in cleaned or '#' in cleaned or '*' in cleaned or '[' in cleaned:
                        logger.warning("Gemini returned formatted text, cleaning with regex...")
                        cleaned = re.sub(r'[`#*\[\]]', '', cleaned)
                    return cleaned
            except Exception as e:
                logger.warning(f"Gemini cleaning attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
        logger.warning("Gemini cleaning failed, falling back to rule-based cleaning")
        return clean_text_for_embedding(text)
        
    except Exception as e:
        logger.error(f"Error in Gemini text cleaning: {str(e)}")
        return clean_text_for_embedding(text)

def validate_vector_for_pinecone(vector: Dict) -> bool:
    """
    Validate if a vector meets Pinecone's requirements.
    
    Args:
        vector: Vector dictionary with 'id', 'values', and 'metadata'
        
    Returns:
        bool: True if vector is valid for Pinecone
    """
    try:
        # Check vector values
        if not vector.get('values') or len(vector['values']) != 1024:
            logger.error(f"Invalid vector dimension for {vector.get('id')}")
            return False
            
        if not all(isinstance(x, float) for x in vector['values']):
            logger.error(f"Vector values must be floats for {vector.get('id')}")
            return False
            
        # Check metadata size (40KB limit)
        metadata_str = json.dumps(vector.get('metadata', {}))
        if len(metadata_str.encode('utf-8')) > 40000:  # 40KB in bytes
            logger.error(f"Metadata too large for {vector.get('id')}")
            return False
            
        # Validate text length in metadata
        text = vector.get('metadata', {}).get('text', '')
        if len(text.encode('utf-8')) > 20000:  # Keep text under 20KB to allow for other metadata
            logger.error(f"Text too long in metadata for {vector.get('id')}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating vector: {str(e)}")
        return False

def is_meaningful_content(text: str) -> bool:
    """
    Check if the text content is meaningful enough to create a vector.
    
    Args:
        text: Text content to validate
        
    Returns:
        bool: True if content is meaningful enough for vectorization
    """
    # Remove common header patterns
    text = re.sub(r'^[IVXLCDMivxlcdm]+\.\s*', '', text)  # Roman numerals
    text = re.sub(r'^\d+\.\s*', '', text)  # Arabic numerals
    text = re.sub(r'^[A-Za-z]\.\s*', '', text)  # Single letter headers
    
    # Remove common section header words
    header_words = ['introduction', 'methods', 'methodology', 'results', 
                   'discussion', 'conclusion', 'references', 'bibliography',
                   'abstract', 'background', 'appendix']
    
    cleaned_text = text.lower().strip()
    if cleaned_text in header_words:
        return False
        
    # Check if it's just a short title/header
    words = cleaned_text.split()
    if len(words) < 5:  # Too few words to be meaningful
        return False
        
    # Check if it has enough characters after cleaning
    if len(cleaned_text) < 50:  # Too short to be meaningful
        return False
        
    return True

def process_research_paper(paper_path: str, pc: Pinecone) -> List[Dict]:
    try:
        # Read and validate file
        logger.info(f"Reading file: {paper_path}")
        if not os.path.exists(paper_path):
            logger.error(f"Paper file not found: {paper_path}")
            return []
            
        try:
            with open(paper_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Successfully read file with utf-8 encoding")
        except UnicodeDecodeError:
            # Try alternate encodings
            logger.info("Trying alternate encodings...")
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(paper_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.warning(f"Used alternate encoding {encoding} for {paper_path}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error(f"Failed to decode file {paper_path} with any encoding")
                return []
                
        # Basic content validation
        content_length = len(content.strip())
        logger.info(f"Content length: {content_length} characters")
        if content_length < 100:  # Too short to be meaningful
            logger.warning(f"File content too short: {paper_path}")
            return []
            
        # Get document ID
        doc_id = os.path.basename(paper_path).replace('.md', '')
        
        # Parse into sections
        logger.info(f"Starting section parsing for {doc_id}")
        sections = parse_academic_markdown(content, 'paper')  # Default to paper type
        logger.info(f"Found {len(sections)} sections")
        if not sections:
            logger.warning(f"No valid sections found in {doc_id}")
            return []
            
        # Process sections
        vectors = []
        total_sections = len(sections)
        
        logger.info(f"Starting section processing ({total_sections} sections)")
        for i, (section_text, section_hierarchy) in enumerate(sections, 1):
            logger.info(f"Processing section {i}/{total_sections} of {doc_id}")
            
            # Clean text using basic markdown cleaning
            clean_text = clean_text_for_embedding(section_text)
            if not clean_text:
                logger.warning(f"Empty section {i} after cleaning in {doc_id}")
                continue
                
            # Skip sections that aren't meaningful
            if not is_meaningful_content(clean_text):
                logger.info(f"Skipping non-meaningful section {i} in {doc_id}")
                continue
            
            # Truncate text if needed (ensure it fits in metadata)
            clean_text = truncate_text(clean_text, max_bytes=20000)  # 20KB limit for text
            
            # Get embeddings
            logger.info(f"Getting embeddings for section {i}")
            embeddings = get_embeddings_batch([clean_text], pc)
            if not embeddings:
                logger.warning(f"Failed to get embeddings for section {i} in {doc_id}")
                continue
            logger.info(f"Successfully got embeddings for section {i}")
            
            # Create vector with minimal metadata
            chunk_id = f"{doc_id}_chunk_{i}"
            
            vector = {
                'id': chunk_id,
                'values': embeddings[0],
                'metadata': {
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'text': clean_text,
                    'section_path': ' > '.join(section_hierarchy)  # Add section hierarchy for context
                }
            }
            
            # Validate vector before adding
            if validate_vector_for_pinecone(vector):
                vectors.append(vector)
                logger.info(f"Successfully processed section {i}")
            else:
                logger.warning(f"Skipping invalid vector for section {i}")
            
        if not vectors:
            logger.warning(f"No valid vectors generated for {doc_id}")
            return []
            
        logger.info(f"Successfully processed {len(vectors)} sections from {doc_id}")
        return vectors
        
    except Exception as e:
        logger.error(f"Error processing paper {paper_path}: {str(e)}")
        logger.error(f"Stack trace: ", exc_info=True)
        return []

def upsert_chunks_with_retry(index, vectors: List[Dict], max_retries=5, base_delay=2.0, batch_size=50):
    """
    Upsert chunks with improved retry logic, rate limiting, and smaller batch sizes.
    
    Args:
        index: Pinecone index
        vectors: List of vectors to upsert
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        batch_size: Maximum number of vectors per upsert
    """
    if not vectors:
        logger.warning("No valid vectors to upsert")
        return False
        
    rate_limiters = APIRateLimiters()
    
    # Split vectors into batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        logger.debug(f"Upserting batch {i//batch_size + 1} of {(len(vectors) + batch_size - 1)//batch_size}")
        
        # Apply rate limiting
        wait_time = rate_limiters.pinecone_limiter.acquire()
        if wait_time > 0:
            logger.info(f"Rate limiting: waiting {wait_time:.1f}s before upserting batch")
            time.sleep(wait_time)
        
        for attempt in range(max_retries):
            try:
                index.upsert(
                    vectors=batch,
                    namespace="research_papers"
                )
                # Add delay between successful batches
                time.sleep(1)
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to upsert batch after {max_retries} attempts: {str(e)}")
                    return False
                    
                delay = exponential_backoff(attempt, base_delay)
                logger.warning(f"Upsert attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
    
    return True

def get_processed_chunks(index) -> dict:
    """Get mapping of processed papers and their chunks from Pinecone index."""
    try:
        response = index.query(
            vector=[0] * 1024,  # Dummy vector
            top_k=10000,  # Get as many as possible
            include_metadata=True,
            namespace="research_papers"
        )
        
        processed_chunks = {}
        for match in response.matches:
            if 'metadata' in match and 'doc_id' in match.metadata:
                doc_id = match.metadata['doc_id']
                chunk_id = match.metadata['chunk_id']
                if doc_id not in processed_chunks:
                    processed_chunks[doc_id] = set()
                processed_chunks[doc_id].add(chunk_id)
        
        return processed_chunks
    except Exception as e:
        logger.error(f"Error fetching processed chunks: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Process academic papers and create semantic search index.")
    parser.add_argument('-d', '--papers_directory', required=True,
                        help='Path to the directory containing academic paper Markdown files')
    parser.add_argument('-i', '--index_name', required=True,
                        help='Name of the semantic search index')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force reprocessing of already processed papers')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for vector upserts (default: 50)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between processing papers in seconds (default: 1.0)')
    args = parser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    embedding_model = "llama-text-embed-v2"
    dimension = 1024

    try:
        index_list = pc.list_indexes()
        if args.index_name not in [idx.name for idx in index_list]:
            logger.info(f"Creating new index: {args.index_name}")
            pc.create_index(
                name=args.index_name,
                dimension=dimension,
                spec=ServerlessSpec(
                    cloud=CloudProvider.AWS,
                    region=AwsRegion.US_EAST_1
                ),
                vector_type=VectorType.DENSE
            )
    except Exception as e:
        logger.error(f"❌ Error creating/checking index: {str(e)}")
        return

    semantic_index = pc.Index(args.index_name)
    
    # Get processed papers from Pinecone
    processed_chunks = {}
    if not args.force:
        logger.info("Checking existing papers in Pinecone...")
        processed_chunks = get_processed_chunks(semantic_index)
    
    logger.info(f"\n📊 Processing Status:")
    logger.info(f"  • Previously processed papers: {len(processed_chunks)}")
    
    paper_files = [f for f in os.listdir(args.papers_directory) if f.endswith('.md')]
    remaining_papers = [f for f in paper_files 
                       if f.replace('.md', '') not in processed_chunks]
    
    logger.info(f"  • Total papers found: {len(paper_files)}")
    logger.info(f"  • Papers to process: {len(remaining_papers)}")
    logger.info(f"  • Papers to skip: {len(paper_files) - len(remaining_papers)}")
    
    if not remaining_papers:
        logger.info("\n✨ All papers have already been processed!")
        return
        
    successful_papers = 0
    failed_papers = 0
    
    logger.info("\n🚀 Starting paper processing...")
    
    try:
        for i, filename in enumerate(remaining_papers, 1):
            try:
                doc_id = filename.replace('.md', '')
                logger.info(f"\n[{i}/{len(remaining_papers)}] 📝 Processing: {filename}")
                
                # Skip if already processed (double-check with Pinecone)
                if doc_id in processed_chunks:
                    logger.info(f"⏩ Skipping already processed paper: {filename}")
                    continue
                
                paper_path = os.path.join(args.papers_directory, filename)
                paper_vectors = process_research_paper(paper_path, pc)
                
                if paper_vectors:
                    logger.info(f"📤 Upserting {len(paper_vectors)} vectors...")
                    if upsert_chunks_with_retry(semantic_index, paper_vectors, batch_size=args.batch_size):
                        # Update local tracking
                        processed_chunks[doc_id] = set(v['metadata']['chunk_id'] for v in paper_vectors)
                        successful_papers += 1
                        logger.info(f"✅ Successfully processed and upserted: {filename}")
                    else:
                        failed_papers += 1
                        logger.error(f"❌ Failed to upsert vectors for: {filename}")
                else:
                    failed_papers += 1
                    logger.error(f"❌ No vectors generated for: {filename}")
                
                # Progress summary
                success_rate = (successful_papers / i) * 100
                logger.info(f"\n📊 Progress Summary:")
                logger.info(f"  • Completed: {i}/{len(remaining_papers)} ({i/len(remaining_papers)*100:.1f}%)")
                logger.info(f"  • Success rate: {success_rate:.1f}%")
                logger.info(f"  • Successful: {successful_papers}")
                logger.info(f"  • Failed: {failed_papers}")
                
                # User-configurable delay between papers
                time.sleep(args.delay)
                
            except Exception as e:
                failed_papers += 1
                logger.error(f"❌ Error processing {filename}: {str(e)}")
                continue
                
    except KeyboardInterrupt:
        logger.info("\n⚠️  Processing interrupted by user")
        return

    logger.info(f"\n🎉 Processing Complete!")
    logger.info(f"📊 Final Results:")
    logger.info(f"  ✅ Successfully processed: {successful_papers} papers")
    logger.info(f"  ❌ Failed to process: {failed_papers} papers")
    logger.info(f"  ⏩ Skipped: {len(paper_files) - len(remaining_papers)} papers")
    
    if failed_papers > 0:
        logger.info(f"\n⚠️  Some papers failed to process. Check the logs for details.")

if __name__ == "__main__":
    main()