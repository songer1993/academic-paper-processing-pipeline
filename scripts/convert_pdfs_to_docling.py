#!/usr/bin/env python3
"""
Convert PDF papers to Docling documents and save as JSON/markdown.
Uses docling for high-quality PDF to structured document conversion with hierarchical chunking.
No external API dependencies required.
"""

import os
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
import torch

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Suppress common warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file (optional - no API keys required)
load_dotenv()

def check_gpu_availability():
    """Check if GPU acceleration is available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA GPU is enabled: {torch.cuda.get_device_name(0)}")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("MPS GPU is enabled.")
        return device
    else:
        logger.warning("No GPU acceleration available. Using CPU.")
        return torch.device("cpu")

def setup_document_converter() -> DocumentConverter:
    """Setup document converter with appropriate pipeline options."""
    
    # Configure pipeline options for PDF processing
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Enable OCR for scanned PDFs
    pipeline_options.do_table_structure = True  # Extract table structure
    pipeline_options.table_structure_options.do_cell_matching = True
    
    # Configure document converter - using simplified API
    doc_converter = DocumentConverter()
    
    return doc_converter

def convert_pdf_to_docling(pdf_path: Path, doc_converter: DocumentConverter, max_retries: int = 2) -> Optional[object]:
    """Convert a single PDF to Docling document with retry logic."""
    import time
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Converting PDF: {pdf_path} (attempt {attempt + 1}/{max_retries + 1})")
            
            # Convert PDF to Docling document
            result = doc_converter.convert(pdf_path)
            
            if result.document:
                logger.info(f"Successfully converted: {pdf_path}")
                return result.document
            else:
                logger.error(f"Failed to convert: {pdf_path}")
                return None
                
        except Exception as e:
            if "SSLError" in str(e) or "HTTPSConnectionPool" in str(e) or "MaxRetryError" in str(e):
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 5  # Progressive backoff
                    logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Network error after {max_retries + 1} attempts for {pdf_path}")
            else:
                logger.error(f"Error converting {pdf_path}: {e}")
                
            if attempt == max_retries:
                return None

def save_docling_document(document, pdf_path: Path, output_dir: Path, 
                         save_format: str = "both") -> bool:
    """Save Docling document in specified format(s)."""
    try:
        # Create output subdirectory named after the PDF (without extension)
        pdf_name = pdf_path.stem
        doc_output_dir = output_dir / pdf_name
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        
        success = True
        
        if save_format in ["markdown", "both"]:
            # Save as markdown
            markdown_path = doc_output_dir / f"{pdf_name}.md"
            try:
                markdown_content = document.export_to_markdown()
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                logger.info(f"Saved markdown: {markdown_path}")
            except Exception as e:
                logger.error(f"Error saving markdown for {pdf_name}: {e}")
                success = False
        
        if save_format in ["json", "both"]:
            # Save as JSON (Docling document format)
            json_path = doc_output_dir / f"{pdf_name}.json"
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(document.export_to_dict(), f, indent=2, ensure_ascii=False)
                logger.info(f"Saved JSON: {json_path}")
            except Exception as e:
                logger.error(f"Error saving JSON for {pdf_name}: {e}")
                success = False
        
        return success
        
    except Exception as e:
        logger.error(f"Error saving document for {pdf_path}: {e}")
        return False

def process_pdf_batch(pdf_files: List[Path], output_dir: Path, 
                     save_format: str = "both") -> tuple:
    """Process a batch of PDF files."""
    
    # Check GPU availability
    device = check_gpu_availability()
    
    # Setup document converter
    doc_converter = setup_document_converter()
    
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        logger.info(f"\nProcessing: {pdf_file}")
        
        # Convert PDF to Docling document
        document = convert_pdf_to_docling(pdf_file, doc_converter)
        
        if document:
            # Save the document
            if save_docling_document(document, pdf_file, output_dir, save_format):
                successful += 1
            else:
                failed += 1
        else:
            failed += 1
    
    return successful, failed

def main():
    parser = argparse.ArgumentParser(description="""
    Convert PDF papers to Docling documents with hierarchical structure.
    Uses docling for high-quality PDF to structured document conversion.
    No external API dependencies required.
    """)
    parser.add_argument("--input_dir", type=str, required=True, 
                      help="Path to a single PDF file or directory containing PDF files")
    parser.add_argument("--output_dir", type=str, required=True, 
                      help="Output directory for generated documents")
    parser.add_argument("--save_format", choices=["markdown", "json", "both"], 
                      default="both", help="Output format (default: both)")
    parser.add_argument("--simple_mode", action="store_true",
                      help="Use simple mode (basic parsing without downloading models)")
    parser.add_argument("--debug", action="store_true", 
                      help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("docling").setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle both single file and directory input
    input_dir = Path(args.input_dir)
    if input_dir.is_file():
        if not input_dir.suffix.lower() == '.pdf':
            logger.error(f"Input file must be a PDF: {input_dir}")
            return
        pdf_files = [input_dir]
    else:
        # Get list of PDF files from directory and subdirectories
        pdf_files = []
        for pdf_path in input_dir.rglob("*.pdf"):
            pdf_files.append(pdf_path)
    
    if not pdf_files:
        logger.error(f"No PDF files found at {input_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    
    if args.simple_mode:
        logger.info("Running in simple mode - basic PDF parsing without advanced models")
    
    # Process PDF files
    successful, failed = process_pdf_batch(
        pdf_files, 
        output_dir, 
        save_format=args.save_format
    )
    
    logger.info(f"Processing complete. Successfully processed: {successful}, Failed: {failed}")

if __name__ == "__main__":
    main() 