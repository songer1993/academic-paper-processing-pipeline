#!/usr/bin/env python3
"""
Convert PDF papers to markdown files using marker_single.
Uses Gemini for enhanced text extraction and formatting.
Requires GEMINI_API_KEY in .env file when using --use_llm.
"""

import os
import argparse
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
if not load_dotenv():
    logger.warning("No .env file found. Make sure to set GEMINI_API_KEY if using --use_llm")

def process_pdf(pdf_path, output_dir, use_llm=True, llm_service="marker.services.gemini.GoogleGeminiService"):
    """Process a single PDF file using marker_single."""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "marker_single",
        str(pdf_path),
        "--output_dir", str(output_dir)
    ]
    
    if use_llm:
        cmd.extend(["--use_llm"])
        cmd.extend(["--llm_service", llm_service])
    
    # Set up environment with API key
    env = os.environ.copy()
    
    # Get Gemini API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        env["GEMINI_API_KEY"] = api_key
        # Also pass it as a command line argument for marker
        cmd.extend(["--gemini_api_key", api_key])
    elif use_llm:
        logger.error("GEMINI_API_KEY not found in environment variables. Set it in .env file.")
        return
    
    try:
        subprocess.run(cmd, check=True, env=env)
        logger.info(f"Successfully processed: {pdf_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {pdf_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="""
    Convert PDF papers to markdown files using marker_single.
    Uses Gemini for enhanced text extraction and formatting when --use_llm is enabled.
    Requires GEMINI_API_KEY in .env file when using --use_llm.
    """)
    parser.add_argument("--input_dir", type=str, required=True, 
                      help="Path to a single PDF file or directory containing PDF files")
    parser.add_argument("--output_dir", type=str, required=True, 
                      help="Output directory for generated markdown files")
    parser.add_argument("--use_llm", action="store_true", 
                      help="Use Gemini for enhanced extraction (requires GEMINI_API_KEY)")
    parser.add_argument("--llm_service", type=str, 
                      default="marker.services.gemini.GoogleGeminiService", 
                      help="LLM service to use (default: Gemini)")
    parser.add_argument("--debug", action="store_true", 
                      help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
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
        # Get list of PDF files from directory
        pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found at {input_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    
    # Process each PDF file
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        logger.info(f"\nProcessing: {pdf_file}")
        try:
            process_pdf(
                pdf_file,
                output_dir,
                use_llm=args.use_llm,
                llm_service=args.llm_service
            )
            successful += 1
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {e}")
            failed += 1
    
    logger.info(f"Processing complete. Successfully processed: {successful}, Failed: {failed}")

if __name__ == "__main__":
    main() 