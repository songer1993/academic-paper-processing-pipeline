#!/usr/bin/env python3

import os
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def process_pdf(pdf_path, output_dir, use_llm=True, llm_service="marker.services.gemini.GoogleGeminiService", gemini_api_key=None):
    """Process a single PDF file using marker_single."""
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
    
    # Try to get API key from arguments or environment
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    if api_key:
        env["GEMINI_API_KEY"] = api_key
        # Also pass it as a command line argument for marker
        cmd.extend(["--gemini_api_key", api_key])
    elif use_llm:
        print("Warning: No Gemini API key found. Set it via --gemini_api_key or GEMINI_API_KEY in .env")
        return
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"Successfully processed: {pdf_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {pdf_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch process PDFs using marker_single")
    parser.add_argument("--input_dir", type=str, help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed files")
    parser.add_argument("--use_llm", action="store_true", help="Use LLM for processing")
    parser.add_argument("--llm_service", type=str, default="marker.services.gemini.GoogleGeminiService", 
                      help="LLM service to use")
    parser.add_argument("--gemini_api_key", type=str, help="Gemini API key (or set GEMINI_API_KEY in .env)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of PDF files
    input_dir = Path(args.input_dir)
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file}")
        process_pdf(
            pdf_file,
            output_dir,
            use_llm=args.use_llm,
            llm_service=args.llm_service,
            gemini_api_key=args.gemini_api_key
        )

if __name__ == "__main__":
    main() 