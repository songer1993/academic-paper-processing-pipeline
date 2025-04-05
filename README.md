# Academic Paper Processing Pipeline

This repository contains a set of tools to process academic papers from PDFs to searchable vectors in Pinecone. The workflow integrates with Zotero for reference management and uses [marker](https://github.com/VikParuchuri/marker) (installed as `marker-pdf`) for high-quality PDF to markdown conversion.

## Workflow Overview

1. **Convert PDFs to Markdown**:
   - Use `scripts/convert_pdfs.py` to convert PDFs to markdown
   - Specify any input and output directories you prefer
   - Uses marker for high-quality conversion

2. **Upload to Pinecone**:
   - Export .bib file from Zotero
   - Use `scripts/upsert_to_pinecone.py` to:
     - Process markdown files
     - Extract bibliographic metadata
     - Generate embeddings
     - Upload to Pinecone index
   - Each document is stored with:
     - Full text content
     - Bibliographic metadata (title, authors, year, etc.)
     - Vector embedding for semantic search

3. **Configure Semantic Search in Cursor**:
   - Create `.cursor/settings.json` in your project root:
     ```json
     {
         "mcpServers": {
             "mcp-pinecone": {
                 "command": "uvx",
                 "args": [
                     "mcp-pinecone",
                     "--index-name",
                     "your-index-name",
                     "--api-key",
                     "your-pinecone-api-key"
                 ]
             }
         }
     }
     ```
   - This project-specific configuration ensures the correct Pinecone index is used
   - Usage in Cursor:
     - Ask questions about your papers in natural language
     - Example: "Find papers discussing haptic feedback in VR"
     - Results include relevant excerpts with citation keys
     - Click results to view full paper content

## Prerequisites

- Python 3.8+
- [marker](https://github.com/VikParuchuri/marker) installed (as `marker-pdf` package)
- [Zotero](https://www.zotero.org/) for reference management
- Pinecone API key
- (Optional) Google Gemini API key for improved conversion quality

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/paper-processing
cd paper-processing

# Install dependencies
pip install -r requirements.txt

# Verify marker installation
marker_single --help
```

## Configuration

### Environment Variables
Create a `.env` file in the root directory with your API keys:
```ini
# Required
PINECONE_API_KEY=your_pinecone_key

# Optional - for better quality conversion
GEMINI_API_KEY=your_gemini_key

# Optional - for GPU acceleration
TORCH_DEVICE=cuda
```

All scripts will automatically load these environment variables from the `.env` file. You can also override them:
- Via command line arguments (for `convert_pdfs.py`)
- Via environment variables in your shell

Priority order:
1. Command line arguments (if available)
2. Environment variables in shell
3. Values from `.env` file

## Directory Structure

```
.
├── input/
│   ├── pdfs/           # Place your PDFs here
│   │   └── .gitkeep    # Preserves empty directory
│   └── bibliography/   # Place your .bib file here
│       └── .gitkeep    # Preserves empty directory
├── output/
│   ├── raw_md/        # Converted markdown files
│   │   └── .gitkeep    # Preserves empty directory
│   └── renamed_md/    # Renamed markdown files
│       └── .gitkeep    # Preserves empty directory
├── scripts/
│   ├── convert_pdfs.py      # PDF to markdown conversion
│   ├── rename_by_citekey.py # Rename using citation keys
│   └── upsert_vectors.py    # Create and upload vectors
└── README.md
```

The repository includes `.gitkeep` files to preserve empty directories in version control. These directories are:
- `input/pdfs/`: Place your exported PDFs from Zotero here
- `input/bibliography/`: Place your exported .bib file from Zotero here
- `output/raw_md/`: Initial markdown conversions will be stored here
- `output/renamed_md/`: Final renamed markdown files will be stored here

## Usage

1. **Convert PDFs**:
   ```bash
   python scripts/convert_pdfs.py \
     --input_dir path/to/your/pdfs \
     --output_dir path/to/output/markdown \
     --use_llm
   ```

2. **Upload to Pinecone**:
   ```bash
   python scripts/upsert_to_pinecone.py \
     --markdown_dir path/to/markdown/files \
     --bibtex_file path/to/your/library.bib \
     --index_name your_pinecone_index \
     --batch_size 100
   ```

## Notes

- The pipeline preserves Zotero's metadata through citation keys
- Use marker's `--use_llm` flag for higher quality conversion
- Pinecone vectors include full text and bibliographic metadata
- Files are processed incrementally - already processed files are skipped
- For complex PDFs with tables or forms, using `--use_llm` is recommended
- If you encounter garbled text, use the `--force_ocr` flag
- All scripts automatically load configuration from .env file

## License

MIT License - See LICENSE file for details 