# Academic Paper Processing Pipeline

This repository contains a set of tools to process academic papers from PDFs to searchable vectors in Pinecone. The workflow integrates with Zotero for reference management and uses [marker](https://github.com/VikParuchuri/marker) (installed as `marker-pdf`) for high-quality PDF to markdown conversion.

## Workflow Overview

1. **Collect PDFs**: 
   - Save academic papers in Zotero
   - Export PDFs with Zotero's file naming convention
   - Place PDFs in `input/pdfs` directory

2. **Convert to Markdown**:
   - Use `scripts/convert_pdfs.py` to batch convert PDFs to markdown
   - Outputs go to `output/raw_md`
   - Uses marker for high-quality conversion

3. **Rename by Citation Key**:
   - Export .bib file from Zotero
   - Copy markdown files to `output/renamed_md`
   - Use `scripts/rename_by_citekey.py` to rename based on citation keys

4. **Upload to Pinecone**:
   - Use `scripts/upsert_vectors.py` to chunk and vectorize content
   - Creates searchable vectors in Pinecone index
   - Enables semantic search via MCP servers in Cursor

## Prerequisites

- Python 3.8+
- [marker](https://github.com/VikParuchuri/marker) installed (as `marker-pdf` package)
- [Zotero](https://www.zotero.org/) for reference management
- Pinecone API key
- (Optional) Google Gemini API key for improved conversion quality

## Installation

You can install dependencies in two ways:

### Using pip (Simple)
```bash
# Clone repository
git clone https://github.com/yourusername/paper-processing
cd paper-processing

# Install dependencies
pip install -r requirements.txt

# Verify marker installation
marker_single --help
```

### Using poetry (Recommended by marker)
```bash
# Clone repository
git clone https://github.com/yourusername/paper-processing
cd paper-processing

# Install poetry if you haven't
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies including marker
poetry install

# Verify marker installation
poetry run marker_single --help
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
   # Using environment variables from .env
   python scripts/convert_pdfs.py \
     --input_dir input/pdfs \
     --output_dir output/raw_md \
     --use_llm

   # Or override with command line
   python scripts/convert_pdfs.py \
     --input_dir input/pdfs \
     --output_dir output/raw_md \
     --use_llm \
     --gemini_api_key YOUR_API_KEY
   ```

2. **Rename Files**:
   ```bash
   python scripts/rename_by_citekey.py \
     --input_dir output/raw_md \
     --output_dir output/renamed_md \
     --bibtex_file input/bibliography/library.bib
   ```

3. **Upload to Pinecone**:
   ```bash
   # Uses PINECONE_API_KEY from .env
   python scripts/upsert_vectors.py \
     --papers_directory output/renamed_md \
     --index_name your_index_name
   ```

## Notes

- The pipeline preserves Zotero's metadata through citation keys
- Use marker's `--use_llm` flag for higher quality conversion
- Pinecone vectors include section hierarchy for better context
- Files are processed incrementally - already processed files are skipped
- For complex PDFs with tables or forms, using `--use_llm` is recommended
- If you encounter garbled text, use the `--force_ocr` flag
- All scripts automatically load configuration from .env file

## License

MIT License - See LICENSE file for details 