# Academic Paper Processing Pipeline

This repository contains tools to process academic papers from PDFs to searchable vectors in Pinecone. It uses [marker](https://github.com/VikParuchuri/marker) for high-quality PDF to markdown conversion and integrates with your reference management system through BibTeX files.

## Prerequisites

- Python 3.8+
- Pinecone account and API key
- Google Gemini API key (required for both scripts)
  - PDF conversion: Enhanced text extraction
  - Vector upload: Content filtering
- [Zotero](https://www.zotero.org/) with [Better BibTeX plugin](https://retorque.re/zotero-better-bibtex/) installed
  - Required for proper file paths in BibTeX export
  - Ensures consistent citation keys
  - Maintains file attachments metadata

## Installation

```bash
# Clone repository
git clone https://github.com/songer1993/academic-paper-processing-pipeline
cd academic-paper-processing-pipeline

# Install dependencies
pip install -r requirements.txt

# Verify marker installation
marker_single --help
```

## Configuration

Create a `.env` file in the root directory:
```ini
# Required for both scripts - used for enhanced PDF extraction and content filtering
GEMINI_API_KEY=your_gemini_key

# Required for vector storage in second script
PINECONE_API_KEY=your_pinecone_key
```

## Usage

The pipeline consists of two main steps, both utilizing Google's Gemini API:

### 1. Convert PDFs to Markdown

Uses marker for high-quality PDF to markdown conversion. Requires GEMINI_API_KEY for enhanced text extraction and formatting.

```bash
python scripts/convert_pdfs_to_mds.py \
  --input_dir path/to/pdfs \
  --output_dir path/to/output \
  [--debug]
```

Options:
- `--input_dir`: Path to a single PDF or directory of PDFs
- `--output_dir`: Where to save the markdown files
- `--debug`: Enable debug logging

### 2. Upload to Pinecone

Processes markdown files into chunks, uses Gemini to filter out non-research content, and uploads to Pinecone with metadata from your BibTeX file. **Important**: Export your BibTeX file using Zotero's Better BibTeX plugin to ensure proper file paths and metadata.

```bash
python scripts/upload_mds_to_pinecone_by_paragraphs.py \
  --input_dir path/to/markdown/files \
  --bibtex_file path/to/references.bib \
  --index_name your_pinecone_index \
  [--namespace papers] \
  [--debug]
```

Options:
- `--input_dir`: Path to a single markdown file or directory
- `--bibtex_file`: Path to BibTeX file exported from Zotero using Better BibTeX
- `--index_name`: Name of your Pinecone index
- `--namespace`: Optional Pinecone namespace
- `--debug`: Enable debug logging for chunk processing

### Vector Structure

Each document is split into chunks of 1-2 paragraphs. Each chunk is stored in Pinecone as a vector with:

1. `id`: Unique identifier in format `{citation_key}_chunk_{index}`
2. `values`: Vector embedding generated using Pinecone's "llama-text-embed-v2" model
3. `metadata`:
   - `text`: The actual content of the chunk
   - `title`: Paper title
   - `author`: Author(s)
   - `year`: Publication year
   - `abstract`: Paper abstract
   - `doi`: Digital Object Identifier (if available)
   - `venue`: Publication venue (journal/conference/publisher)
   - `chunk_index`: Position of this chunk in the document (0-based)
   - `total_chunks`: Total number of chunks in the document

The venue field is intelligently extracted based on the entry type:
- For articles: journal name
- For conference papers: conference/proceedings name
- For books: publisher name
- For theses: school name

### Content Processing

The upload script:
1. Splits documents into 1-2 paragraph chunks
2. Uses Gemini to filter out non-research content (acknowledgments, references, etc.)
3. Maintains bibliographic metadata from BibTeX entries
4. Generates embeddings for semantic search
5. Uploads to your Pinecone index

## Notes

- Both scripts support single file or directory input
- The pipeline preserves bibliographic metadata through BibTeX integration
- Content filtering removes non-research sections like acknowledgments and references
- All scripts automatically load configuration from .env file
- Debug logging available for troubleshooting

## License

MIT License - See LICENSE file for details 