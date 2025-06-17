# Academic Paper Processing Pipeline

This repository contains tools to process academic papers from PDFs to searchable vectors in Pinecone. It uses [Docling](https://github.com/DS4SD/docling) for high-quality PDF to structured document conversion with hierarchical chunking and extracts metadata directly from Zotero filenames.

## Prerequisites

- Python 3.8+
- Pinecone account and API key
- [Zotero](https://www.zotero.org/) for organizing PDFs with standardized filenames (`Author et al. - Year - Title.pdf`)
- GPU recommended for faster processing (CUDA or Apple Silicon)

## Installation

### Using uv (Recommended)

```bash
# Clone repository
git clone https://github.com/songer1993/academic-paper-processing-pipeline
cd academic-paper-processing-pipeline

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Verify installation
uv run python -c "from docling.document_converter import DocumentConverter; print('Docling successfully installed')"
```

### Using pip (Alternative)

```bash
# Clone repository
git clone https://github.com/songer1993/academic-paper-processing-pipeline
cd academic-paper-processing-pipeline

# Install dependencies
pip install -r requirements.txt

# Verify docling installation
python -c "from docling.document_converter import DocumentConverter; print('Docling successfully installed')"
```

## Configuration

Create a `.env` file in the root directory:
```ini
# Required for vector storage
PINECONE_API_KEY=your_pinecone_key

# Required for DSPy-based chunk quality assessment with Gemini
GOOGLE_API_KEY=your_google_api_key
```

## Usage

The pipeline consists of two main steps using Docling for enhanced PDF processing and hierarchical chunking:

### 0. Prepare Input Files (Zotero)

Before running the scripts, ensure your PDFs are organized with standardized Zotero filenames:

1.  **Organize PDFs:** Ensure your PDFs are stored by Zotero with standardized filenames. Zotero automatically uses the format `Author et al. - Year - Title.pdf` for PDF attachments.
2.  **Verify Filename Format:** Check that your PDF files follow the pattern:
    - `De Zambotti et al. - 2019 - Wearable Sleep Technology in Clinical and Research Settings.pdf`
    - `Roberts et al. - 2020 - Detecting sleep using heart rate and motion data from multisensor consumer-grade wearables.pdf`
3.  **Locate PDF Directory:** Identify the directory containing your PDFs (this will be the `--input_dir` for the first script).

**No BibTeX export needed!** The pipeline extracts all metadata (author, year, title) directly from the standardized Zotero filenames, eliminating the complexity of BibTeX file matching.

### 1. Convert PDFs to Docling Documents

Uses Docling for high-quality PDF to structured document conversion with hierarchical layout understanding. Uses GPU acceleration when available for faster processing.

```bash
# Using uv
uv run python scripts/convert_pdfs_to_docling.py \
  --input_dir path/to/pdfs \
  --output_dir path/to/output \
  [--save_format both] \
  [--debug]

# Using pip
python scripts/convert_pdfs_to_docling.py \
  --input_dir path/to/pdfs \
  --output_dir path/to/output \
  [--save_format both] \
  [--debug]
```

Options:
- `--input_dir`: Path to a single PDF or directory of PDFs
- `--output_dir`: Where to save the structured documents
- `--save_format`: Output format - "markdown", "json", or "both" (default: both)
- `--debug`: Enable debug logging

### 2. Upload to Pinecone with Hierarchical Chunking and Quality Assessment

Processes Docling documents using hierarchical chunking based on document structure, extracts metadata directly from Zotero filenames, and uploads to Pinecone. **No BibTeX file needed** - metadata is extracted from standardized Zotero filenames (`Author et al. - Year - Title.pdf`). **Intelligent quality filtering** - uses DSPy-powered Gemini classifier to filter out low-quality chunks (manuscript headers, author affiliations, funding statements, reference lists) while preserving substantive research content.

```bash
# Using uv
uv run python scripts/upload_docling_to_pinecone.py \
  --input_dir path/to/docling/files \
  --index_name your_pinecone_index \
  [--namespace papers] \
  [--debug]

# Using pip
python scripts/upload_docling_to_pinecone.py \
  --input_dir path/to/docling/files \
  --index_name your_pinecone_index \
  [--namespace papers] \
  [--debug]
```

Options:
- `--input_dir`: Path to a single Docling JSON file or directory containing Docling files
- `--index_name`: Name of your Pinecone index
- `--namespace`: Optional Pinecone namespace
- `--debug`: Enable debug logging for chunk processing

### 3. Configure Cursor MCP Servers (Optional)

If you intend to use Cursor's advanced features like the Memory tool or potentially custom Pinecone integrations directly within the IDE, you might need to configure MCP (Multi-Component Process) servers. This is typically done in a `mcp.json` file located in the `./.cursor/` directory within your workspace root.

Create or update the `./.cursor/mcp.json` file with an `mcpServers` object. Here is a generalized example for a custom Pinecone tool integration:

```json
{
    "mcpServers": {
        "mcp-pinecone": { // Example for a custom Pinecone tool 
            "command": "uvx", // Or another command runner like 'docker', 'python', etc.
            "args": [
                "mcp-pinecone", // The command or script to run
                "--index-name",
                "your_pinecone_index_name", // Replace with your index name
                "--api-key",
                "your_pinecone_api_key" // Replace with your Pinecone API key
                // Add other necessary arguments for your specific tool
            ]
        }
        // Add configurations for other MCP servers as needed
    }
}
```

**Notes:**

*   Replace placeholder values like `your_pinecone_index_name` and `your_pinecone_api_key` with your actual configuration.
*   The specific `command` and `args` will depend heavily on the tool you are integrating and how it's packaged (e.g., Docker container, Python script).
*   Refer to the documentation of the specific Cursor feature or tool for the exact configuration required.

### Vector Structure

Each document is processed using Docling's hierarchical chunking based on document structure. Each chunk is stored in Pinecone as a vector with:

1. `id`: Sophisticated citation key format `{citekey}_chunk_{index}` (e.g., `dezambottiStateScienceRecommendations2024_chunk_0`)
2. `values`: Vector embedding generated using Pinecone's "llama-text-embed-v2" model (pure content, no metadata mixing)
3. `metadata`:
   - `text`: The actual content of the chunk
   - `title`: Paper title (extracted from filename)
   - `author`: Author(s) (extracted from filename)
   - `year`: Publication year (extracted from filename)
   - `citekey`: Sophisticated citation key (auto-generated)
   - `chunk_index`: Position of this chunk in the document (0-based)

**Citation Key Format:** `firstAuthorTitleWordsYear`
- `De Zambotti et al. - 2024 - State of the science and recommendations` 
- → `dezambottiStateScienceRecommendations2024`

### Content Processing

The upload script:
1. Uses Docling's hierarchical chunker to split documents based on structure
2. Extracts metadata directly from Zotero filenames with sophisticated citation keys
3. Preserves document structure and heading information
4. **Intelligent quality assessment** - uses DSPy framework with Gemini to classify chunks
5. **Filters low-quality content** - removes manuscript headers, reference lists, funding statements, author affiliations
6. **Preserves research content** - keeps substantive academic content, methods, results, discussions
7. Generates embeddings for semantic search (pure content, no metadata mixing)
8. Uploads to your Pinecone index in batches

## Notes

- Both scripts support single file or directory input
- The pipeline uses Docling's advanced PDF processing with GPU acceleration when available
- Hierarchical chunking preserves document structure and context
- **Intelligent quality filtering** - DSPy-powered classifier optimizes content quality for semantic search
- **Gemini API required** - for chunk quality assessment (in addition to Pinecone API)
- Sophisticated citation keys: `dezambottiStateScienceRecommendations2024`
- Use uv for faster, more reliable dependency management
- Debug logging available for troubleshooting and quality assessment insights

### GPU Acceleration

For optimal performance, the pipeline supports:
- **CUDA GPU** acceleration for NVIDIA GPUs
- **MPS (Metal Performance Shaders)** acceleration for Apple Silicon Macs
- Automatic fallback to CPU processing if no GPU is available

### Natural Language Search with mcp-pinecone

This pipeline integrates with [mcp-pinecone](https://github.com/sirmews/mcp-pinecone) to enable natural language search capabilities over your paper collection. Once your papers are processed and indexed in Pinecone, you can:

- Search paper content using natural language queries
- Get semantically relevant excerpts from papers
- View full paper content with citation context

## Legacy Scripts

Previous versions of this pipeline used marker-pdf and required BibTeX files. These legacy scripts are preserved in the `archive/` directory for reference but are no longer recommended. See `archive/README.md` for details.

## Migration from Previous Versions

If you're migrating from the previous marker/BibTeX-based pipeline:

1. **Update dependencies**: Run `uv sync` (or `pip install -r requirements.txt`) to install Docling, DSPy, and updated dependencies
2. **Use new scripts**: Replace old scripts with `convert_pdfs_to_docling.py` and `upload_docling_to_pinecone.py`
3. **No BibTeX needed**: Metadata is now extracted directly from Zotero filenames
4. **API keys required**: Both Pinecone API key and Google API key (for Gemini) are now required
5. **Enhanced citation keys**: New format like `dezambottiStateScienceRecommendations2024`
6. **Intelligent filtering**: DSPy-powered quality assessment replaces statistical metrics

The new pipeline provides:
- **Intelligent content curation** - DSPy framework with Gemini for quality assessment
- **Simplified setup** - no BibTeX export needed
- **Better citation keys** with author and title information
- **Hierarchical chunking** that respects document organization
- **GPU acceleration** for faster processing
- **Optimized content quality** - removes noise while preserving research substance

## License

MIT License - See LICENSE file for details 
