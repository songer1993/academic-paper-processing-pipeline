# Changelog

## [2.0.0] - 2025-06-17

### Added
- **DSPy-based Chunk Quality Classifier**: Implemented intelligent chunk filtering using DSPy framework with Gemini
- **Quality Assessment**: Automatically filters out low-quality chunks including:
  - Manuscript headers and footers
  - Author affiliations and contact information
  - Funding statements and acknowledgments
  - Reference lists and citations
  - Keywords lists without context
  - Page numbers and formatting artifacts
- **Content Preservation**: Intelligently preserves substantive research content:
  - Methods and methodology sections
  - Results and findings
  - Discussion and analysis
  - Statistical data and mathematical expressions
  - Academic narrative content

### Changed
- **Complete architecture overhaul**: Replaced marker-pdf with Docling for superior PDF processing
- **New script structure**: 
  - `convert_pdfs_to_docling.py` replaces `convert_pdfs.py`
  - `upload_docling_to_pinecone.py` replaces `upsert_vectors.py`
  - Removed `rename_by_citekey.py` (now handled automatically)
- **Eliminated BibTeX dependency**: Metadata now extracted directly from Zotero filenames
- **Hierarchical chunking**: Upgraded from simple paragraph splitting to structure-aware chunking
- **Quality assessment**: Added DSPy-powered intelligent filtering
- **Chunk Processing**: Now includes quality confidence scores in metadata
- **Logging**: Added detailed logging for chunk assessment decisions and quality metrics
- **Dependencies**: Updated to tested versions - `dspy>=2.6.27`, `litellm>=1.72.6`, `pinecone>=7.1.0`, `docling>=2.7.1`, `torch>=2.7.1`

### Technical Details
- **Framework**: Uses DSPy's `ChainOfThought` module for classification
- **Model**: Integrates with Gemini 1.5 Flash via LiteLLM
- **Performance**: Typical retention rate of ~28% (filters ~72% of chunks as low-quality)
- **Efficiency**: Maintains consecutive chunk numbering for improved semantic continuity

### Configuration
- **New Environment Variable**: `GOOGLE_API_KEY` required for Gemini integration
- **Existing**: `PINECONE_API_KEY` still required for vector storage

### Benefits
- **Improved Search Quality**: Removes noise that would degrade semantic search results
- **Better Embeddings**: Cleaner content leads to more accurate vector representations
- **Reduced Storage**: Only stores meaningful content, reducing Pinecone usage
- **Enhanced Retrieval**: Higher precision in RAG applications due to quality filtering

## [1.1.0] - 2025-04-09
### Added
- Resume capability to Pinecone upload script by checking for existing chunks
- Fuzzy matching for BibTeX entries to improve robustness against filename variations
- Optional step for configuring Cursor MCP servers

### Changed
- Enhanced upload script with better error handling
- Updated README with Zotero/Better BibTeX preparation steps
- Refined MCP server configuration examples

## [1.0.0] - 2025-04-01
### Initial Release
- **marker-pdf based processing**: Initial implementation using marker-pdf for PDF conversion
- **BibTeX integration**: Required BibTeX files for metadata extraction
- **Basic chunking**: Simple paragraph-based chunking without quality assessment
- **Pinecone upload**: Direct upload to Pinecone without filtering
- **Scripts included**:
  - `convert_pdfs.py` - PDF to markdown conversion using marker-pdf
  - `rename_by_citekey.py` - File renaming based on citation keys
  - `upsert_vectors.py` - Vector upload to Pinecone
- **Manual setup**: Required input/output directory structure
- **No quality filtering**: All content preserved regardless of quality