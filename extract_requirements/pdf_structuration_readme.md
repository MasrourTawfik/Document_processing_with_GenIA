# PDF Structuration Tool

## Purpose

This tool automatically processes large PDF documents by splitting them into individual pages and extracting all content (text, tables, images) with intelligent structuring. It generates organized markdown and text files for each page.

## What It Does

- **Splits PDF**: Divides a multi-page PDF into separate single-page PDFs
- **Extracts Text**: Pulls all readable text from each page
- **Extracts Tables**: Identifies and extracts tabular data with structure preserved
- **Extracts Images**: Saves embedded images as PNG files with automatic captions
- **Generates Files**: Creates both `.txt` (raw text) and `.md` (structured markdown) for each page

## Key Features

### 1. Page Splitting
Each page of your PDF becomes a separate file in `OUTPUT_DIR/` (e.g., `document_page_1.pdf`)

### 2. Intelligent Table Detection
- Identifies tables automatically
- Extracts row/column structure
- Generates descriptions: "Table of 5 rows and 3 columns"
- Preserves data in both plain text and markdown format

### 3. Image Extraction with Captions
- Extracts images to `extracted_images/`
- Attempts to find text below image as caption
- Uses French NLP model (spaCy) for caption extraction
- Saves as `page{N}_img{N}.png`

### 4. Dual Output Format

**Text Files** (`extracted_text/`):
- Raw text content
- Tables in pipe-separated format
- Simple and portable

**Markdown Files** (`extracted_readme/`):
- Structured with headers
- Tables in markdown format
- Embedded image references
- Better for documentation

## Installation

```bash
pip install PyPDF2 pdfplumber spacy matplotlib pillow
python -m spacy download fr_core_news_sm
```

## Usage

```python
# Process your PDF
process_pdf("/path/to/your/document.pdf")

# Clean up generated files when done
clean_generated_files(base_name="document", total_pages=100)
```

## Output Structure

```
OUTPUT_DIR/
├── document_page_1.pdf
├── document_page_2.pdf
└── ...

extracted_text/
├── document_page_1.txt
├── document_page_2.txt
└── ...

extracted_readme/
├── document_page_1.md
├── document_page_2.md
└── ...

extracted_images/
├── page1_img1.png
├── page1_img2.png
└── ...
```

## Example Output

### Markdown Structure
```markdown
# Page 1

## Texte principal
Lorem ipsum dolor sit amet...

## Tableau 1
- Description : Tableau de 3 lignes et 2 colonnes

| Header 1 | Header 2 |
| --- | --- |
| Data 1 | Data 2 |

## Image 1
![Image extraite](extracted_images/page1_img1.png)
- Légende automatique : Figure 1 - System Architecture
```

## Technical Details

### Libraries Used
- **PyPDF2**: PDF splitting and manipulation
- **pdfplumber**: Text and table extraction
- **spaCy**: Natural language processing for captions (French model)
- **Pillow**: Image processing and cropping

### Image Extraction Method
1. Detects image bounding box coordinates
2. Converts page to high-resolution image (200 DPI)
3. Crops specific image region
4. Looks for text 30 pixels below image as caption
5. Uses NLP to extract relevant caption text

### Table Detection
- Automatically identifies table structures
- Extracts headers and data rows
- Calculates dimensions
- Preserves cell alignment

## Cleanup Function

The `clean_generated_files()` function removes all generated content:
- Deletes individual page PDFs
- Removes text files
- Removes markdown files
- Deletes extracted images
- Removes empty directories

```python
clean_generated_files(base_name="document", total_pages=450)
```

## Limitations

- Optimized for French language (uses `fr_core_news_sm`)
- Image caption detection is heuristic-based (may not always find captions)
- Very complex table layouts might not extract perfectly
- Requires scanned PDFs to have OCR already applied

## Use Cases

- **Documentation Processing**: Extract content from technical manuals
- **Data Migration**: Convert PDF reports to structured markdown
- **Content Analysis**: Break down large documents for easier processing
- **Archive Management**: Extract and organize content from PDF archives

## Configuration

Key parameters you can adjust:

```python
OUTPUT_DIR = "/path/to/output"      # Where split PDFs go
IMAGES_DIR = "extracted_images"     # Image output folder
TEXT_DIR = "extracted_text"         # Text output folder
README_DIR = "extracted_readme"     # Markdown output folder
```

