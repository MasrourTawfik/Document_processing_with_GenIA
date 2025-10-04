# Enhanced Automotive Requirements Extractor

## Overview

The **Enhanced Automotive Requirements Extractor** is a sophisticated Python-based tool built with Streamlit that automatically extracts, analyzes, and exports functional requirements from automotive specification documents (Word format). This tool is specifically designed for automotive industry professionals who need to process large specification documents and extract structured requirement data with high accuracy.

## 🎯 Purpose

In the automotive industry, specification documents often contain hundreds or thousands of functional requirements scattered across multiple pages and sections. Manually extracting these requirements is:
- **Time-consuming**: Hours or days of manual work
- **Error-prone**: Easy to miss or misidentify requirements
- **Inconsistent**: Different team members may extract differently

This tool solves these problems by:
- Automatically identifying requirement IDs using configurable patterns
- Accurately mapping requirements to their source pages
- Organizing requirements by document sections
- Providing confidence scoring for extraction quality
- Enabling quick export to CSV/Excel for further processing

##  Key Features

### 1. **Accurate Page Mapping**
- Uses advanced algorithms to track page numbers in Word documents
- Detects explicit page breaks and section breaks
- Estimates pages based on content length when explicit breaks are absent
- Maps each requirement to its exact page number

### 2. **Section-Aware Extraction**
- Automatically identifies section headers (e.g., "Functional Requirements", "Non-Functional Requirements")
- Groups requirements by their containing sections
- Supports filtering by specific sections
- Recognizes multiple section naming conventions (English/French)

### 3. **Flexible Pattern Matching**
- Six pre-configured requirement ID patterns:
  - **WAVE5**: `WAVE5-AVAS-ST-FUNC-001`
  - **Generic Automotive**: `REQ-12345`
  - **Hierarchical**: `SYS-001-002-003`
  - **Standard Format**: `FUNC-REQ-001`
  - **Extended WAVE**: `WAVE5-ADVANCED-SYS-FUNC-001`
  - **Custom Pattern**: User-configurable patterns
- Multi-pattern selection for mixed document formats
- Case-insensitive matching

### 4. **Intelligent Table Processing**
- Extracts from structured requirement tables
- Handles multi-column layouts
- Identifies requirement IDs in any column
- Cleans and normalizes description text

### 5. **Quality Assurance**
- Confidence scoring for each extracted requirement
- Duplicate detection and removal
- Minimum confidence threshold filtering
- Validation of requirement structure

### 6. **Rich Analytics**
- Visual distribution of requirements by section
- Page-by-page requirement density
- Pattern usage statistics
- Confidence score distribution
- Summary metrics dashboard

### 7. **Flexible Export Options**
- **CSV Export**: Simple tabular format
- **Excel Export**: Multi-sheet workbook with sections
- Maintains all metadata (page, section, confidence)
- Ready for import into requirement management tools

## 🏗️ Architecture

### Core Components

#### 1. **EnhancedPageTracker**
```python
class EnhancedPageTracker:
```
- Tracks page numbers throughout the document
- Uses multiple strategies:
  - Explicit page break detection
  - Section break detection
  - Character-based estimation (2500 chars/page)
  - Line-based estimation (45 lines/page)
- Maps document elements to page numbers

#### 2. **SectionAwareExtractor**
```python
class SectionAwareExtractor:
```
- Main extraction engine
- Combines XML and document object processing
- Features:
  - Section header detection with regex patterns
  - Requirement ID extraction with multiple patterns
  - Table structure analysis
  - Description cleaning and normalization
  - Confidence calculation
  - Deduplication logic

#### 3. **Requirement Dataclass**
```python
@dataclass
class Requirement:
```
Stores extracted requirement information:
- `id`: Requirement identifier
- `description`: Requirement text
- `page`: Source page number
- `section`: Containing section name
- `table_index`: Table number in document
- `row_index`: Row number in table
- `confidence`: Extraction quality score (0-1)

## 🔧 Technical Details

### Document Processing Strategy

The tool uses a **dual-processing approach**:

1. **XML-Level Processing**
   - Accesses raw document XML structure
   - Precise page break detection
   - Direct element-to-page mapping
   - Handles complex table structures

2. **Document Object Processing**
   - Uses python-docx for high-level access
   - Paragraph and table iteration
   - Fallback for missed content
   - Cross-validation of results

### Pattern Matching

Requirement IDs are extracted using regex patterns:

```python
requirement_id_patterns = {
    'WAVE5': r'WAVE5[-_][A-Z]+[-_][A-Z]+[-_][A-Z]+[-_]\d+',
    'Generic Automotive': r'[A-Z]{2,8}[-_]\d{3,6}',
    # ... more patterns
}
```

### Confidence Scoring Algorithm

```python
def _calculate_confidence(req_id, description):
    confidence = 1.0
    
    # Penalties
    if len(description) < 20: confidence *= 0.7
    if is_header_text(description): confidence *= 0.8
    
    # Bonuses
    if has_structured_id(req_id): confidence *= 1.1
    
    return min(confidence, 1.0)
```

### Deduplication

Requirements are deduplicated based on:
- Identical requirement IDs
- Similar description content (first 100 characters)
- Prevents double-counting from XML and docx processing

## 📊 User Interface

### Sidebar Controls

**File Upload**
- Accepts `.docx` files only
- Supports large documents (100+ pages)

**Page Range Settings**
- Start Page: Define extraction start point
- End Page: Define extraction end point
- Useful for focusing on specific document sections

**Pattern Selection**
- Multi-select dropdown
- Choose patterns matching your document
- Preview pattern definitions

**Section Filtering**
- Filter by: All, Functional, Non-Functional, Technical, Performance, Safety
- Keyword-based matching
- Supports English and French terminology

**Advanced Settings**
- Minimum Confidence Threshold (0.0-1.0)
- Show/Hide Potential Duplicates
- Fine-tune extraction quality

### Main View Tabs

#### Tab 1: Results 📊
- Searchable, sortable table of all requirements
- Section filter for focused viewing
- Color-coded confidence indicators
- Full description preview

#### Tab 2: Analytics 📈
- **Pie Chart**: Requirements by section
- **Bar Chart**: Requirements by page
- **Metrics Dashboard**: Total count, sections, page range, average confidence
- Visual insights into document structure

#### Tab 3: Details 🔍
- Pattern usage statistics
- Confidence distribution histogram
- Quality metrics
- Extraction diagnostics

#### Tab 4: Export 💾
- CSV download button
- Excel download with multi-sheet format
- Export summary with statistics
- Timestamped filenames

## 🚀 Usage Guide

### Step 1: Installation

```bash
# Install required packages
pip install streamlit pandas python-docx openpyxl plotly

# Run the application
streamlit run app.py
```

### Step 2: Document Preparation

Ensure your Word document has:
- Requirements in table format
- Clear section headers
- Consistent ID formatting
- Two-column minimum (ID + Description)

### Step 3: Configuration

1. Upload your `.docx` file
2. Set page range (if needed)
3. Select matching ID patterns
4. Choose section filter (optional)
5. Adjust confidence threshold

### Step 4: Extraction

Click "Extract" or let auto-extraction run:
- Processing time: ~2-5 seconds per 100 requirements
- Progress indicator shows status
- Error messages provide troubleshooting info

### Step 5: Review & Validate

- Check extraction count
- Review sample requirements
- Verify page mappings
- Adjust settings if needed

### Step 6: Export

- Download CSV for simple use cases
- Download Excel for organized multi-section output
- Files named with document name and page range

## 📋 Example Use Cases

### Use Case 1: Full Document Extraction
```
Document: Vehicle_Specs_v2.5.docx
Pages: 1-150
Patterns: WAVE5, Generic Automotive
Section: All
Result: 450 requirements across 8 sections
```

### Use Case 2: Focused Section Extraction
```
Document: System_Requirements.docx
Pages: 20-45
Patterns: Standard Format
Section: Functional
Result: 127 functional requirements from pages 20-45
```

### Use Case 3: Quality-Filtered Extraction
```
Document: Draft_Specifications.docx
Pages: 1-999
Patterns: All
Min Confidence: 0.8
Result: 234 high-confidence requirements
```

## 🛠️ Troubleshooting

### Issue: No Requirements Found

**Solutions:**
- Verify pattern selection matches your document
- Check if requirements are in table format
- Reduce confidence threshold
- Expand page range

### Issue: Incorrect Page Numbers

**Solutions:**
- Check for unusual document formatting
- Verify page breaks are standard
- Use broader page range
- Contact support for custom documents

### Issue: Missing Descriptions

**Solutions:**
- Ensure tables have at least 2 columns
- Check for merged cells
- Verify text isn't in images
- Review description cleaning rules

### Issue: Duplicate Requirements

**Solutions:**
- Enable duplicate detection (default: ON)
- Check if requirements appear in multiple tables
- Adjust deduplication sensitivity
- Manual review in Details tab

## 🔒 Limitations

1. **File Format**: Only `.docx` files supported (not `.doc`)
2. **Table Structure**: Requires tabular format for extraction
3. **Images**: Cannot extract from images or scanned PDFs
4. **Complex Layouts**: May struggle with heavily formatted documents
5. **Language**: Optimized for English/French, may need adjustments for other languages

## 🔮 Future Enhancements

- [ ] Support for `.doc` legacy format
- [ ] PDF extraction capabilities
- [ ] Custom pattern creation UI
- [ ] Requirement traceability matrix generation
- [ ] Integration with DOORS/JIRA
- [ ] Multi-language support expansion
- [ ] ML-based requirement classification
- [ ] Collaborative review features

## 📄 License

This tool is provided as-is for automotive industry use. Modify and distribute according to your organization's policies.

## 👥 Support

For issues, questions, or feature requests:
- Review the troubleshooting guide
- Check extraction analytics for clues
- Document your issue with sample files
- Contact your development team

## 📚 Technical Requirements

**Python Version**: 3.8+

**Dependencies**:
- `streamlit>=1.20.0`
- `pandas>=1.5.0`
- `python-docx>=0.8.11`
- `openpyxl>=3.0.0`
- `plotly>=5.10.0`

**System Requirements**:
- RAM: 4GB minimum (8GB recommended)
- Storage: 100MB for installation
- Browser: Modern web browser (Chrome, Firefox, Edge)

---

**Version**: 1.0  
**Last Updated**: 01/05/2025  
**Category**: Automotive Requirements Engineering Tools