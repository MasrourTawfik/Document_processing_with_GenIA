# Enhanced Automotive Test Plan Generator

## Purpose

AI-powered test plan generator with visual context analysis, specification document integration, and semantic search. Processes requirements with image analysis and references technical specifications to create comprehensive AVAS test plans.

* In this file we will create a streamlit app to generate test plans for automotive requirements with image processing capabilities that will help to extract more information from images related to the requirements and enhance the test plan generation process 
* The app will allow users to upload images such as diagrams, screenshots, or visual specifications that accompany the requirements. The AI model will analyze these images to extract relevant information that can be used to generate more comprehensive and accurate test plans.
* The app will also allow users to upload a specification document to provide additional context for the requirements.
* The app will use the OpenRouter API to access the DeepSeek R1 model for generating test plans, and it will also support local models like Ollama if available.
* The app will provide options to download the generated test plans in both Excel and JSON formats, with the Excel format matching a specific design layout.


## AI Models

**Primary: DeepSeek R1-Distill (70B)** via OpenRouter - For test generation
**Vision: GPT-4o** via OpenRouter - For image analysis  
**Fallback: Ollama (Llama 3.2)** - Local option if API unavailable
**Embeddings: all-MiniLM-L6-v2** - For specification search

---

## Core Functions & Performance

### 1. `parse_requirement(requirement_text)`
**Role**: Extracts IF/THEN or SI/ALORS structure from requirements

**How It Works**:
- Applies regex patterns for English (IF-THEN) and French (SI-ALORS)
- Separates into: condition, action, structure type
- Falls back to keyword detection for unstructured text

**Input Example**:
```
"IF vehicle speed < 20 km/h THEN AVAS shall emit warning sound"
```

**Output**:
```python
{
  'condition': 'vehicle speed < 20 km/h',
  'action': 'AVAS shall emit warning sound',
  'structure': 'if_then'
}
```

**Performance**: ~0.01 seconds

---

### 2. `extract_parameters(requirement_text)`
**Role**: Identifies technical parameters (speed, sound, states)

**How It Works**:
- Regex patterns for speed values (km/h)
- Keyword matching for sound/audio/acoustic systems
- Vehicle state detection (parked, moving, reverse, etc.)
- AVAS-specific parameter extraction

**Input Example**:
```
"AVAS sound at 15 km/h when vehicle moving forward"
```

**Output**:
```python
['Speed: 15 km/h', 'Sound/Audio system', 'AVAS system', 'Vehicle state: moving']
```

**Performance**: ~0.015 seconds

---

### 3. `process_image_with_ai(image_file, requirement_description)`
**Role**: Analyzes uploaded images (diagrams, screenshots) to extract visual specifications

**How It Works**:
1. Converts image to base64 encoding
2. Sends to GPT-4o vision model with requirement context
3. Extracts: UI elements, diagrams, system states, thresholds, technical details
4. Returns structured description as additional context

**Input Example**: Screenshot of AVAS control panel + requirement text

**Output**:
```
"Image shows AVAS control interface with speed threshold indicator at 20 km/h.
Display includes sound level meter (60-70 dB range) and vehicle state selector
with options: Parked, Forward, Reverse. Green indicator shows system active state."
```

**Performance**: 
- Image processing: 2-4 seconds
- Vision API call: 3-8 seconds
- **Total**: 5-12 seconds per image

---

### 4. `load_embedding_model()` & `create_embeddings(texts, model)`
**Role**: Creates semantic embeddings for specification search

**How It Works**:
1. Loads Sentence-BERT model (all-MiniLM-L6-v2)
2. Converts text chunks to 384-dimensional vectors
3. Stores embeddings for similarity search

**Input Example**: 1000 specification chunks

**Output**: NumPy array (1000, 384) of embeddings

**Performance**:
- Model load: 2-3 seconds (one-time)
- Embedding creation: ~0.5 seconds per 100 chunks
- 1000 chunks: ~5 seconds

---

### 5. `chunk_text(text, chunk_size=800, overlap=200)`
**Role**: Splits specification documents into searchable chunks

**How It Works**:
1. Divides text into 800-character chunks
2. Maintains 200-character overlap between chunks
3. Breaks at sentence boundaries (periods/newlines)
4. Filters chunks shorter than 50 characters

**Input Example**: 50-page specification (100,000 characters)

**Output**: ~150 overlapping chunks

**Performance**: ~0.3 seconds for 100,000 characters

---

### 6. `search_specification(query, chunks, embeddings, model, top_k=5)`
**Role**: Semantic search through specification documents

**How It Works**:
1. Converts query to embedding vector
2. Calculates cosine similarity with all chunk embeddings
3. Returns top-k most similar chunks with relevance scores
4. Filters results below 0.3 similarity threshold

**Input Example**:
```python
query = "AVAS sound emission at low speed"
```

**Output**:
```python
[
  ("Section 4.2: AVAS shall emit sound when speed < 20 km/h...", 0.87),
  ("The acoustic warning system activates below threshold...", 0.76),
  ("Sound level requirements: 56-75 dB at 2m distance...", 0.68)
]
```

**Performance**: 
- Single query: ~0.02 seconds
- 100 queries: ~2 seconds

---

### 7. `process_specification_file(uploaded_file)`
**Role**: Full specification document processing pipeline

**How It Works**:
1. Extracts text from DOCX or TXT files
2. Chunks text with `chunk_text()`
3. Loads embedding model
4. Creates embeddings for all chunks
5. Stores in session state for search

**Input Example**: 50-page AVAS specification DOCX

**Output**: 
- 150 chunks created
- 150 embeddings stored
- Search interface enabled

**Performance**:
- 10-page doc: ~5 seconds
- 50-page doc: ~15 seconds
- 100-page doc: ~30 seconds

---

### 8. `generate_test_plan_with_image_context(req_id, req_desc, image_context)`
**Role**: Main test generation with visual and specification context

**How It Works**:
1. Parses requirement structure
2. Extracts parameters
3. Searches specification (if loaded) for relevant sections
4. Builds enhanced AI prompt with:
   - Requirement structure
   - Extracted parameters
   - Image analysis results
   - Specification references
5. Calls DeepSeek R1 for intelligent test generation
6. Attaches spec references and image context to output

**Input Example**:
```python
req_id = "REQ-AVAS-001"
req_desc = "IF speed < 20 km/h THEN emit sound"
image_context = "Diagram shows 20 km/h threshold indicator"
```

**Output**:
```python
{
  "test_name": "REQ-AVAS-001 var0",
  "variant": "var0",
  "requirement_id": "REQ-AVAS-001",
  "description": "Verify AVAS sound emission below speed threshold",
  "steps": [...],  # 10-15 steps
  "spec_references": [{
    "parameter": "Speed: 20 km/h",
    "spec_reference": "Section 4.2: Speed threshold...",
    "relevance_score": 0.87
  }],
  "image_context": "Diagram shows 20 km/h threshold indicator",
  "covered_requirement": "REQ-AVAS-001"
}
```

**Performance**:
- With AI + image + spec: 8-15 seconds
- With AI + spec (no image): 4-8 seconds
- Without AI (fallback): 0.2 seconds

---




### 9. `create_enhanced_excel_file(test_plan)`
**Role**: Professional Excel formatting with specification references

**How It Works**:
1. Creates main test plan sheet with color-coded steps
2. Creates specification references sheet (if available)
3. Applies styling: colors, borders, fonts, column widths
4. Adds title row and merged cells

**Input Example**: Test plan JSON with 12 steps + 3 spec references

**Output**: Excel file with:
- Main sheet: Color-coded test steps (CI=Blue, AC=Yellow, RA=Green)
- References sheet: Specification citations with relevance scores

**Performance**: ~0.3 seconds per file

---

### 10. `generate_batch_test_plans(df, max_requirements, include_images)`
**Role**: Batch processing for multiple requirements

**How It Works**:
1. Iterates through DataFrame rows
2. Calls test generation for each requirement
3. Tracks progress with progress bar
4. Collects failures for reporting
5. 1-second delay between requests (rate limiting)

**Input Example**: CSV with 50 requirements

**Output**: 
- 50 test plans (if all successful)
- Failed requirement list
- Summary metrics

**Performance**:
- 10 requirements: ~1-2 minutes (with AI)
- 50 requirements: ~5-10 minutes (with AI)
- 100 requirements: ~10-20 minutes (with AI)

**Without AI (fallback)**: 50 requirements in ~10 seconds

---

## Feature Comparison

| Feature | This Tool | Basic Tool |
|---------|-----------|------------|
| Image Analysis | ✅ GPT-4o Vision | ❌ |
| Spec Search | ✅ Semantic Search | ❌ |
| Spec References | ✅ Auto-linked | ❌ |
| Visual Context | ✅ Integrated | ❌ |
| Embeddings | ✅ Sentence-BERT | ❌ |
| Excel Sheets | 2+ (Test + References) | 1 (Test only) |

---

## Installation

```bash
pip install streamlit pandas openpyxl python-docx pillow
pip install openai sentence-transformers scikit-learn numpy
```

## Usage

```bash
streamlit run app.py
```

**Workflow**:
1. Enter OpenRouter API key
2. Upload specification document (optional)
3. Upload requirement + image (optional)
4. Generate enhanced test plan
5. Download Excel with spec references

---

## Performance Summary

### Single Requirement
- Parse: 0.01s
- Parameters: 0.015s
- Spec search: 0.02s
- Image analysis: 5-12s
- AI generation: 4-8s
- Excel creation: 0.3s
- **Total (full)**: 9-20 seconds
- **Total (no image)**: 4-8 seconds

### Batch Processing
- 10 requirements: 1-2 minutes
- 50 requirements: 5-10 minutes
- 100 requirements: 10-20 minutes

### Specification Processing
- 10 pages: 5 seconds
- 50 pages: 15 seconds
- 100 pages: 30 seconds

---

**Version**: 2.0  
**AI Models**: DeepSeek R1 (70B), GPT-4o Vision, all-MiniLM-L6-v2  
**Output**: Excel with spec references, JSON, ZIP archives


# Example 


![input](Gen_Plan_Test/analyse_image.png)
![analyse](Gen_Plan_Test/analyse_Image.png)
![result](Gen_Plan_Test/reslt_P_T_With_Image.png)