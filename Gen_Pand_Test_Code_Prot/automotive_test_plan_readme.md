# Automotive Test Plan Generator

## Purpose

Automatically generates detailed test plans for automotive AVAS (Acoustic Vehicle Alerting System) requirements using AI-powered analysis. Converts requirement specifications into structured, color-coded Excel test plans.

## AI Model Options

**Primary: OpenRouter API (DeepSeek R1-Distill 70B)** - Cloud-based, recommended
- Requires API key
- Better accuracy and understanding
- No local installation needed

**Fallback: Ollama (DeepSeek R1 32B)** - Local option if API unavailable

## Core Functions & Performance

### 1. `parse_requirement(req_description)`
**Role**: Extracts structured components from requirement text

**How It Works**:
- Searches for IF/THEN (English) or SI/ALORS (French) patterns using regex
- Separates requirement into: condition1, condition2, result
- Falls back to generic sentence splitting if no pattern found

**Input Example**:
```
"IF sound is required (CMD=1) AND speed ≤ 20 km/h THEN emit AVAS sound"
```

**Output**:
```python
{
  "condition1": "sound is required (CMD=1)",
  "condition2": "speed ≤ 20 km/h", 
  "result": "emit AVAS sound"
}
```

**Performance**: Processes requirements in ~0.01 seconds

---

### 2. `extract_parameters(req_parts)`
**Role**: Identifies technical parameters and values from parsed requirement

**How It Works**:
- Applies 20+ regex patterns to detect automotive-specific parameters
- Searches for: sound commands, speed values, operators, system states, vehicle types
- Extracts parameter names and their values/operators

**Input Example**:
```python
{
  "condition1": "CMD_AVER_SON_VEH_SIL = 1",
  "condition2": "VITESSE_VEHICULE_ROUES < Vitesse_max_AVAS"
}
```

**Output**:
```python
{
  "sound_command": "1",
  "speed_parameter": ("<", "Vitesse_max_AVAS"),
  "max_speed_param": "Vitesse_max_AVAS"
}
```

**Performance**: Processes in ~0.02 seconds per requirement

---

### 3. `generate_test_name(req_id, req_parts, params)`
**Role**: Creates meaningful test names based on requirement analysis

**How It Works**:
- Extracts domain from requirement ID (AVAS, NVP, etc.)
- Analyzes parameters to determine test category
- Combines domain + behavior type + context

**Input Example**:
```python
req_id = "WAVE5-AVAS-ST-FUNC-001"
params = {"sound_command": "1", "speed_parameter": ("<", "20")}
```

**Output**:
```
"Check AVAS Sound Emission based on Vehicle Speed"
```

**Performance**: Instant generation

---

### 4. `generate_detailed_test_steps(req_parts, params)`
**Role**: Generates complete test step sequences with CI/AC/RA types

**How It Works**:
1. **Initial Setup (CI steps)**: Power, system state, diagnostics
2. **Parameter Configuration (AC steps)**: Based on extracted parameters
3. **Test Execution (AC steps)**: Multi-point testing (below/at/above thresholds)
4. **Verification (RA steps)**: Result validation and consistency checks
5. **Cleanup (AC/RA steps)**: System reset and final verification

**Input Example**:
```python
params = {
  "sound_command": "1",
  "speed_parameter": ("<", "Vitesse_max_AVAS")
}
```

**Output Sample**:
```python
[
  {"step_number": 1, "type": "CI", "description": "Ensure power supply is ON"},
  {"step_number": 2, "type": "AC", "description": "Send CAN signal CMD_AVER_SON_VEH_SIL = 1"},
  {"step_number": 3, "type": "AC", "description": "Set VITESSE_VEHICULE_ROUES = 5 km/h (below threshold)"},
  {"step_number": 4, "type": "RA", "description": "Verify AVAS sound is emitted from speakers"},
  {"step_number": 5, "type": "RA", "description": "Measure sound intensity with decibel meter at 1m"},
  ...
]
```

**Performance**: 
- Generates 8-15 steps per requirement
- ~0.05 seconds execution time
- Rule-based, deterministic output

---

### 5. `generate_test_plan_with_ai(req_id, req_description, use_api_key)`
**Role**: Main orchestration function - combines all parsing with AI intelligence

**How It Works**:
1. Calls `parse_requirement()` to extract structure
2. Calls `extract_parameters()` to identify technical values
3. Calls `generate_test_name()` for naming
4. Builds detailed AI prompt with extracted context
5. Sends to DeepSeek R1 model for intelligent test generation
6. Falls back to `generate_detailed_test_steps()` if AI fails
7. Structures output as complete test plan JSON

**Input Example**:
```python
req_id = "WAVE5-AVAS-ST-FUNC-001"
req_description = "IF sound required AND speed < 20 km/h THEN emit AVAS sound"
use_api_key = True
```

**Output**:
```python
{
  "requirement_id": "WAVE5-AVAS-ST-FUNC-001",
  "test_name": "Check AVAS Sound Emission based on Vehicle Speed",
  "description": "Verifies AVAS sound emission when vehicle speed is below threshold",
  "variant": "var01",
  "steps": [...],  # 8-15 steps
  "covered_requirement": "WAVE5-AVAS-ST-FUNC-001"
}
```

**Performance**:
- **With AI**: 3-8 seconds per requirement (network dependent)
- **Without AI (fallback)**: 0.1 seconds per requirement
- **Batch (100 reqs)**: 5-13 minutes with AI, 10 seconds without AI

---

### 6. `create_excel_file(test_plan_json)`
**Role**: Converts JSON test plan to professional Excel format

**How It Works**:
- Creates DataFrame with proper column structure
- Applies xlsxwriter formatting engine
- Assigns colors: CI=Blue, AC=Yellow, RA=Green
- Sets column widths, borders, text wrapping
- Merges cells for test metadata

**Input Example**:
```python
{
  "test_name": "Check AVAS Sound",
  "steps": [{"step_number": 1, "type": "CI", "description": "..."}]
}
```

**Output**: Excel file (.xlsx) with:
- Header row (light yellow background)
- Color-coded step types
- 7 columns with optimized widths
- Professional borders and alignment

**Performance**: ~0.2 seconds per Excel file

---

### 7. `parse_input_file(uploaded_file)`
**Role**: Extracts requirements from CSV/Excel for batch processing

**How It Works**:
- Detects file type (.csv, .xlsx, .xls)
- Uses pandas to read data
- Maps column names (handles 10+ variations of "Requirement ID", "Description")
- Validates and filters empty rows
- Returns list of requirement dictionaries

**Input Example**: CSV with columns "ID", "Description"

**Output**:
```python
[
  {"id": "WAVE5-AVAS-001", "description": "IF..."},
  {"id": "WAVE5-AVAS-002", "description": "IF..."},
  ...
]
```




## Installation

```bash
pip install streamlit pandas openpyxl xlsxwriter openai
```


1. Enter OpenRouter API key (optional, for AI enhancement)
2. Choose single or batch mode
3. Input requirements or upload file
4. Generate and download Excel test plans





# Example

![result](Gen_Plan_Test/R_T_P_Without_Image.png)

![result](Gen_Plan_Test/result.png)

---


**Version**: 1.0  
**Primary AI**: DeepSeek R1-Distill (70B) via OpenRouter  
**Output**: Excel (.xlsx), JSON, ZIP archives
