#for more information you need to read the readme file and for the usage also you need to run this file with streamlit in lighnting.ia to be more faster
# you can find the readme file in the same folder
# If you want to run it locally you need to install streamlit and other libraries like pandas openpyxl xlsxwriter 
#to use the openrouter api you need to create an account on openrouter.ai and get your api key it's free and frendly to use 
#if you use the openrouter api and lighning ia you will get better results and also with less time because the model is hosted on their servers 
# if do you have any question you can contact me on my email




import streamlit as st
import pandas as pd
import subprocess
import json
import io
import re
import time
from io import BytesIO
import zipfile
import os
from openai import OpenAI



#OPENROUTER_API_KEY = "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"



# Set page configuration
st.set_page_config(page_title="Automotive Test Plan Generator", layout="wide")

# Load API key from environment or secrets
def get_api_key():
    # Try to get API key from Streamlit secrets if deployed
    if hasattr(st, 'secrets') and 'OPENROUTER_API_KEY' in st.secrets:
        return st.secrets['OPENROUTER_API_KEY']
    # Otherwise try to get it from environment variable
    return os.environ.get('OPENROUTER_API_KEY', '')

# Initialize OpenRouter client with configurable API key
def get_openrouter_client():
    api_key = get_api_key()
    if not api_key:
        api_key = st.session_state.get('api_key', '')
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    return client

# Function to call OpenRouter API for R1 Distill Llama
def call_r1_distill_llama(prompt, temperature=0.7, max_tokens=4000):
    try:
        client = get_openrouter_client()
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "automotive-test-plan-generator.app",
                "X-Title": "Automotive Test Plan Generator",
            },
            model="deepseek/deepseek-r1-distill-llama-70b:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert automotive test engineer specializing in detailed AVAS (Acoustic Vehicle Alerting System) test plan generation. Your task is to create comprehensive test plans based on requirements."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return completion.choices[0].message.content, None
    except Exception as e:
        return None, f"Error calling OpenRouter API: {str(e)}"

# Function to call Ollama API with better error handling (fallback option)
def run_ollama(prompt, model="deepseek-r1:32b", max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout
            )
            if result.returncode == 0:
                return result.stdout, None
            else:
                return None, f"Error (code {result.returncode}): {result.stderr}"
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                st.warning(f"Request timed out, retrying ({attempt+1}/{max_retries})...")
                time.sleep(2)  # Wait before retry
            else:
                return None, "Request timed out. The model is taking too long to respond."
        except Exception as e:
            return None, f"Error calling Ollama: {str(e)}"

# Improved function to parse requirements with multi-language support
def parse_requirement(req_description):
    """Parse the requirement to extract key components (IF/SI, AND/ET, THEN/ALORS sections)"""
    requirement_parts = {}
    
    # Try to identify IF/THEN structure (English)
    if_match = re.search(r'IF\s*(.*?)(?:AND\s*(.*?))?THEN\s*(.*)', req_description, re.DOTALL | re.IGNORECASE)
    
    if if_match:
        requirement_parts["condition1"] = if_match.group(1).strip() if if_match.group(1) else ""
        requirement_parts["condition2"] = if_match.group(2).strip() if if_match.group(2) else ""
        requirement_parts["result"] = if_match.group(3).strip() if if_match.group(3) else ""
    else:
        # Try to identify SI/ALORS structure (French)
        si_match = re.search(r'SI\s*(.*?)(?:ET\s*(.*?))?ALORS\s*(.*)', req_description, re.DOTALL | re.IGNORECASE)
        if si_match:
            requirement_parts["condition1"] = si_match.group(1).strip() if si_match.group(1) else ""
            requirement_parts["condition2"] = si_match.group(2).strip() if si_match.group(2) else ""
            requirement_parts["result"] = si_match.group(3).strip() if si_match.group(3) else ""
        else:
            # Advanced parsing with more flexible pattern matching
            patterns = [
                # Try to match condition-result pattern without explicit keywords
                r'(.*?)\s*[.,;]\s*(.*?)\s*(?:results in|leads to|causes|produces)\s*(.*)',
                # Simple sentence splitting as fallback
                r'(.*?)[.]\s*(.*)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, req_description, re.DOTALL | re.IGNORECASE)
                if match and match.groups():
                    if len(match.groups()) >= 3:
                        requirement_parts["condition1"] = match.group(1).strip()
                        requirement_parts["condition2"] = match.group(2).strip()
                        requirement_parts["result"] = match.group(3).strip()
                        break
                    elif len(match.groups()) >= 2:
                        requirement_parts["condition1"] = match.group(1).strip()
                        requirement_parts["result"] = match.group(2).strip()
                        break
            
            # If still no match, store the whole description
            if not requirement_parts:
                requirement_parts["description"] = req_description.strip()
    
    return requirement_parts

# Enhanced parameter extraction with more patterns
def extract_parameters(req_parts):
    """Extract key parameters from parsed requirement parts with improved pattern matching"""
    params = {}
    
    # Extended parameters to look for
    param_patterns = {
        "sound_command": [
            r'CMD_AVER_SON_VEH_SIL\s*=\s*([\w]+)',
            r'CMD_AVER_SON\s*=\s*([\w]+)',
            r'son\s*.*\s*(?:requis|required)\s*.*\s*(activ[ée]|active)'
        ],
        "speed_parameter": [
            r'VITESSE_VEHICULE_ROUES\s*([<>=]+)\s*([0-9]+\s*km\/h|\w+)',
            r'vitesse\s*.*\s*([<>=]+)\s*([0-9]+\s*km\/h|\w+)',
            r'speed\s*.*\s*([<>=]+)\s*([0-9]+\s*km\/h|\w+)',
            r'below\s*\(?=?<?[<>=]*\)?\s*([a-zA-Z_]+)',
            r'dépasse\s*\(?[<>=]*\)?\s*([a-zA-Z_]+)'
        ],
        "sound_output": [
            r'(GPS_PILOT_AVERT_SON_VEH_SIL)',
            r'(AVAS)\s*(?:emitt?|émis|emis|emit)',
            r'son\s*(?:is|est)\s*(?:emitt?|émis|emis|emit)'
        ],
        "system_state": [
            r'(ETAT_MA|ETAT_MAR)',
            r'(Phase_vie)',
            r'(state|état)\s*=\s*(\w+)'
        ],
        "vehicle_type": [
            r'TYPE_CHAINE_TRACTION\s*=\s*(HY|ELEC)',
            r'(?:vehicle|véhicule)\s*type\s*=\s*(\w+)'
        ],
        "max_speed_param": [
            r'(Vitesse_max_AVAS)',
            r'(V_MAX_AVAS)',
            r'(MAX_SPEED_AVAS)'
        ]
    }
    
    # Combine all parts for searching
    full_text = " ".join([v for k, v in req_parts.items() if v])
    
    # Try each pattern in each category
    for param_name, patterns in param_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                if len(match.groups()) > 1:
                    params[param_name] = (match.group(1), match.group(2))
                else:
                    params[param_name] = match.group(1)
                break  # Found a match for this parameter, move to next
    
    return params

# Improved test name generation with better context understanding
def generate_test_name(req_id, req_parts, params):
    """Generate a meaningful test name based on requirement analysis"""
    # Default test name based on ID
    test_name = f"Test for {req_id}"
    
    # Extract domain from requirement ID
    domain_match = re.search(r'(?:WAVE5|ASCW4)[-_]([A-Z]+)', req_id)
    domain = domain_match.group(1) if domain_match else "AVAS"
    
    # Sound-related test
    if "sound_command" in params or "sound_output" in params:
        if "speed_parameter" in params:
            test_name = f"Check {domain} Sound Emission based on Vehicle Speed"
        else:
            test_name = f"Check {domain} Sound Emission"

    # Speed-related test
    elif "speed_parameter" in params and not ("sound_command" in params or "sound_output" in params):
        test_name = f"Check {domain} Speed-Related Behavior"
    
    # System state test
    elif "system_state" in params:
        test_name = f"Check {domain} System State Transition"
    
    # Vehicle type test
    elif "vehicle_type" in params:
        test_name = f"Check {domain} Behavior for Specific Vehicle Type"
    
    # Create a more specific name based on conditions if possible
    if "condition1" in req_parts and req_parts["condition1"]:
        condition = req_parts["condition1"].lower()
        if "sound" in condition or "son" in condition:
            test_name = f"Check {domain} Sound Emission"
        elif "speed" in condition or "vitesse" in condition:
            test_name = f"Check {domain} Speed-Related Behavior"
        elif "state" in condition or "état" in condition:
            test_name = f"Check {domain} System State Transition"
    
    # Add specific context if available
    full_text = " ".join([v for k, v in req_parts.items() if v])
    context_terms = {
        "warn": "Warning", "alert": "Alert", "avert": "Warning",
        "pedestrian": "Pedestrian", "piéton": "Pedestrian",
        "safety": "Safety", "sécurité": "Safety"
    }
    
    for term, context in context_terms.items():
        if term in full_text.lower():
            test_name = f"{test_name} - {context}"
            break
    
    return test_name

# Enhanced test steps generation with intelligent step creation
def generate_detailed_test_steps(req_parts, params):
    """Generate detailed test steps based on requirement analysis with improved intelligence"""
    steps = []
    step_num = 1
    
    # Standard initial conditions - expanded with more detailed setup
    steps.append({
        "step_number": step_num, 
        "type": "CI", 
        "description": "Ensure power supply is connected and ON for all test components"
    })
    step_num += 1
    
    steps.append({
        "step_number": step_num, 
        "type": "CI", 
        "description": "Verify system is in operational mode with CAN = ON (Var1) / Set Phase_vie = normal mode (Var2)"
    })
    step_num += 1
    
    steps.append({
        "step_number": step_num, 
        "type": "CI", 
        "description": "Connect diagnostic tool and confirm no active DTCs present in the system"
    })
    step_num += 1
    
    # Vehicle type specific condition if needed
    if "vehicle_type" in params:
        v_type = params["vehicle_type"]
        if isinstance(v_type, tuple):
            v_type = v_type[0]
        
        steps.append({
            "step_number": step_num,
            "type": "CI",
            "description": f"Configure vehicle type parameter TYPE_CHAINE_TRACTION = {v_type} in the system"
        })
        step_num += 1
    
    # System state specific condition if needed
    if "system_state" in params:
        state = params["system_state"]
        if isinstance(state, tuple):
            state = state[0]
        
        steps.append({
            "step_number": step_num,
            "type": "CI",
            "description": f"Verify system is in {state} state before proceeding"
        })
        step_num += 1
    
    # Add specific steps based on the parsed requirements
    if "sound_command" in params:
        command_value = params["sound_command"] if isinstance(params["sound_command"], str) else params["sound_command"][0]
        if command_value in ["activé", "active", "1"]:
            command_value = "1"  # Normalize to binary value
        
        # Add detailed steps for sound command
        steps.append({
            "step_number": step_num, 
            "type": "AC", 
            "description": f"Send CAN signal CMD_AVER_SON_VEH_SIL = {command_value} to the AVAS control unit"
        })
        step_num += 1
        
        steps.append({
            "step_number": step_num, 
            "type": "AC", 
            "description": f"Verify signal CMD_AVER_SON_VEH_SIL = {command_value} is correctly received by the AVAS control unit using diagnostic tool"
        })
        step_num += 1
    
    # Speed parameter handling with more detail
    if "speed_parameter" in params:
        speed_info = params["speed_parameter"]
        
        # Different handling based on operator type
        if isinstance(speed_info, tuple):
            operator, value = speed_info
            
            # Handle max speed parameter if available
            max_speed_param = params.get("max_speed_param", "Vitesse_max_AVAS")
            
            # Set appropriate test values based on operator
            if ">" in operator or "dépasse" in operator:
                below_threshold = 5  # A low value below typical threshold
                at_threshold = 10    # Threshold value
                above_threshold = 15  # Value above threshold
                
                # Three step speed test at different values
                steps.append({
                    "step_number": step_num,
                    "type": "AC",
                    "description": f"Set VITESSE_VEHICULE_ROUES = {below_threshold} km/h (below {max_speed_param})"
                })
                step_num += 1
                
                steps.append({
                    "step_number": step_num,
                    "type": "RA",
                    "description": f"Verify no sound is emitted as speed is below {max_speed_param}"
                })
                step_num += 1
                
                steps.append({
                    "step_number": step_num,
                    "type": "AC",
                    "description": f"Set VITESSE_VEHICULE_ROUES = {at_threshold} km/h (at {max_speed_param} threshold)"
                })
                step_num += 1
                
                steps.append({
                    "step_number": step_num,
                    "type": "RA",
                    "description": f"Verify sound behavior at threshold speed"
                })
                step_num += 1
                
                steps.append({
                    "step_number": step_num,
                    "type": "AC",
                    "description": f"Set VITESSE_VEHICULE_ROUES = {above_threshold} km/h (above {max_speed_param})"
                })
                step_num += 1
                
            elif "<" in operator or "below" in operator:
                above_threshold = 15  # A high value above typical threshold
                at_threshold = 10     # Threshold value
                below_threshold = 5   # Value below threshold
                
                # Three step speed test at different values
                steps.append({
                    "step_number": step_num,
                    "type": "AC",
                    "description": f"Set VITESSE_VEHICULE_ROUES = {above_threshold} km/h (above {max_speed_param})"
                })
                step_num += 1
                
                steps.append({
                    "step_number": step_num,
                    "type": "RA",
                    "description": f"Verify system behavior when speed is above {max_speed_param}"
                })
                step_num += 1
                
                steps.append({
                    "step_number": step_num,
                    "type": "AC",
                    "description": f"Set VITESSE_VEHICULE_ROUES = {at_threshold} km/h (at {max_speed_param} threshold)"
                })
                step_num += 1
                
                steps.append({
                    "step_number": step_num,
                    "type": "RA",
                    "description": f"Verify system behavior at threshold speed"
                })
                step_num += 1
                
                steps.append({
                    "step_number": step_num,
                    "type": "AC",
                    "description": f"Set VITESSE_VEHICULE_ROUES = {below_threshold} km/h (below {max_speed_param})"
                })
                step_num += 1
            
            else:
                # Equal operator or unspecified
                test_value = "10"  # A default value
                steps.append({
                    "step_number": step_num,
                    "type": "AC",
                    "description": f"Set VITESSE_VEHICULE_ROUES = {test_value} km/h"
                })
                step_num += 1
        else:
            # If we have a parameter name but not an operator, use default value and multi-step approach
            param_name = speed_info
            steps.append({
                "step_number": step_num,
                "type": "AC",
                "description": f"Set VITESSE_VEHICULE_ROUES = 0 km/h (initial value)"
            })
            step_num += 1
            
            steps.append({
                "step_number": step_num,
                "type": "AC",
                "description": f"Gradually increase VITESSE_VEHICULE_ROUES from 0 to 20 km/h"
            })
            step_num += 1
            
            steps.append({
                "step_number": step_num,
                "type": "RA",
                "description": f"Monitor and record when sound changes occur relative to {param_name}"
            })
            step_num += 1
    
    # Add result assertions with expanded verification steps
    if "sound_output" in params:
        sound_output = params["sound_output"]
        if isinstance(sound_output, tuple):
            sound_output = sound_output[0]
            
        steps.append({
            "step_number": step_num,
            "type": "RA",
            "description": f"Verify that {sound_output} is correctly emitted from the AVAS speakers"
        })
        step_num += 1
        
        steps.append({
            "step_number": step_num,
            "type": "RA",
            "description": "Measure sound intensity with decibel meter at 1m from vehicle to verify it meets required specifications"
        })
        step_num += 1
        
        steps.append({
            "step_number": step_num,
            "type": "RA",
            "description": "Verify that the sound characteristics match the expected profile for current vehicle state"
        })
        step_num += 1
    else:
        # Generate result step based on the "result" part of the requirement
        if "result" in req_parts and req_parts["result"]:
            # Clean up the result text
            result_text = req_parts["result"]
            result_text = result_text.replace("ALORS", "").replace("THEN", "").strip()
            
            # Simplify and format the result text
            simplified_result = re.sub(r'\(.*?\)', '', result_text)  # Remove parenthetical details
            simplified_result = re.sub(r'\s+', ' ', simplified_result)  # Normalize whitespace
            
            steps.append({
                "step_number": step_num,
                "type": "RA",
                "description": simplified_result[:150]  # Limit length but allow longer descriptions
            })
            step_num += 1
            
            # Add validation step to make sure requirement is properly met
            steps.append({
                "step_number": step_num,
                "type": "RA",
                "description": "Confirm system behavior is consistent across 3 consecutive test cycles"
            })
            step_num += 1
        else:
            # Generic result step if specific output not identified
            steps.append({
                "step_number": step_num,
                "type": "RA",
                "description": "Verify the expected system behavior according to the requirement"
            })
            step_num += 1
    
    # Add steps to restore initial conditions
    steps.append({
        "step_number": step_num,
        "type": "AC",
        "description": "Reset all modified parameters to their initial values"
    })
    step_num += 1
    
    steps.append({
        "step_number": step_num,
        "type": "RA",
        "description": "Verify system has returned to normal operation with no fault codes"
    })
    
    return steps

# Improved function to extract covered requirements
def extract_covered_requirements(req_id, req_description):
    """Extract referenced requirements from the requirement description with better pattern matching"""
    # Look for requirement IDs with various formats
    patterns = [
        r'([A-Z]+-[A-Z]+-[A-Z]+-[A-Z]+-\d+\(\d+\))',  # Standard format: WAVE5-VHL-IEV-NVP-124(2)
        r'([A-Z]+[A-Z0-9_]+-[A-Z]+[A-Z0-9_]+-\d+\(\d+\))',  # Alternate format: WAVE5-AVAS-0010(0)
        r'([A-Z]+[A-Z0-9_]+-[A-Z]+[A-Z0-9_]+-[A-Z0-9_]+-\d+)',  # Format without version: WAVE5-VHL-FCT-124
        r'([A-Z]+[A-Z0-9_]+-[A-Z]+[A-Z0-9_]+-[A-Z0-9_]+-\d+\(\d+\))'  # Format with underscore: WAVE5_VHL_FCT_124(0)
    ]
    
    # Extract all matches across all patterns
    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, req_description)
        all_matches.extend(matches)
    
    # Process the matches
    if all_matches:
        # Filter out duplicates and the original req_id
        unique_matches = [m for m in all_matches if m != req_id]
        if unique_matches:
            # Join multiple references with commas
            return ", ".join(unique_matches)
    
    # Extract from specific sections
    input_req_match = re.search(r'Input\s*requirement[^:]*:?\s*([A-Z]+-[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+-\d+)', 
                               req_description, re.IGNORECASE)
    if input_req_match:
        return input_req_match.group(1)
    
    # If no explicit reference is found, return original ID
    return req_id

# Function to create Excel file with proper formatting
def create_excel_file(test_plan_json):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Prepare data for Excel in the specific format from the example
    test_data = []
    
    # Add header row
    test_data.append(["Test Name", "variant", "Description", "Step N", "Type", "Description (Design Steps)", "Covered Requirement (Design Step)"])
    
    # Add test info row
    base_data = [
        test_plan_json["test_name"],
        test_plan_json["variant"],
        test_plan_json["description"]
    ]
    
    # Add test steps rows - repeat test info for each step
    for step in test_plan_json["steps"]:
        row_data = base_data.copy()
        row_data.extend([
            step["step_number"],
            step["type"],
            step["description"],
            # Only add covered requirement to the last step
            test_plan_json["covered_requirement"] if step == test_plan_json["steps"][-1] else ""
        ])
        test_data.append(row_data)
    
    # Convert to DataFrame
    test_df = pd.DataFrame(test_data[1:], columns=test_data[0])
    
    # Write to Excel - Format similar to the example image
    test_df.to_excel(writer, sheet_name='Test Plan', index=False, startrow=1)
    
    # Get the xlsxwriter workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Test Plan']
    
    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'bg_color': '#FFFFCC',  # Light yellow for header
        'border': 1
    })
    
    # Define cell formats with colors matching the example
    ci_format = workbook.add_format({
        'text_wrap': True,
        'valign': 'top',
        'bg_color': '#B7DEE8',  # Light blue for CI
        'border': 1
    })
    
    ac_format = workbook.add_format({
        'text_wrap': True,
        'valign': 'top',
        'bg_color': '#FFFF00',  # Yellow for AC
        'border': 1
    })
    
    ra_format = workbook.add_format({
        'text_wrap': True,
        'valign': 'top',
        'bg_color': '#92D050',  # Green for RA
        'border': 1
    })
    
    cell_format = workbook.add_format({
        'text_wrap': True,
        'valign': 'top',
        'border': 1
    })
    
    # Write headers with format
    for col_num, value in enumerate(test_data[0]):
        worksheet.write(0, col_num, value, header_format)
    
    # Format all data cells and apply type-specific formatting
    for row_idx, row in enumerate(test_data[1:]):
        for col_idx, value in enumerate(row):
            # Apply specific formatting to Type column (the 5th column, index 4)
            if col_idx == 4:  # Type column
                if value == 'CI':
                    worksheet.write(row_idx + 1, col_idx, value, ci_format)
                elif value == 'AC':
                    worksheet.write(row_idx + 1, col_idx, value, ac_format)
                elif value == 'RA':
                    worksheet.write(row_idx + 1, col_idx, value, ra_format)
                else:
                    worksheet.write(row_idx + 1, col_idx, value, cell_format)
            else:
                worksheet.write(row_idx + 1, col_idx, value, cell_format)
    
    # Set column widths to match example
    column_widths = [15, 10, 30, 5, 5, 40, 25]
    for i, width in enumerate(column_widths):
        worksheet.set_column(i, i, width)
    
    writer.close()
    return output.getvalue()

# Function to display test plan in the UI
def display_test_plan(test_plan_json):
    """Display the test plan in a formatted way in the Streamlit UI"""
    # Convert test plan JSON to DataFrames for display
    test_info_df, steps_df = json_to_dataframe(test_plan_json)
    
    # Display test info
    st.markdown("### Test Information")
    st.table(test_info_df)
    
    # Display steps with color coding
    st.markdown("### Test Steps")
    
    # Apply colors to step types
    def color_step_type(val):
        if val == 'CI':
            return 'background-color: #B7DEE8'  # Light blue
        elif val == 'AC':
            return 'background-color: #FFFF00'  # Yellow
        elif val == 'RA':
            return 'background-color: #92D050'  # Green
        return ''
    
    # Apply the styling
    styled_steps = steps_df.style.applymap(color_step_type, subset=['Type'])
    st.table(styled_steps)
    
    # Provide download buttons
    excel_data = create_excel_file(test_plan_json)
    
    st.download_button(
        label="Download Test Plan as Excel",
        data=excel_data,
        file_name=f"test_plan_{test_plan_json['requirement_id'].replace('(', '_').replace(')', '')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Also offer JSON download for integration purposes
    json_str = json.dumps(test_plan_json, indent=2)
    st.download_button(
        label="Download Test Plan as JSON",
        data=json_str,
        file_name=f"test_plan_{test_plan_json['requirement_id'].replace('(', '_').replace(')', '')}.json",
        mime="application/json"
    )

# Continue where previous code left off

# Function to convert test plan JSON to DataFrame for display
def json_to_dataframe(test_plan_json):
    # Basic test info for the header
    test_info = {
        "Test Name": [test_plan_json["test_name"]],
        "Variant": [test_plan_json["variant"]],
        "Description": [test_plan_json["description"]],
        "Requirement ID": [test_plan_json["requirement_id"]],
        "Covered Requirements": [test_plan_json["covered_requirement"]]
    }
    
    # Convert to DataFrame
    test_info_df = pd.DataFrame(test_info)
    
    # Steps info
    steps_data = []
    for step in test_plan_json["steps"]:
        steps_data.append({
            "Step N": step["step_number"],
            "Type": step["type"],
            "Description": step["description"]
        })
    
    # Convert to DataFrame
    steps_df = pd.DataFrame(steps_data)
    
    return test_info_df, steps_df

# Function to create a zip archive with all test plans
def create_zip_archive(all_test_plans):
    """Create a zip file containing all test plans as Excel files"""
    buffer = BytesIO()
    
    with zipfile.ZipFile(buffer, 'w') as zipf:
        for test_plan in all_test_plans:
            # Create Excel file
            excel_data = create_excel_file(test_plan)
            
            # Add to zip with filename based on requirement ID
            safe_filename = f"test_plan_{test_plan['requirement_id'].replace('(', '_').replace(')', '')}.xlsx"
            zipf.writestr(safe_filename, excel_data)
    
    buffer.seek(0)
    return buffer

# Enhanced function to generate more detailed test plan with AI assistance
def generate_test_plan_with_ai(req_id, req_description, use_api_key=True):
    """Generate test plan with AI assistance for deeper understanding and better results"""
    # Parse the requirement
    req_parts = parse_requirement(req_description)
    
    # Extract parameters
    params = extract_parameters(req_parts)
    
    # Prepare basic test plan structure
    test_name = generate_test_name(req_id, req_parts, params)
    
    # Build the AI prompt with detailed context
    prompt = f"""
    I need to create a detailed automotive test plan for the following AVAS (Acoustic Vehicle Alerting System) requirement:
    
    Requirement ID: {req_id}
    Requirement Description: {req_description}
    
    The requirement has been analyzed as follows:
    - Conditions: {req_parts.get('condition1', '')} {req_parts.get('condition2', '')}
    - Expected Result: {req_parts.get('result', '')}
    
    Parameters identified:
    {json.dumps(params, indent=2)}
    
    Based on this analysis, please help me:
    
    1. Create a refined test name that captures the essence of the requirement
    2. Provide a concise test description that summarizes the test objective
    3. Generate detailed test steps including:
       - Initial conditions setup (CI steps)
       - Actions to be performed (AC steps)
       - Results to be verified (RA steps)
    
    Important: Steps should focus on practical verification of AVAS sound emission based on vehicle speed, 
    system state, and control commands. Explain how to verify the specific behavior in the requirement.
    
    Structure your response as JSON in the following format:
    {{
      "test_name": "...",
      "description": "...",
      "variant": "var01",
      "steps": [
        {{
          "step_number": 1,
          "type": "CI/AC/RA",
          "description": "..."
        }},
        ...
      ]
    }}
    """
    
    # Attempt to use AI model to generate detailed test plan
    if use_api_key:
        response, error = call_r1_distill_llama(prompt)
    else:
        response, error = run_ollama(prompt)
    
    # Handle API errors
    if error:
        st.error(f"Error generating test plan with AI: {error}")
        # Fall back to rule-based approach
        test_steps = generate_detailed_test_steps(req_parts, params)
        
        test_plan = {
            "requirement_id": req_id,
            "test_name": test_name,
            "description": f"This test verifies the AVAS system behavior as specified in requirement {req_id}",
            "variant": "var01",
            "steps": test_steps,
            "covered_requirement": extract_covered_requirements(req_id, req_description)
        }
        return test_plan
    
    # Process AI response
    try:
        # Try to extract JSON from the response - look for JSON format
        json_match = re.search(r'```json(.*?)```|{.*}', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
            json_str = json_str.strip('`').strip()
            ai_plan = json.loads(json_str)
            
            # Ensure all required fields are present, use fallbacks if needed
            if "test_name" not in ai_plan:
                ai_plan["test_name"] = test_name
                
            if "description" not in ai_plan:
                ai_plan["description"] = f"This test verifies the AVAS system behavior as specified in requirement {req_id}"
                
            if "variant" not in ai_plan:
                ai_plan["variant"] = "var01"
                
            # Add missing fields
            ai_plan["requirement_id"] = req_id
            ai_plan["covered_requirement"] = extract_covered_requirements(req_id, req_description)
            
            return ai_plan
        else:
            # If JSON extraction failed, use rule-based approach
            test_steps = generate_detailed_test_steps(req_parts, params)
            
            test_plan = {
                "requirement_id": req_id,
                "test_name": test_name,
                "description": f"This test verifies the AVAS system behavior as specified in requirement {req_id}",
                "variant": "var01",
                "steps": test_steps,
                "covered_requirement": extract_covered_requirements(req_id, req_description)
            }
            return test_plan
            
    except Exception as e:
        st.error(f"Error processing AI response: {str(e)}")
        # Fall back to rule-based approach
        test_steps = generate_detailed_test_steps(req_parts, params)
        
        test_plan = {
            "requirement_id": req_id,
            "test_name": test_name,
            "description": f"This test verifies the AVAS system behavior as specified in requirement {req_id}",
            "variant": "var01",
            "steps": test_steps,
            "covered_requirement": extract_covered_requirements(req_id, req_description)
        }
        return test_plan

# Function to parse csv or excel input
def parse_input_file(uploaded_file):
    """Parse CSV or Excel file to extract requirements"""
    requirements = []
    
    try:
        # Determine file type
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."
        
        # Check for required columns
        required_columns = ["Requirement ID", "Requirement Description"]
        # Try alternate column names if needed
        alternate_columns = {
            "Requirement ID": ["ID", "REQ_ID", "RequirementID", "Req ID", "Req. ID"],
            "Requirement Description": ["Description", "DESC", "Text", "Requirement Text", "Req. Description"]
        }
        
        # Check if required columns exist, or find alternatives
        column_mapping = {}
        for req_col, alt_cols in alternate_columns.items():
            if req_col in data.columns:
                column_mapping[req_col] = req_col
            else:
                found = False
                for alt_col in alt_cols:
                    if alt_col in data.columns:
                        column_mapping[req_col] = alt_col
                        found = True
                        break
                if not found:
                    return None, f"Required column '{req_col}' or alternatives not found in the file."
        
        # Extract requirements using the mapped columns
        for _, row in data.iterrows():
            req_id = str(row[column_mapping["Requirement ID"]])
            req_desc = str(row[column_mapping["Requirement Description"]])
            
            # Skip empty rows
            if pd.isna(req_id) or pd.isna(req_desc) or req_id.strip() == "" or req_desc.strip() == "":
                continue
                
            requirements.append({
                "id": req_id.strip(),
                "description": req_desc.strip()
            })
            
        return requirements, None
        
    except Exception as e:
        return None, f"Error parsing file: {str(e)}"

# Function to provide detailed analysis of the requirement
def analyze_requirement(req_id, req_description):
    """Analyze requirement with AI to provide insights before test plan generation"""
    prompt = f"""
    Please analyze the following automotive AVAS (Acoustic Vehicle Alerting System) requirement:
    
    Requirement ID: {req_id}
    Requirement Description: {req_description}
    
    Provide a concise analysis including:
    1. The main objective of this requirement
    2. Key parameters and their significance (e.g., vehicle speed, sound command)
    3. Conditions that trigger the required behavior
    4. Expected system response
    5. Potential challenges in testing this requirement
    
    Format your response as a brief technical analysis suitable for test engineers.
    """
    
    try:
        response, error = call_r1_distill_llama(prompt, temperature=0.3, max_tokens=1000)
        
        if error:
            return f"Unable to analyze requirement: {error}"
        
        return response
    except Exception as e:
        return f"Error during requirement analysis: {str(e)}"

# Main UI
def main():
    st.title("Automotive Test Plan Generator")
    st.markdown("### Generate detailed test plans for automotive AVAS requirements")
    
    # API configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key to use AI-enhanced test plan generation")
        
        if api_key:
            st.session_state['api_key'] = api_key
            st.success("API key saved for this session")
        
        use_api = st.checkbox("Use AI-enhanced generation", value=True, help="Use AI model for improved test plan generation")
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This tool generates detailed test plans for Acoustic Vehicle Alerting System (AVAS) requirements.
        
        It can parse requirements in IF/THEN or SI/ALORS format, extract key parameters,
        and produce structured test steps with appropriate actions and verifications.
        
        For best results, provide well-structured requirements with clear conditions and expected outcomes.
        """)

    # Input method selection
    input_method = st.radio("Choose input method:", ["Single Requirement", "Batch Processing (CSV/Excel)"])
    
    if input_method == "Single Requirement":
        # Single requirement input
        req_id = st.text_input("Requirement ID", help="Enter the requirement ID (e.g., WAVE5-AVAS-0010(0))")
        req_description = st.text_area("Requirement Description", height=150, 
                                      help="Enter the requirement description using IF/THEN or SI/ALORS format for best results")
        
        analyze_button = st.button("Analyze Requirement")
        generate_button = st.button("Generate Test Plan")
        
        if analyze_button and req_description:
            with st.spinner("Analyzing requirement..."):
                analysis = analyze_requirement(req_id, req_description)
                st.subheader("Requirement Analysis")
                st.markdown(analysis)
        
        if generate_button and req_description:
            with st.spinner("Generating test plan..."):
                # Check if API key is available and AI is enabled
                use_api_key = use_api and 'api_key' in st.session_state and st.session_state['api_key']
                
                # Generate test plan
                test_plan = generate_test_plan_with_ai(req_id, req_description, use_api_key=use_api_key)
                
                # Display test plan
                st.success("Test plan generated successfully!")
                display_test_plan(test_plan)
    
    else:
        # Batch processing
        st.markdown("### Upload Requirements File")
        uploaded_file = st.file_uploader("Upload CSV or Excel file with requirements", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            with st.spinner("Parsing file..."):
                requirements, error = parse_input_file(uploaded_file)
                
                if error:
                    st.error(error)
                else:
                    st.success(f"Successfully parsed {len(requirements)} requirements")
                    
                    # Display requirements table
                    req_df = pd.DataFrame(requirements)
                    st.dataframe(req_df, use_container_width=True)
                    
                    # Option to select which requirements to process
                    st.markdown("### Generate Test Plans")
                    all_reqs = st.checkbox("Process all requirements", value=True)
                    
                    selected_reqs = requirements
                    if not all_reqs:
                        # Create multiselect with requirement IDs
                        selected_ids = st.multiselect("Select requirements to process", 
                                                     options=[req["id"] for req in requirements])
                        selected_reqs = [req for req in requirements if req["id"] in selected_ids]
                    
                    # Process button
                    process_button = st.button("Generate Selected Test Plans")
                    
                    if process_button and selected_reqs:
                        with st.spinner(f"Generating {len(selected_reqs)} test plans... This may take a while."):
                            # Check if API key is available and AI is enabled
                            use_api_key = use_api and 'api_key' in st.session_state and st.session_state['api_key']
                            
                            progress_bar = st.progress(0)
                            all_test_plans = []
                            
                            for i, req in enumerate(selected_reqs):
                                # Update progress
                                progress = int((i + 1) / len(selected_reqs) * 100)
                                progress_bar.progress(progress)
                                
                                # Generate test plan
                                test_plan = generate_test_plan_with_ai(req["id"], req["description"], use_api_key=use_api_key)
                                all_test_plans.append(test_plan)
                            
                            # Create zip file with all test plans
                            zip_buffer = create_zip_archive(all_test_plans)
                            
                            # Offer download
                            st.success(f"Successfully generated {len(all_test_plans)} test plans!")
                            st.download_button(
                                label="Download All Test Plans (ZIP)",
                                data=zip_buffer,
                                file_name="automotive_test_plans.zip",
                                mime="application/zip"
                            )
                            
                            # Show the first test plan as example
                            if all_test_plans:
                                st.subheader("Example Test Plan Preview")
                                display_test_plan(all_test_plans[0])

if __name__ == "__main__":
    main()