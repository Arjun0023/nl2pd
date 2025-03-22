from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import uvicorn
import os
import re
from typing import Dict, Optional
import ast
import google.generativeai as genai
import json
from summary_prompt import SUMMARY_PROMPT
from pandas_prompt import PANDAS_PROMPT
from codefix_prompt import FIX_CODE_PROMPT

app = FastAPI(title="Data Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom JSON encoder to handle NaN, Infinity, and other non-JSON serializable values
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None  # Replace NaN and Infinity with None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

uploaded_df = {}
uploaded_file_info = {}  # Store original file info for reference

# Configure Gemini API
def get_genai_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Gemini client: {str(e)}")

def get_genai_client_thinkingmodel():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Gemini client: {str(e)}")

# Helper function to sanitize data before JSON serialization
def sanitize_for_json(data):
    """Helper function to make data JSON serializable"""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(i) for i in data]
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    elif pd.isna(data):
        return None
    else:
        return data

async def generate_insights_from_gemini(df):
    """Generate insights from Gemini based on the dataframe"""
    # Configure the API key
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    
    # Convert first 100 rows to CSV string format
    data_sample = df.head(100)
    csv_buffer = io.StringIO()
    data_sample.to_csv(csv_buffer, index=False)
    csv_text = csv_buffer.getvalue()
    
    # Set up the model
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Create the prompt with system instructions and example
    system_instruction = "You're an expert data analyst, your task is to ask 4 insightful questions on this given piece of data to plot a visualisation out of this data:"
    
    example_response = """{
  "question": [
    "your question 1",
    "your question 2",
    "your question 3",
    "your question 4"
  ]
}"""
    
    # The full prompt that includes the system instruction, example, and the actual data
    prompt = f"{system_instruction}\n\nExample output format:\n{example_response}\n\nData to analyze:\n{csv_text}\n\nGenerate 4 insightful questions for this dataset"
    
    # Generate content with structured output
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json"
    }
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Parse the response
        try:
            # Check if response.text exists and contains valid JSON
            if hasattr(response, 'text'):
                return json.loads(response.text)
            else:
                # Alternative ways to get the response content if 'text' attribute doesn't exist
                if hasattr(response, 'parts'):
                    response_text = response.parts[0].text
                    return json.loads(response_text)
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    content = response.candidates[0].content
                    if hasattr(content, 'parts') and len(content.parts) > 0:
                        response_text = content.parts[0].text
                        return json.loads(response_text)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error parsing response: {str(e)}")
            # If we can't parse JSON, try to extract text and format it
            if hasattr(response, 'text'):
                raw_text = response.text
            elif hasattr(response, 'parts'):
                raw_text = response.parts[0].text
            else:
                raw_text = str(response)
                
            # Try to format the raw text as JSON
            try:
                # Look for JSON-like content in the response
                start_idx = raw_text.find("{")
                end_idx = raw_text.rfind("}")
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = raw_text[start_idx:end_idx+1]
                    return json.loads(json_str)
            except:
                pass
                
            # If all else fails, create a simple structure with the raw text
            return {"question": ["Could not parse response properly.", 
                              "Please check the data format.",
                              "Consider using a different prompt.",
                              "Raw response may contain insights but in wrong format."]}
    except Exception as e:
        print(f"Error generating insights: {str(e)}")
        return {"question": ["Could not generate insights from the data.",
                          "Error connecting to Gemini API.",
                          "Please check your API key and network connection.",
                          "Try with a smaller dataset sample."]}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(...)):
    """Upload Excel or CSV file, convert Excel to CSV if needed, store as a pandas DataFrame and generate insights"""
    if file.filename.endswith(('.csv', '.xlsx', '.xls')):
        contents = await file.read()
        
        try:
            # Store original file info
            original_filename = file.filename
            original_file_type = "csv" if file.filename.endswith('.csv') else "excel"
            encoding_used = "utf-8"  # Default encoding
            
            # Convert to pandas dataframe based on file type
            if file.filename.endswith('.csv'):
                # Try reading with different encodings if UTF-8 fails
                try:
                    df = pd.read_csv(io.BytesIO(contents))
                except UnicodeDecodeError:
                    # Try with different encodings
                    encodings = ['latin-1', 'iso-8859-1', 'windows-1252', 'cp1252']
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                            encoding_used = encoding
                            break
                        except Exception:
                            continue
                    else:
                        raise HTTPException(status_code=400, 
                            detail="Unable to decode CSV file. Please try saving it with UTF-8 encoding.")
            else:  # Excel file
                try:
                    excel_df = pd.read_excel(io.BytesIO(contents))
                    
                    # Convert Excel DataFrame to CSV format (in memory)
                    csv_buffer = io.StringIO()
                    excel_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    # Read back the CSV data
                    df = pd.read_csv(csv_buffer)
                except Exception as excel_error:
                    raise HTTPException(status_code=400, 
                        detail=f"Error processing Excel file: {str(excel_error)}")
            
            # Store the dataframe and file info with the session ID
            uploaded_df[session_id] = df
            uploaded_file_info[session_id] = {
                "original_filename": original_filename,
                "original_type": original_file_type,
                "converted_to_csv": original_file_type == "excel",
                "encoding": encoding_used
            }
            
            # Get column information
            columns = df.columns.tolist()
            
            # Get the first 10 rows for preview
            sample_data = df.head(10).replace({np.nan: None})
            # Ensure all data is properly sanitized for JSON
            sample_data_dict = sanitize_for_json(sample_data.to_dict(orient='records'))
            
            # Generate insights using Gemini
            insights = await generate_insights_from_gemini(df)
            
            conversion_message = ""
            if original_file_type == "excel":
                conversion_message = " Excel file was converted to CSV format for processing."
            elif encoding_used != "utf-8":
                conversion_message = f" File was read using {encoding_used} encoding."
            
            # Return sanitized data using JSONResponse with the custom encoder
            response_data = {
                "filename": original_filename,
                "columns": columns,
                "num_rows_total": len(df),
                "first_10_rows": sample_data_dict,
                "converted_to_csv": original_file_type == "excel",
                "encoding_used": encoding_used,
                "insights": insights,
                "message": f"File uploaded successfully.{conversion_message} You can now ask questions about your data."
            }
            
            return JSONResponse(content=response_data, media_type="application/json")
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")


def validate_code(code: str) -> bool:
    """Validate if the code is safe to execute"""
    # Check for potentially dangerous functions
    dangerous_patterns = [
        r"os\.", r"subprocess\.", r"sys\.", r"shutil\.", r"eval\(", r"exec\(",
        r"__import__", r"open\(", r"file\(", r"importlib"
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            return False
    
    # Validate the code is proper Python syntax
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

@app.post("/ask")
async def ask_question(
    question: str = Form(...), 
    session_id: str = Form(...),
    language: str = Form(...),
    model = Depends(get_genai_client)
):
    """Ask a question about the uploaded data and get a pandas code snippet as answer"""
    # Check if the DataFrame exists for this session
    if session_id not in uploaded_df:
        raise HTTPException(status_code=404, detail="No file uploaded for this session. Please upload a file first.")
    
    df = uploaded_df[session_id]
    file_info = uploaded_file_info.get(session_id, {})
    
    # Get DataFrame info to provide context to the AI
    df_info = {
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "sample_data": sanitize_for_json(df.head(5).replace({np.nan: None}).to_dict()),
        "original_file_type": file_info.get("original_type", "unknown"),
        "converted_to_csv": file_info.get("converted_to_csv", False)
    }
    
    original_question = question
    
    # Check if translation is needed
    needs_translation = language.lower() != "en-us"
    
    if needs_translation:
        # Translate question from user's language to English
        translation_prompt = f"Translate the following text from {language} to English. Return ONLY the translated text with no additional explanations: {question}"
        
        try:
            translation_response = model.generate_content(translation_prompt, generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1024,
            })
            
            # Use the translated question for processing
            question = translation_response.text.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")
    
    # Construct initial prompt for Gemini
    prompt = PANDAS_PROMPT.format(
        columns=df_info['columns'],
        dtypes=df_info['dtypes'],
        sample_data=df.head(5).fillna('NaN').to_string(),
        question=question
    )
    
    try:
        # Generate response
        response = model.generate_content(prompt, generation_config={
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        })
        
        generated_text = response.text
        
        # Extract code
        code_pattern = r"```python\s*(.*?)\s*```"
        code_match = re.search(code_pattern, generated_text, re.DOTALL)
        
        if code_match:
            code = code_match.group(1)
        else:
            # If no code block is found, try to extract code directly
            code = generated_text.strip()
            
        # Validate the generated code
        if not validate_code(code):
            error_message = "Generated code contains potentially unsafe operations."
            if needs_translation:
                error_translation_prompt = f"Translate the following text from English to {language}. Return ONLY the translated text with no additional explanations: {error_message}"
                error_translation = model.generate_content(error_translation_prompt, generation_config={"temperature": 0.1})
                error_message = error_translation.text.strip()
            
            return JSONResponse(
                status_code=400,
                content={"error": error_message}
            )
            
        # Create a local copy of the variables to use in exec
        local_vars = {"df": df.copy()}
        
        # Execute the code
        try:
            exec(code, {"pd": pd, "np": np}, local_vars)
            
            # Get the result (assuming the last variable assigned is the result)
            result = None
            for var_name, var_value in local_vars.items():
                if var_name != "df" and isinstance(var_value, (pd.DataFrame, pd.Series)):
                    result = var_value
            
            # If no result variable was found, use the modified df
            if result is None and "df" in local_vars:
                result = local_vars["df"]
                
            # Convert result to JSON-serializable format using the custom encoder
            result_json = None
            if isinstance(result, pd.DataFrame):
                # Replace NaN values first and limit to 50 rows
                result_limited = result.head(50).replace({np.nan: None})
                result_json = sanitize_for_json(result_limited.to_dict(orient='records'))
                result_type = "dataframe"
            elif isinstance(result, pd.Series):
                # Replace NaN values first and limit to 50 entries
                result_limited = result.head(50).replace({np.nan: None})
                result_json = sanitize_for_json(result_limited.to_dict())
                result_type = "series"
            else:
                # Handle other types of results
                try:
                    # Replace any NumPy or pandas values
                    result_json = sanitize_for_json(result)
                except (TypeError, OverflowError):
                    # If result cannot be serialized to JSON, convert to string
                    result_json = str(result)
                result_type = type(result).__name__
                
            # If translation is needed, translate code comments
            if needs_translation:
                # Extract comments from the code
                comment_pattern = r"#(.+?)(?=\n|$)"
                comments = re.findall(comment_pattern, code)
                
                if comments:
                    comments_text = "\n".join(comments)
                    translation_prompt = f"Translate the following Python code comments from English to {language}. Return ONLY the translated comments, one per line, with no additional explanations:\n\n{comments_text}"
                    
                    try:
                        translation_response = model.generate_content(translation_prompt, generation_config={
                            "temperature": 0.1,
                            "max_output_tokens": 2048,
                        })
                        
                        translated_comments = translation_response.text.strip().split("\n")
                        
                        # Replace comments in the code
                        for i, original_comment in enumerate(comments):
                            if i < len(translated_comments):
                                code = code.replace(
                                    f"#{original_comment}", 
                                    f"#{translated_comments[i]}"
                                )
                    except Exception as e:
                        # If translation fails, keep original comments
                        pass
            
            response_data = {
                "code": code,
                "result": result_json,
                "result_type": result_type,
                "file_info": {
                    "original_filename": file_info.get("original_filename", "unknown"),
                    "converted_from_excel": file_info.get("converted_to_csv", False)
                }
            }
            
            # Use JSONResponse with custom encoder to handle any remaining edge cases
            return JSONResponse(content=response_data, media_type="application/json")
            
        except Exception as e:
            # When execution error occurs, send error back to Gemini to fix it
            error_str = str(e)
            fix_prompt = FIX_CODE_PROMPT.format(
                question=question,
                columns=df_info['columns'],
                sample_data=df.head(5).fillna('NaN').to_string(),
                code=code,
                error_str=error_str
            )
            
            try:
                # Generate fixed code
                fix_response = model.generate_content(fix_prompt, generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                })
                
                fixed_text = fix_response.text
                
                # Extract code from the fixed response
                fixed_code_match = re.search(code_pattern, fixed_text, re.DOTALL)
                
                if fixed_code_match:
                    fixed_code = fixed_code_match.group(1)
                else:
                    # If no code block is found, try to extract code directly
                    fixed_code = fixed_text.strip()
                
                # Validate the fixed code
                if not validate_code(fixed_code):
                    error_message = "Generated code contains potentially unsafe operations."
                    original_error = error_str
                    
                    if needs_translation:
                        error_translation_prompt = f"Translate the following text from English to {language}. Return ONLY the translated text with no additional explanations: {error_message}"
                        original_error_prompt = f"Translate the following text from English to {language}. Return ONLY the translated text with no additional explanations: {original_error}"
                        
                        try:
                            error_translation = model.generate_content(error_translation_prompt, generation_config={"temperature": 0.1})
                            original_error_translation = model.generate_content(original_error_prompt, generation_config={"temperature": 0.1})
                            
                            error_message = error_translation.text.strip()
                            original_error = original_error_translation.text.strip()
                        except Exception:
                            # If translation fails, use original error messages
                            pass
                    
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": error_message,
                            "original_error": original_error,
                            "original_code": code,
                            "fixed_code": fixed_code
                        }
                    )
                
                # Try executing the fixed code
                local_vars = {"df": df.copy()}
                exec(fixed_code, {"pd": pd, "np": np}, local_vars)
                
                # Get the result from the fixed code
                result = None
                for var_name, var_value in local_vars.items():
                    if var_name != "df" and isinstance(var_value, (pd.DataFrame, pd.Series)):
                        result = var_value
                
                # If no result variable was found, use the modified df
                if result is None and "df" in local_vars:
                    result = local_vars["df"]
                    
                # Convert result to JSON-serializable format
                result_json = None
                if isinstance(result, pd.DataFrame):
                    result_limited = result.head(50).replace({np.nan: None})
                    result_json = sanitize_for_json(result_limited.to_dict(orient='records'))
                    result_type = "dataframe"
                elif isinstance(result, pd.Series):
                    result_limited = result.head(50).replace({np.nan: None})
                    result_json = sanitize_for_json(result_limited.to_dict())
                    result_type = "series"
                else:
                    try:
                        result_json = sanitize_for_json(result)
                    except (TypeError, OverflowError):
                        result_json = str(result)
                    result_type = type(result).__name__
                
                # Translate fixed code comments if needed
                if needs_translation:
                    # Extract comments from the fixed code
                    comment_pattern = r"#(.+?)(?=\n|$)"
                    comments = re.findall(comment_pattern, fixed_code)
                    
                    if comments:
                        comments_text = "\n".join(comments)
                        translation_prompt = f"Translate the following Python code comments from English to {language}. Return ONLY the translated comments, one per line, with no additional explanations:\n\n{comments_text}"
                        
                        try:
                            translation_response = model.generate_content(translation_prompt, generation_config={
                                "temperature": 0.1,
                                "max_output_tokens": 2048,
                            })
                            
                            translated_comments = translation_response.text.strip().split("\n")
                            
                            # Replace comments in the code
                            for i, original_comment in enumerate(comments):
                                if i < len(translated_comments):
                                    fixed_code = fixed_code.replace(
                                        f"#{original_comment}", 
                                        f"#{translated_comments[i]}"
                                    )
                        except Exception:
                            # If translation fails, keep original comments
                            pass
                
                # Translate error message if needed
                original_error = error_str
                if needs_translation:
                    error_translation_prompt = f"Translate the following text from English to {language}. Return ONLY the translated text with no additional explanations: {error_str}"
                    try:
                        error_translation = model.generate_content(error_translation_prompt, generation_config={"temperature": 0.1})
                        original_error = error_translation.text.strip()
                    except Exception:
                        # If translation fails, use original error message
                        pass
                
                # Return the fixed code and result
                response_data = {
                    "code": fixed_code,
                    "result": result_json,
                    "result_type": result_type,
                    "file_info": {
                        "original_filename": file_info.get("original_filename", "unknown"),
                        "converted_from_excel": file_info.get("converted_to_csv", False)
                    },
                    "error_fixed": True,
                    "original_error": original_error
                }
                
                return JSONResponse(content=response_data, media_type="application/json")
                
            except Exception as fix_error:
                # If fixing also fails, return both the original error and the fixing error
                original_error = error_str
                fixing_error = str(fix_error)
                
                if needs_translation:
                    original_error_prompt = f"Translate the following text from English to {language}. Return ONLY the translated text with no additional explanations: {error_str}"
                    fixing_error_prompt = f"Translate the following text from English to {language}. Return ONLY the translated text with no additional explanations: {fixing_error}"
                    
                    try:
                        original_error_translation = model.generate_content(original_error_prompt, generation_config={"temperature": 0.1})
                        fixing_error_translation = model.generate_content(fixing_error_prompt, generation_config={"temperature": 0.1})
                        
                        original_error = original_error_translation.text.strip()
                        fixing_error = fixing_error_translation.text.strip()
                    except Exception:
                        # If translation fails, use original error messages
                        pass
                
                error_message = f"Original error: {original_error}\nError fixing code: {fixing_error}"
                
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": error_message,
                        "original_code": code,
                        "fixing_attempt_failed": True
                    }
                )
    
    except Exception as e:
        error_message = f"Error generating or executing code: {str(e)}"
        
        if needs_translation:
            translation_prompt = f"Translate the following text from English to {language}. Return ONLY the translated text with no additional explanations: {error_message}"
            
            try:
                translation_response = model.generate_content(translation_prompt, generation_config={"temperature": 0.1})
                error_message = translation_response.text.strip()
            except Exception:
                # If translation fails, use original error message
                pass
        
        raise HTTPException(status_code=500, detail=error_message)
    
@app.post("/summarize")
async def summarize_data(request: Request):
    try:
        # Parse the JSON body from the request
        data = await request.json()
        
        # Extract user question and data from the request
        user_question = data.get("question", "")
        input_data = data.get("data", {})
        language = data.get("language", "")
        # Handle empty data
        if not input_data:
            raise HTTPException(status_code=400, detail="No data provided for summarization")
            
        # Convert data to string representation for the model
        data_str = json.dumps(input_data, cls=NpEncoder, indent=2)
        
        # Get Gemini client
        model = get_genai_client()
        
        # Create a combined prompt without using system role
        prompt = SUMMARY_PROMPT.format(
            question=user_question,
            language=language,
            data=data_str
        )
        
        # Call the Gemini model with the combined prompt
        response = model.generate_content(prompt)
        
        # Return the summary
        return {"summary": response.text}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summarization: {str(e)}")

@app.get("/")
async def root():
    return JSONResponse(content={"message": "Welcome to the Data Analysis API. Upload a CSV or Excel file and ask questions about your data."})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)