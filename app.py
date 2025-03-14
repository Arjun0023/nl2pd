from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
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

# Helper function to sanitize data before JSON serialization
def sanitize_for_json(obj):
    """Convert data to JSON-safe format using the custom encoder"""
    return json.loads(json.dumps(obj, cls=NpEncoder))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(...)):
    """Upload Excel or CSV file, convert Excel to CSV if needed, and store as a pandas DataFrame"""
    if file.filename.endswith(('.csv', '.xlsx', '.xls')):
        contents = await file.read()
        
        try:
            # Store original file info
            original_filename = file.filename
            original_file_type = "csv" if file.filename.endswith('.csv') else "excel"
            
            # Convert to pandas dataframe based on file type
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(contents))
            else:  # Excel file
                excel_df = pd.read_excel(io.BytesIO(contents))
                
                # Convert Excel DataFrame to CSV format (in memory)
                csv_buffer = io.StringIO()
                excel_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                # Read back the CSV data
                df = pd.read_csv(csv_buffer)
            
            # Store the dataframe and file info with the session ID
            uploaded_df[session_id] = df
            uploaded_file_info[session_id] = {
                "original_filename": original_filename,
                "original_type": original_file_type,
                "converted_to_csv": original_file_type == "excel"
            }
            
            # Get column information
            columns = df.columns.tolist()
            
            # Get the first 10 rows
            sample_data = df.head(10).replace({np.nan: None})
            # Ensure all data is properly sanitized for JSON
            sample_data_dict = sanitize_for_json(sample_data.to_dict(orient='records'))
            
            conversion_message = ""
            if original_file_type == "excel":
                conversion_message = " Excel file was converted to CSV format for processing."
            
            # Return sanitized data using JSONResponse with the custom encoder
            response_data = {
                "filename": original_filename,
                "columns": columns,
                "num_rows_total": len(df),
                "first_10_rows": sample_data_dict,
                "converted_to_csv": original_file_type == "excel",
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
    
    # Construct prompt for Gemini
    prompt = f"""
    I have a pandas DataFrame with the following structure:
    
    Columns: {df_info['columns']}
    Data types: {df_info['dtypes']}
    
    Here are the first few rows:
    {df.head(5).fillna('NaN').to_string()}
    
    Question: {question}
    
    Please generate ONLY valid Python code using pandas to answer this question.
    Only include the code needed to perform the requested operations on a DataFrame called 'df'.
    Do not include explanations, print statements, or any non-code content.
    """
    
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
            return JSONResponse(
                status_code=400,
                content={"error": "Generated code contains potentially unsafe operations."}
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
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Error executing the generated code: {str(e)}",
                    "code": code
                }
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating or executing code: {str(e)}")

@app.get("/")
async def root():
    return JSONResponse(content={"message": "Welcome to the Data Analysis API. Upload a CSV or Excel file and ask questions about your data."})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)