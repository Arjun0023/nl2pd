FIX_CODE_PROMPT = '''
You are an expert pandas developer tasked with fixing broken data analysis code.

Question from user: {question}

DataFrame structure:
Columns: {columns}

First few rows of data:
{sample_data}

The following code was generated but produced an error:
```python
{code}
```

Error message: "{error_str}"

Your task: Fix this code to correctly answer the user's question.
Focus on addressing the specific error while ensuring the code properly handles the data structure.
Examine the column names carefully and make sure they match the actual DataFrame.
Return ONLY the corrected Python code without any explanations or markdown formatting.
'''