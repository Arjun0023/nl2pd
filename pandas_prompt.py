PANDAS_PROMPT = '''
I have a pandas DataFrame with the following structure:

Columns: {columns}
Data types: {dtypes}

Here are the first few rows:
{sample_data}

Question: {question}

Please generate ONLY valid Python code using pandas to answer this question.
Only include the code needed to perform the requested operations on a DataFrame called 'df'.
Do not include explanations, print statements, or any non-code content.
'''