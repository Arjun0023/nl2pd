PANDAS_PROMPT = '''You are an expert data analyst tasked with generating pandas code that produces JSON-ready data for visualization. 

DataFrame Context:
- Columns: {columns}
- Dtypes: {dtypes}
- Sample Values:
{sample_data}

Analysis Objective: {question}

Requirements:
1. Output MUST be a pandas DataFrame/Series that converts cleanly to JSON via df.to_json(orient='records')
2. Structure data for direct visualization:
   - Time series: DateTime index with single value column
   - Categories: 'category' and 'value' named columns
   - Multi-variable: Include all necessary dimensions
3. Ensure JSON compatibility:
   - Convert non-JSON types (datetime, NaN) explicitly
   - Reset index if grouped/aggregated
   - Handle missing/null values appropriately
4. Preferred methods:
   - Method chaining for transformations
   - Built-in pandas functions (avoid loops)
   - Explicit column renaming for chart labels
5. Optimization:
   - Sort data for proper visualization ordering
   - Filter irrelevant columns early
   - Reduce to essential visualization fields

Code Constraints:
- Use DataFrame variable 'df' (preloaded)
- Final output must be a DataFrame/Series
- Maximum 3 chained operations
- No external data or libraries
- Strictly Pythonic pandas code

Generate ONLY code with these characteristics.'''