"""
Data Loading Module.

Provides simple, robust functions to load CSV and Excel files into Pandas DataFrames.
No data transformations or cleaning are performed here.
"""
import os
import pandas as pd


from typing import Union, Any


def load_csv(file_or_path: Union[str, Any]) -> pd.DataFrame:
    """
    Loads a CSV file from a path string or file-like buffer into a Pandas DataFrame.
    
    Parameters:
        file_or_path: The filesystem path (str) or file-like buffer (e.g. Streamlit UploadedFile).
        
    Returns:
        pd.DataFrame: The loaded dataset.
        
    Raises:
        FileNotFoundError: If a specified filesystem path does not exist.
        ValueError: If the file is not a CSV or cannot be parsed.
    """
    if isinstance(file_or_path, str):
        # Verify that the file actually exists on the filesystem
        if not os.path.exists(file_or_path):
            raise FileNotFoundError(f"File not found at path: '{file_or_path}'")
        if not file_or_path.lower().endswith(".csv"):
            raise ValueError(f"Expected a .csv file, but received: '{file_or_path}'")
    else:
        name = getattr(file_or_path, "name", "")
        if name and not name.lower().endswith(".csv"):
            raise ValueError(f"Expected a .csv file, but received: '{name}'")
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
    
    try:
        df = pd.read_csv(file_or_path)
        return df
    except Exception as e:
        source_name = file_or_path if isinstance(file_or_path, str) else getattr(file_or_path, "name", "uploaded_file")
        raise ValueError(f"Failed to read CSV file '{source_name}': {str(e)}")


def load_excel(file_or_path: Union[str, Any]) -> pd.DataFrame:
    """
    Loads an Excel (.xlsx or .xls) file from a path string or file-like buffer into a Pandas DataFrame.
    
    Parameters:
        file_or_path: The filesystem path (str) or file-like buffer (e.g. Streamlit UploadedFile).
        
    Returns:
        pd.DataFrame: The loaded dataset.
        
    Raises:
        FileNotFoundError: If a specified filesystem path does not exist.
        ValueError: If the file is not an Excel file or cannot be parsed.
    """
    valid_extensions = (".xlsx", ".xls")
    if isinstance(file_or_path, str):
        # Verify that the file actually exists on the filesystem
        if not os.path.exists(file_or_path):
            raise FileNotFoundError(f"File not found at path: '{file_or_path}'")
        if not file_or_path.lower().endswith(valid_extensions):
            raise ValueError(f"Expected an Excel file (.xlsx or .xls), but received: '{file_or_path}'")
    else:
        name = getattr(file_or_path, "name", "")
        if name and not name.lower().endswith(valid_extensions):
            raise ValueError(f"Expected an Excel file (.xlsx or .xls), but received: '{name}'")
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
    
    try:
        df = pd.read_excel(file_or_path)
        return df
    except Exception as e:
        source_name = file_or_path if isinstance(file_or_path, str) else getattr(file_or_path, "name", "uploaded_file")
        raise ValueError(f"Failed to read Excel file '{source_name}': {str(e)}")


def load_dataset(file_or_path: Union[str, Any]) -> pd.DataFrame:
    """
    Convenience function: Automatically detects whether the file is a CSV or Excel
    and calls the appropriate loader. Supports both file paths and file-like objects.
    """
    if isinstance(file_or_path, str):
        target_name = file_or_path.lower()
    else:
        target_name = getattr(file_or_path, "name", "").lower()
        
    if target_name.endswith(".csv"):
        return load_csv(file_or_path)
    elif target_name.endswith((".xlsx", ".xls")):
        return load_excel(file_or_path)
    else:
        source_repr = file_or_path if isinstance(file_or_path, str) else getattr(file_or_path, "name", str(type(file_or_path)))
        raise ValueError(f"Unsupported file format for '{source_repr}'. Please provide a .csv or .xlsx file.")
