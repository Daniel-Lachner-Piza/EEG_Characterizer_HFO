"""
Utility functions for file I/O operations.
"""

from pathlib import Path
from typing import List


def get_files_in_folder(folder_path: str, extension: str) -> List[Path]:
    """
    Get all files in a folder with a specific extension.
    
    Parameters:
    folder_path (str): Path to the folder to search
    extension (str): File extension to search for (e.g., '.lay', '.edf')
    
    Returns:
    List[Path]: List of Path objects for files with the specified extension
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []
    
    # Use glob to find files with the specified extension
    pattern = f"*{extension}"
    return list(folder.glob(pattern))
