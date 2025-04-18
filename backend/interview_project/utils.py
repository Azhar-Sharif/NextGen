import re
from typing import Optional

def extract_name_from_text(text: str) -> str:
    """
    Extracts the user's first name from the transcribed introduction text.
    
    Args:
        text (str): The transcribed introduction text
        
    Returns:
        str: The extracted first name or "User" if no name is found
    """
    if not text or not isinstance(text, str):
        return "User"
        
    # Pattern to match common name introduction phrases
    name_pattern = r"\b(?:my name is|I'm|I am|call me|this is|myself)\s+([a-zA-Z]+)"
    
    try:
        match = re.search(name_pattern, text, re.IGNORECASE)
        if match:
            extracted_name = match.group(1)
            return extracted_name.title()  # Return capitalized first name
        else:
            return "User"
    except Exception as e:
        print(f"Error extracting name: {e}")
        return "User"

