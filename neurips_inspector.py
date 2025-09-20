import json
import pandas as pd
from collections import Counter

# Load and inspect the JSON structure
def inspect_neurips_data(file_path):
    """Inspect the structure of NeurIPS JSON data"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total papers: {len(data)}")
    print(f"Data type: {type(data)}")
    
    # Look at first few entries
    if isinstance(data, list):
        sample_entry = data[0]
    else:
        # If it's a dict, get first value
        sample_entry = next(iter(data.values()))
    
    print("\nSample entry structure:")
    for key, value in sample_entry.items():
        print(f"  {key}: {type(value).__name__} - {str(value)[:100]}...")
    
    # Check what fields are available across all entries
    all_keys = set()
    for entry in (data if isinstance(data, list) else data.values()):
        all_keys.update(entry.keys())
    
    print(f"\nAll available fields: {sorted(all_keys)}")
    
    # Check for abstract/description fields
    text_fields = [k for k in all_keys if any(term in k.lower() for term in ['abstract', 'description', 'summary', 'text'])]
    print(f"\nPotential text fields for analysis: {text_fields}")
    
    return data

# Run the inspection
data = inspect_neurips_data('/home/bytestorm/Downloads/neurips-2025/NeurIPS 2025 Events.json')