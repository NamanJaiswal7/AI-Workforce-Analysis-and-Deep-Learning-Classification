import json
import re
import unicodedata
import os
from collections import Counter

# Paths
INPUT_FILE = "data/raw/study_programs_raw.json"
OUTPUT_FILE = "data/processed/programs_cleaned.json"

def clean_text(text):
    """Clean and normalize text while preserving Lithuanian diacritics."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Normalize Unicode (NFC form preserves Lithuanian characters)
    text = unicodedata.normalize("NFC", text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    
    # Remove special characters but keep Lithuanian diacritics
    # Lithuanian alphabet: a ą b c č d e ę ė f g h i į y j k l m n o
    #                      p r s š t u ų ū v z ž
    text = re.sub(r"[^\w\s\-āąčęėįšųūžĄČĘĖĮŠŲŪŽ]", " ", text, flags=re.UNICODE)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Lowercase
    text = text.lower()
    
    return text

def process_data():
    print(f"Loading raw data from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    print(f"Loaded {len(records)} records.")
    
    # Filter active records
    active_records = [r for r in records if not r.get("prg_isreg_data")]
    print(f"Active records: {len(active_records)}")
    
    cleaned_records = []
    
    for r in active_records:
        title_lt = r.get("prg_pavad_lt") or ""
        title_en = r.get("prg_pavad_en") or ""
        
        # Combine fields for description (using titles and fields as dummy description if none exists)
        # In actual data, there might be a description field, but let's use what we have
        desc_lt = f"{title_lt} {r.get('stud_sritis_lt', '')} {r.get('stud_posritis_lt', '')}"
        
        cleaned_lt = clean_text(desc_lt)
        
        if cleaned_lt:
            r["cleaned_text_lt"] = cleaned_lt
            cleaned_records.append(r)
            
    print(f"Records with valid text: {len(cleaned_records)}")
    
    # Save cleaned records
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)
        
    print(f"Saved cleaned records to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()
