"""
Data Loader and Preprocessing Pipeline for Amazon Customer Support on Twitter.
Extracts, cleans, and pairs customer inquiries with historic brand resolutions.
"""

import os
import re
import json
from typing import List, Dict, Any, Generator, Optional
import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, BRAND_HANDLE

def clean_tweet_text(text: str) -> str:
    """
    Cleans raw tweet text:
    - Normalizes excessive whitespace
    - Anonymizes random user mention numbers (@123456)
    - Retains semantic meaning of URLs and punctuation
    """
    if not text:
        return ""
    # Replace anonymized user IDs (@123456) with generic @user
    text = re.sub(r'@[0-9]{4,}', '@user', text)
    # Remove redundant brand tag at the beginning if present
    text = re.sub(r'^@AmazonHelp\s*', '', text, flags=re.IGNORECASE)
    # Normalize multiple whitespace/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_conversation_turns(conv_text: str) -> Optional[Dict[str, str]]:
    """
    Parses conversation string into customer inquiry and support agent reply.
    Format is typically:
    Customer: ...
    Support: ...
    """
    turns = conv_text.split("\n")
    cust_parts = []
    support_parts = []
    current_role = None

    for line in turns:
        line = line.strip()
        if line.startswith("Customer:"):
            current_role = "Customer"
            cust_parts.append(line[len("Customer:"):].strip())
        elif line.startswith("Support:"):
            current_role = "Support"
            support_parts.append(line[len("Support:"):].strip())
        elif current_role == "Customer":
            cust_parts.append(line)
        elif current_role == "Support":
            support_parts.append(line)

    if not cust_parts or not support_parts:
        return None

    cust_text = clean_tweet_text(" ".join(cust_parts))
    supp_text = clean_tweet_text(" ".join(support_parts))

    # Basic English heuristics: must contain common English stop words
    english_words = {"the", "to", "and", "is", "my", "your", "for", "in", "it", "on", "have", "with", "at", "not", "this", "help"}
    cust_words = set(re.findall(r'\b[a-z]{2,}\b', cust_text.lower()))
    if len(cust_words & english_words) < 2:
        return None

    if len(cust_text.split()) < 4 or len(supp_text.split()) < 4:
        return None

    return {
        "customer_query": cust_text,
        "agent_reply": supp_text
    }

def process_and_extract_pairs(
    raw_parquet_path: str = "data/raw/conversations.parquet",
    output_jsonl_path: str = "data/processed/amazon_help_pairs.jsonl",
    max_pairs: int = 10000
) -> int:
    """
    Extracts high-quality English customer-agent conversation pairs for AmazonHelp.
    """
    if not os.path.exists(raw_parquet_path):
        raise FileNotFoundError(f"Raw parquet not found at {raw_parquet_path}")

    print(f"Reading conversations from {raw_parquet_path}...")
    df = pd.read_parquet(raw_parquet_path, columns=["company", "conversation"])
    amz_df = df[df["company"] == "AmazonHelp"]
    print(f"Total AmazonHelp conversations found: {len(amz_df)}")

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    count = 0

    with open(output_jsonl_path, "w", encoding="utf-8") as f_out:
        for idx, row in amz_df.iterrows():
            conv = row["conversation"]
            parsed = parse_conversation_turns(conv)
            if parsed:
                parsed["id"] = f"amz_{idx}"
                f_out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                count += 1
                if count >= max_pairs:
                    break

    print(f"Successfully extracted and saved {count} conversation pairs to {output_jsonl_path}")
    return count

def load_processed_pairs(jsonl_path: str = "data/processed/amazon_help_pairs.jsonl") -> List[Dict[str, Any]]:
    """Loads processed conversation pairs."""
    if not os.path.exists(jsonl_path):
        return []
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
