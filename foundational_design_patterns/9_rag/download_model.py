"""
Script to download the sentence-transformers model with SSL verification disabled.
"""
import os
import sys

from pathlib import Path

ROOT_DIR = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "ssl_fix.py").exists()
)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example

configure_example(__file__)

from sentence_transformers import SentenceTransformer

print("Downloading model: all-MiniLM-L6-v2...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"✅ Model downloaded successfully!")
    print(f"   Model dimension: {model.get_sentence_embedding_dimension()}")
    print(f"   Cache location: {model._model_card_vars.get('model_id', 'N/A')}")
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    raise
