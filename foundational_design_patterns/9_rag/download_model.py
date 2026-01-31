"""
Script to download the sentence-transformers model with SSL verification disabled.
"""
import os
import sys

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import ssl_fix  # Apply SSL bypass for corporate networks

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
