"""
AshbyCRISPR Setup Test - Verify all dependencies
Run this before deploying to HuggingFace
"""

import sys
import os

print("🧬 AshbyCRISPR Setup Test")
print("="*60)

# Test 1: Python Version
print("\n✓ Test 1: Python Version")
print(f"  Python {sys.version}")
if sys.version_info < (3, 8):
    print("  ❌ ERROR: Need Python 3.8 or higher!")
    sys.exit(1)
else:
    print("  ✅ Python version OK")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Test 2: Import Dependencies
print("\n✓ Test 2: Check Dependencies")

try:
    import gradio as gr
    print(f"  ✅ gradio {gr.__version__}")
except ImportError:
    print("  ❌ gradio missing. Run: pip install gradio")
    sys.exit(1)

try:
    import datasets
    print(f"  ✅ datasets {datasets.__version__}")
except ImportError:
    print("  ❌ datasets missing. Run: pip install datasets")
    sys.exit(1)

try:
    import sentence_transformers
    print(f"  ✅ sentence-transformers {sentence_transformers.__version__}")
except ValueError as e:
    if "tf-keras" in str(e):
        print("  ❌ tf-keras missing. Run: pip install tf-keras")
        sys.exit(1)
    else:
        raise
except ImportError:
    print("  ❌ sentence-transformers missing. Run: pip install sentence-transformers")
    sys.exit(1)

try:
    import faiss
    print(f"  ✅ faiss-cpu installed")
except ImportError:
    print("  ❌ faiss missing. Run: pip install faiss-cpu")
    sys.exit(1)

try:
    import pandas as pd
    print(f"  ✅ pandas {pd.__version__}")
except ImportError:
    print("  ❌ pandas missing. Run: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
    print(f"  ✅ numpy {np.__version__}")
except ImportError:
    print("  ❌ numpy missing. Run: pip install numpy")
    sys.exit(1)

try:
    from huggingface_hub import InferenceClient
    print(f"  ✅ huggingface-hub installed")
except ImportError:
    print("  ❌ huggingface-hub missing. Run: pip install huggingface-hub")
    sys.exit(1)

try:
    import tqdm
    print(f"  ✅ tqdm installed")
except ImportError:
    print("  ❌ tqdm missing. Run: pip install tqdm")
    sys.exit(1)

# Test 3: HuggingFace Connection
print("\n✓ Test 3: Test HuggingFace Connection")
try:
    from datasets import load_dataset
    print("  📥 Downloading test dataset...")
    test_dataset = load_dataset("yhqu/crispr_delivery", split="train[:10]")
    print(f"  ✅ Loaded {len(test_dataset)} records")
except Exception as e:
    print(f"  ❌ Error: {str(e)[:100]}")
    print("  Check internet connection")
    sys.exit(1)

# Test 4: Embedding Model
print("\n✓ Test 4: Test Sentence Transformers")
try:
    from sentence_transformers import SentenceTransformer
    print("  📥 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  🔢 Testing embeddings...")
    test_embedding = model.encode(["CRISPR test"])
    print(f"  ✅ Embedding dimension: {test_embedding.shape[1]}")
except Exception as e:
    print(f"  ❌ Error: {str(e)[:100]}")
    sys.exit(1)

# Test 5: FAISS
print("\n✓ Test 5: Test FAISS Index")
try:
    import faiss
    dimension = 384
    index = faiss.IndexFlatIP(dimension)
    test_vectors = np.random.random((10, dimension)).astype('float32')
    index.add(test_vectors)
    scores, indices = index.search(test_vectors[:1], 5)
    print(f"  ✅ FAISS working")
except Exception as e:
    print(f"  ❌ Error: {str(e)[:100]}")
    sys.exit(1)

# Test 6: Check Data Folder
print("\n✓ Test 6: Check Data Folder")
from pathlib import Path
data_dir = Path("data")

if data_dir.exists():
    print(f"  ✅ data/ folder found")
    
    folders = ['embeddings', 'summaries', 'terms', 'genomes', 'screens']
    for folder in folders:
        folder_path = data_dir / folder
        if folder_path.exists():
            files = list(folder_path.glob("*"))
            print(f"  ✅ {folder}/ - {len(files)} files")
        else:
            print(f"  ⚠️ {folder}/ - not found")
else:
    print(f"  ⚠️ data/ folder not found (will use HF datasets only)")

# Test 7: Quick Integration Test
print("\n✓ Test 7: Integration Test")
try:
    print("  🧪 Testing full pipeline...")
    
    # Load small dataset
    dataset = load_dataset("yhqu/crispr_delivery", split="train[:20]")
    
    # Create texts
    texts = []
    for item in dataset:
        text = " | ".join([f"{k}: {v}" for k, v in item.items() if v])
        texts.append(text)
    
    # Generate embeddings
    embeddings = model.encode(texts[:10], show_progress_bar=False)
    
    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Test search
    query_embedding = model.encode(["CRISPR delivery methods"])
    scores, indices = index.search(query_embedding, 3)
    
    print(f"  ✅ Pipeline working! Found {len(indices[0])} results")
    
except Exception as e:
    print(f"  ❌ Integration test failed: {str(e)[:100]}")
    sys.exit(1)

# All tests passed!
print("\n" + "="*60)
print("🎉 ALL TESTS PASSED!")
print("="*60)
print("\n✅ Your system is ready!")
print("\n📝 Next steps:")
print("  1. Upload app.py to HuggingFace Space")
print("  2. Upload requirements.txt")
print("  3. Upload data/ folder (if you have it)")
print("  4. Wait for build (~5-10 min)")
print("  5. Test your chatbot!")
print("\n🚀 Good luck!")
