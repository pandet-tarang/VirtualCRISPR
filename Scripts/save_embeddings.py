"""
Save Embeddings Script
Run this once to create embeddings for all datasets
Then app.py will load instantly!
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import pickle


def save_embeddings():
    print("🧬 Creating and Saving All Embeddings")
    print("="*60)
    
    # Initialize model
    print("\n📥 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    data_dir = Path("data")
    embeddings_dir = data_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Track what we save
    saved_files = []
    
    # 1. Summaries folder
    print("\n📁 Processing summaries/")
    summaries_path = data_dir / "summaries"
    if summaries_path.exists():
        for file_path in summaries_path.glob("*"):
            if file_path.suffix in ['.csv', '.xlsx', '.xls']:
                category = file_path.stem
                
                # Check if already exists
                emb_file = embeddings_dir / f"{category}.npy"
                if emb_file.exists():
                    print(f"  ⏭️ {category} already exists, skipping")
                    continue
                
                print(f"\n  🔢 Processing {category}...")
                
                try:
                    # Load data
                    df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_excel(file_path)
                    
                    # Limit rows
                    if len(df) > 5000:
                        df = df.head(5000)
                    
                    # Create texts
                    texts = []
                    for _, item in df.iterrows():
                        parts = []
                        for k, v in item.items():
                            if v is not None and pd.notna(v):
                                v_str = str(v)
                                if v_str.strip():
                                    parts.append(f"{k}: {v_str}")
                        texts.append(" | ".join(parts))
                    
                    # Generate embeddings
                    embeddings = model.encode(
                        texts,
                        batch_size=64,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    
                    # Save
                    np.save(emb_file, embeddings)
                    print(f"  ✅ Saved {category}.npy ({len(embeddings)} vectors)")
                    saved_files.append(f"{category}.npy")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
    
    # 2. Terms folder
    print("\n📁 Processing terms/")
    terms_path = data_dir / "terms"
    if terms_path.exists():
        for file_path in terms_path.glob("*"):
            if file_path.suffix in ['.csv', '.xlsx', '.xls']:
                category = file_path.stem
                
                emb_file = embeddings_dir / f"{category}.npy"
                if emb_file.exists():
                    print(f"  ⏭️ {category} already exists, skipping")
                    continue
                
                print(f"\n  🔢 Processing {category}...")
                
                try:
                    df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_excel(file_path)
                    
                    if len(df) > 5000:
                        df = df.head(5000)
                    
                    texts = []
                    for _, item in df.iterrows():
                        parts = []
                        for k, v in item.items():
                            if v is not None and pd.notna(v):
                                v_str = str(v)
                                if v_str.strip():
                                    parts.append(f"{k}: {v_str}")
                        texts.append(" | ".join(parts))
                    
                    embeddings = model.encode(
                        texts,
                        batch_size=64,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    
                    np.save(emb_file, embeddings)
                    print(f"  ✅ Saved {category}.npy ({len(embeddings)} vectors)")
                    saved_files.append(f"{category}.npy")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
    
    # 3. Genomes folder
    print("\n📁 Processing genomes/")
    genomes_path = data_dir / "genomes"
    if genomes_path.exists():
        for file_path in genomes_path.glob("*.tsv"):
            category = file_path.stem
            
            emb_file = embeddings_dir / f"{category}.npy"
            if emb_file.exists():
                print(f"  ⏭️ {category} already exists, skipping")
                continue
            
            print(f"\n  🔢 Processing {category}...")
            
            try:
                df = pd.read_csv(file_path, sep='\t')
                
                if len(df) > 5000:
                    df = df.head(5000)
                
                texts = []
                for _, item in df.iterrows():
                    parts = []
                    for k, v in item.items():
                        if v is not None and pd.notna(v):
                            v_str = str(v)
                            if v_str.strip():
                                parts.append(f"{k}: {v_str}")
                    texts.append(" | ".join(parts))
                
                embeddings = model.encode(
                    texts,
                    batch_size=64,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                np.save(emb_file, embeddings)
                print(f"  ✅ Saved {category}.npy ({len(embeddings)} vectors)")
                saved_files.append(f"{category}.npy")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    # 4. Screens folder
    print("\n📁 Processing screens/")
    screens_path = data_dir / "screens"
    if screens_path.exists():
        files = list(screens_path.glob("*.txt")) + list(screens_path.glob("*.tsv"))
        
        if len(files) > 0:
            emb_file = embeddings_dir / "crispr_screens.npy"
            
            if emb_file.exists():
                print(f"  ⏭️ crispr_screens already exists, skipping")
            else:
                print(f"\n  🔢 Processing {len(files)} screen files...")
                
                # Limit to 20 files
                if len(files) > 20:
                    files = files[:20]
                
                try:
                    all_screens = []
                    for file_path in tqdm(files, desc="  Loading files"):
                        try:
                            df = pd.read_csv(file_path, sep='\t', nrows=1000, dtype=str, low_memory=False)
                            df['source_file'] = file_path.name
                            all_screens.append(df)
                        except:
                            continue
                    
                    if all_screens:
                        combined = pd.concat(all_screens, ignore_index=True)
                        
                        if len(combined) > 5000:
                            combined = combined.head(5000)
                        
                        texts = []
                        for _, item in combined.iterrows():
                            parts = []
                            for k, v in item.items():
                                if v is not None and pd.notna(v):
                                    v_str = str(v)
                                    if v_str.strip():
                                        parts.append(f"{k}: {v_str}")
                            texts.append(" | ".join(parts))
                        
                        print(f"  🔢 Generating embeddings for {len(texts)} screen records...")
                        embeddings = model.encode(
                            texts,
                            batch_size=64,
                            show_progress_bar=True,
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        
                        np.save(emb_file, embeddings)
                        print(f"  ✅ Saved crispr_screens.npy ({len(embeddings)} vectors)")
                        saved_files.append("crispr_screens.npy")
                        
                except Exception as e:
                    print(f"  ❌ Error: {e}")
    
    # 5. HuggingFace datasets (optional)
    print("\n🤗 HuggingFace Datasets")
    print("Note: These are downloaded fresh each time, but embed quickly")
    
    cache_dir = data_dir / "hf_cache"
    cache_dir.mkdir(exist_ok=True)
    
    hf_datasets = [
        ("yhqu/crispr_delivery", "hf_crispr_delivery", 2000),
        ("conjuncts/ncbi-taxonomy-binomial", "hf_taxonomy", 5000),
        ("jo-mengr/ncbi_cell_types_1000", "hf_cell_types", 1000),
        ("dwb2023/crispr-binary-calls", "hf_crispr_calls", 3000),
    ]
    
    for dataset_name, category, max_rows in hf_datasets:
        emb_file = cache_dir / f"{category}.npy"
        data_file = cache_dir / f"{category}.pkl"
        
        if emb_file.exists():
            print(f"  ⏭️ {category} already cached, skipping")
            continue
        
        print(f"\n  📥 Processing {dataset_name}...")
        
        try:
            dataset = load_dataset(dataset_name, split="train", streaming=False)
            
            if len(dataset) > max_rows:
                indices = np.linspace(0, len(dataset)-1, max_rows, dtype=int)
                dataset = dataset.select(indices)
            
            texts = []
            for item in dataset:
                parts = []
                for k, v in item.items():
                    if v is not None:
                        v_str = str(v)[:200] if isinstance(v, (list, dict)) else str(v)
                        if v_str.strip():
                            parts.append(f"{k}: {v_str}")
                texts.append(" | ".join(parts))
            
            embeddings = model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Save embeddings and data
            np.save(emb_file, embeddings)
            with open(data_file, 'wb') as f:
                pickle.dump(dataset, f)
            
            print(f"  ✅ Cached {category} ({len(embeddings)} vectors)")
            saved_files.append(f"{category}.npy")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ EMBEDDING GENERATION COMPLETE!")
    print("="*60)
    print(f"\n📊 Saved {len(saved_files)} embedding files:")
    for f in saved_files:
        print(f"  • {f}")
    
    print(f"\n💾 Location: {embeddings_dir}")
    print("\n🚀 Next time you run app.py, it will load instantly!")


if __name__ == "__main__":
    save_embeddings()
