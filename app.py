"""
Virtual CRISPR Assistant - Local Version
Predicts CRISPR screen results using RAG + LLMs

Based on research: Can LLMs Predict CRISPR Screen Results?
"""

import gradio as gr
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import numpy as np
import faiss
import pandas as pd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


class VirtualCRISPRAssistant:
    """CRISPR screen prediction with RAG"""
    
    def __init__(self, data_dir: str = "data", max_rows_per_dataset: int = 5000):
        print("🧬 Initializing Virtual CRISPR Assistant")
        print("="*60)
        
        self.data_dir = Path(data_dir)
        self.max_rows = max_rows_per_dataset
        
        print("📥 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.indices = {}
        self.data = {}
        self.loaded_sources = []
        
        self.load_all_local_data()
        self.load_huggingface_datasets()
        
        print("\n" + "="*60)
        print(f"✅ Ready! {len(self.indices)} sources loaded")
        print("="*60)
    
    def load_all_local_data(self):
        """Load all local data"""
        print("\n📂 Loading Local Data")
        print("-"*60)
        
        if not self.data_dir.exists():
            print("⚠️ No 'data/' folder found")
            return
        
        folders = {
            'embeddings': self._load_embeddings_folder,
            'summaries': self._load_summaries_folder,
            'terms': self._load_terms_folder,
            'genomes': self._load_genomes_folder,
            'screens': self._load_screens_folder
        }
        
        for folder_name, load_func in folders.items():
            folder_path = self.data_dir / folder_name
            if folder_path.exists():
                print(f"\n📁 {folder_name}/")
                load_func(folder_path)
    
    def _load_embeddings_folder(self, folder_path: Path):
        emb_files = list(folder_path.glob("*.npy"))
        print(f"  Found {len(emb_files)} files")
        
        for emb_file in emb_files:
            category = emb_file.stem
            
            try:
                embeddings = np.load(emb_file, allow_pickle=True)
                
                if not isinstance(embeddings, np.ndarray) or len(embeddings.shape) != 2:
                    continue
                if embeddings.shape[0] == 0 or embeddings.shape[1] != 384:
                    continue
                
                summaries_path = self.data_dir / "summaries"
                data_file = None
                
                for ext in ['.csv', '.xlsx', '.xls']:
                    for name in [category, category.replace('summarized_', '')]:
                        potential = summaries_path / f"{name}{ext}"
                        if potential.exists():
                            data_file = potential
                            break
                    if data_file:
                        break
                
                if data_file:
                    df = pd.read_csv(data_file) if data_file.suffix == '.csv' else pd.read_excel(data_file)
                    
                    min_rows = min(len(df), len(embeddings), self.max_rows)
                    df = df.head(min_rows)
                    embeddings = embeddings[:min_rows]
                    
                    index = faiss.IndexFlatIP(embeddings.shape[1])
                    index.add(embeddings.astype('float32'))
                    
                    dataset = Dataset.from_pandas(df)
                    self.data[f"local_emb_{category}"] = dataset
                    self.indices[f"local_emb_{category}"] = index
                    
                    print(f"  ✅ {category}: {len(df)} records")
                    self.loaded_sources.append(f"Local: {category}")
                    
            except:
                continue
    
    def _load_summaries_folder(self, folder_path: Path):
        files = list(folder_path.glob("*.csv")) + list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.xls"))
        print(f"  Found {len(files)} files")
        
        for file_path in files:
            category = f"summary_{file_path.stem}"
            
            if any(file_path.stem in src for src in self.loaded_sources):
                continue
            
            try:
                df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_excel(file_path)
                
                if len(df) > self.max_rows:
                    df = df.head(self.max_rows)
                
                dataset = Dataset.from_pandas(df)
                self._process_dataset(dataset, category)
                self.loaded_sources.append(f"Local: {file_path.name}")
                
            except:
                continue
    
    def _load_terms_folder(self, folder_path: Path):
        files = list(folder_path.glob("*.csv")) + list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.xls"))
        print(f"  Found {len(files)} files")
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_excel(file_path)
                
                if len(df) > self.max_rows:
                    df = df.head(self.max_rows)
                
                dataset = Dataset.from_pandas(df)
                self._process_dataset(dataset, f"terms_{file_path.stem}")
                self.loaded_sources.append(f"Local: {file_path.name}")
                
            except:
                continue
    
    def _load_genomes_folder(self, folder_path: Path):
        files = list(folder_path.glob("*.tsv"))
        print(f"  Found {len(files)} files")
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path, sep='\t')
                
                if len(df) > self.max_rows:
                    df = df.head(self.max_rows)
                
                dataset = Dataset.from_pandas(df)
                self._process_dataset(dataset, f"genome_{file_path.stem.replace('genome_', '')}")
                self.loaded_sources.append(f"Local: {file_path.name}")
                
            except:
                continue
    
    def _load_screens_folder(self, folder_path: Path):
        files = list(folder_path.glob("*.txt")) + list(folder_path.glob("*.tsv"))
        print(f"  Found {len(files)} files")
        
        if len(files) > 20:
            files = files[:20]
        
        all_screens = []
        for file_path in files:
            try:
                # Read with all columns as strings to avoid type conflicts
                df = pd.read_csv(file_path, sep='\t', nrows=1000, dtype=str, low_memory=False)
                df['source_file'] = file_path.name
                all_screens.append(df)
            except:
                continue
        
        if all_screens:
            combined = pd.concat(all_screens, ignore_index=True)
            
            if len(combined) > self.max_rows:
                combined = combined.head(self.max_rows)
            
            dataset = Dataset.from_pandas(combined)
            self._process_dataset(dataset, "crispr_screens")
            self.loaded_sources.append(f"Local: {len(all_screens)} CRISPR screens")
    
    def load_huggingface_datasets(self):
        print("\n🤗 Loading HuggingFace Datasets")
        print("-"*60)
        
        datasets = [
            ("yhqu/crispr_delivery", "train", "hf_crispr_delivery", 2000),
            ("conjuncts/ncbi-taxonomy-binomial", "train", "hf_taxonomy", 5000),
            ("jo-mengr/ncbi_cell_types_1000", "train", "hf_cell_types", 1000),
            ("dwb2023/crispr-binary-calls", "train", "hf_crispr_calls", 3000),
            ("R2MED/Bioinformatics", "train", "hf_bioinformatics", 5000),
            ("mlfoundations-dev/stackexchange_bioinformatics", "train", "hf_bioinfo_stack", 5000),
        ]
        
        for dataset_info in datasets:
            self._load_hf_dataset(*dataset_info)
    
    def _load_hf_dataset(self, name: str, split: str, category: str, max_rows: int):
        try:
            print(f"\n📥 {name}")
            
            dataset = load_dataset(name, split=split, streaming=False)
            
            if len(dataset) > max_rows:
                indices = np.linspace(0, len(dataset)-1, max_rows, dtype=int)
                dataset = dataset.select(indices)
            
            self._process_dataset(dataset, category)
            self.loaded_sources.append(f"HF: {name}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:50]}")
    
    def _process_dataset(self, dataset, category: str):
        print(f"  🔢 Embedding {category}...")
        
        self.data[category] = dataset
        
        texts = []
        for item in dataset:
            parts = []
            for k, v in item.items():
                if v is not None:
                    v_str = str(v)[:200] if isinstance(v, (list, dict)) else str(v)
                    if v_str.strip():
                        parts.append(f"{k}: {v_str}")
            texts.append(" | ".join(parts))
        
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        
        self.indices[category] = index
        print(f"  ✅ {len(dataset)} docs")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.indices:
            return []
        
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True).astype('float32')
        
        results = []
        for category, index in self.indices.items():
            dataset = self.data[category]
            scores, indices = index.search(query_emb, min(top_k, len(dataset)))
            
            for i, idx in enumerate(indices[0]):
                if idx < len(dataset):
                    results.append({
                        'category': category,
                        'score': float(scores[0][i]),
                        'content': dataset[int(idx)]
                    })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def format_context(self, results: List[Dict], max_results: int = 3) -> str:
        if not results:
            return "No relevant information found."
        
        context = "### Database Information:\n\n"
        
        for i, result in enumerate(results[:max_results], 1):
            cat = result['category'].replace('_', ' ').title()
            score = result['score']
            
            context += f"**Source {i}** [{cat}] (Score: {score:.3f}):\n"
            
            for key, value in list(result['content'].items())[:8]:
                if value and str(value).strip():
                    val_str = str(value)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "..."
                    context += f"  • {key}: {val_str}\n"
            
            context += "\n"
        
        return context


# Initialize
print("\n🧬 Starting Virtual CRISPR Assistant 🧬\n")

assistant = VirtualCRISPRAssistant(
    data_dir="data",
    max_rows_per_dataset=5000
)


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token,
):
    """Main response function with RAG"""
    
    if not hf_token or hf_token.strip() == "":
        yield "⚠️ Please enter your HuggingFace token.\n\nGet one at: https://huggingface.co/settings/tokens"
        return
    
    if not assistant.indices:
        yield "⚠️ No data loaded."
        return
    
    # Initialize client
    try:
        client = InferenceClient(token=hf_token, model="meta-llama/Llama-3.1-70B-Instruct")
    except Exception as e:
        yield f"⚠️ Error: {e}"
        return
    
    # Search knowledge base
    try:
        results = assistant.search(message, top_k=5)
        context = assistant.format_context(results, max_results=3)
    except:
        context = "Error retrieving context."
    
    # Enhanced system message
    enhanced_system = f"""{system_message}

{context}

Instructions:
- Use database info to answer accurately
- Cite sources (e.g., "According to Source 1...")
- Be precise and scientific
- Explain your reasoning like in the ChatGPT example"""
    
    # Build messages
    messages = [{"role": "system", "content": enhanced_system}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})
    
    # Generate response
    response = ""
    try:
        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            if chunk.choices and chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
                yield response
    except Exception as e:
        yield f"{response}\n\n⚠️ Error: {e}"


# Build interface
sources_list = "\n".join([f"        • {s}" for s in assistant.loaded_sources[:20]])
if len(assistant.loaded_sources) > 20:
    sources_list += f"\n        • ... and {len(assistant.loaded_sources) - 20} more"

demo = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(
            value="""You are an expert in CRISPR screens and gene perturbation prediction.

Your role: Predict CRISPR screen results by answering questions like:
"Does [perturbation] of [gene] in [cells] result in [phenotype]?"

Use the database to provide:
1. Yes/No answer
2. Detailed reasoning (like ChatGPT example)
3. Relevant pathway/mechanism info
4. Confidence level
5. Citations from sources

Be scientific, precise, and explain your reasoning.""",
            label="System message",
            lines=12
        ),
        gr.Slider(1, 2048, 512, step=1, label="Max tokens"),
        gr.Slider(0.1, 4.0, 0.7, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, 0.95, step=0.05, label="Top-p"),
        gr.Textbox(
            label="HuggingFace Token",
            type="password",
            placeholder="hf_...",
            info="Get token at https://huggingface.co/settings/tokens"
        ),
    ],
    title="🧬 Virtual CRISPR Assistant",
    description=f"""
    ### Predict CRISPR Screen Results with AI
    
    Based on research: **"Can LLMs Predict CRISPR Screen Results?"**
    
    **📊 {len(assistant.loaded_sources)} data sources loaded:**
{sources_list}
    
    **💡 Ask questions like:**
    - Does activation of MAP4K1 in CD4+ T cells increase TNF secretion?
    - Will knocking out BRCA1 in K562 cells result in cell death?
    - What genes regulate apoptosis in HeLa cells?
    - Predict the effect of CRISPR targeting TP53 on proliferation
    """,
    examples=[
        ["Does activation of MAP4K1 in primary CD4+ human T cells causally result in increased TNF secretion?"],
        ["Will knockout of BRCA1 in cancer cells lead to increased DNA damage sensitivity?"],
        ["What genes are essential for cell cycle progression in HeLa cells?"],
        ["Predict: Does inhibition of mTOR in neurons result in autophagy induction?"],
        ["What pathway does gene X regulate in cell type Y?"],
    ],
)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Launching Virtual CRISPR Assistant")
    print("="*60)
    print("📝 You'll need a HuggingFace token")
    print("🔗 Get one at: https://huggingface.co/settings/tokens")
    print("="*60)
    print("\n✅ Opening browser at http://localhost:7860...\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )