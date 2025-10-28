# 🧬 Virtual CRISPR Assistant

**AI-powered CRISPR screen prediction using Large Language Models + RAG**

Based on research: *"Can LLMs Predict CRISPR Screen Results?"*

---

## 🎯 What This Does

Predicts CRISPR screen results by answering questions like:

> **"Does activation of MAP4K1 in primary CD4+ T cells result in increased TNF secretion?"**

Returns:
- ✅ Yes/No answer
- 📝 Detailed reasoning
- 🔬 Pathway/mechanism explanation  
- 📊 Confidence level
- 📚 Citations from database

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt

# If you get tf-keras error:
pip install tf-keras
```

### 2. Generate Embeddings (Run Once - 30-60 min; Not Required)
```bash
python save_embeddings.py
```

This creates embeddings for all your data. 

### 3. Run the App
```bash
python app.py
```

Opens at http://localhost:7860

### 4. Use It
- Paste your HF token in the **HuggingFace Token** field (bottom of page)
- Get token at: https://huggingface.co/settings/tokens
- Ask CRISPR prediction questions!

---

## 📊 Data Sources

### Local Data (Put in `data/` folder):
- **embeddings/** - Pre-computed vectors (auto-generated)
- **summaries/** - Gene/phenotype/method/cell data
- **terms/** - Definitions
- **genomes/** - Reference sequences (mouse/human)
- **screens/** - BIOGRID-ORCS CRISPR screen data

### HuggingFace Datasets (Auto-loaded):
- yhqu/crispr_delivery (2K)
- conjuncts/ncbi-taxonomy-binomial (5K)
- jo-mengr/ncbi_cell_types_1000 (1K)
- dwb2023/crispr-binary-calls (3K)
- mlfoundations-dev/stackexchange_bioinformatics (5K)

**Total: ~150K-200K records**

---

## ⚡ Performance & Speed

### Startup Times:

| Scenario | Time |
|----------|------|
| Run | ~10 minutes |

### Query Response:
- Search: 0.5-1 sec
- LLM: 2-10 sec
- **Total: 3-15 sec**

---

## 🧪 Evaluate Model Performance

### Run Evaluation
```bash
# Terminal 1: Keep app running
python app.py

# Terminal 2: Run evaluation
python evaluate.py
```

### Options:
1. **Quick test** (5 manual examples)
2. **Full test** (100 examples from HuggingFace)
3. **Custom CSV** (your own test data)

### Metrics Calculated:
- ✅ **Accuracy** - Overall correct predictions
- ✅ **F1 Score** - Balance of precision/recall
- ✅ **Precision** - True positives / All positives
- ✅ **Recall** - True positives / All actual positives
- ✅ **AUROC** - Area under ROC curve
- ✅ **AUPRC** - Area under precision-recall curve
- ✅ **FPR** - False positive rate
- ✅ **Confusion Matrix** - TP, TN, FP, FN

### Example Results:
```
📊 EVALUATION RESULTS
============================================================

🎯 Model: meta-llama/Llama-3.1-70B-Instruct
📝 Samples: 100

📈 Performance Metrics:
  • Accuracy:  0.650
  • F1 Score:  0.470
  • Precision: 0.550
  • Recall:    0.420
  • AUROC:     0.620
  • AUPRC:     0.580
  • FPR:       0.220

🔢 Confusion Matrix:
  • True Positives:  42
  • True Negatives:  23
  • False Positives: 15
  • False Negatives: 20
```

Results saved to: `evaluation_results.json`

---

## 💡 Example Queries

### Binary Prediction Questions:
```
Does activation of MAP4K1 in primary CD4+ human T cells causally 
result in increased TNF secretion?
```

### Mechanism Questions:
```
What pathway does BRCA1 regulate in cancer cells?
```

### Prediction Questions:
```
Will knockout of TP53 in HeLa cells result in cell death?
```

### Discovery Questions:
```
What genes are essential for cell cycle progression?
```

---

## 🔬 How It Works

### RAG Pipeline:

1. **Query** → User asks: "Does gene X affect phenotype Y?"

2. **Search** → Semantic search across:
   - CRISPR screens
   - Gene databases
   - Pathway info
   - Literature (via HF datasets)

3. **Retrieve** → Top 3-5 most relevant documents

4. **Context** → Formats results for LLM

5. **Generate** → Llama3.1 70B Instruct produces:
   - Answer (Yes/No)
   - Detailed reasoning
   - Mechanism explanation
   - Citations

6. **Return** → User gets scientifically-grounded prediction

---

## 📁 Project Structure

```
VirtualCRISPR/
├── scripts/
    ├── app.py                      # Main application
    ├── save_embeddings.py          # Generate embeddings (run once)
    ├── evaluate.py                 # Test model accuracy
    ├── test_setup.py              # Setup verification
├── requirements.txt            # Dependencies
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
└── data/                      # Your data
    ├── embeddings/            # Auto-generated .npy files
    ├── summaries/             # CSV/Excel files
    ├── terms/                 # Definitions
    ├── genomes/               # TSV files
    ├── screens/               # BIOGRID-ORCS data
    └── hf_cache/              # HF dataset cache (optional)
```

---

## 🎓 Research Background

This implements the approach from:

**"Virtual CRISPR: Can LLMs Predict CRISPR Screen Results?"**

Key insights:
- LLMs can predict gene perturbation effects
- RAG improves accuracy significantly
- Binary questions (Yes/No) work best
- Reasoning/explanation is crucial
- Performance varies by model (GPT-4o: F1=0.47, FPR=0.22)

Our implementation uses:
- **Model:** Llama3.1 70B Instruct
- **RAG:** Semantic search with FAISS
- **Embeddings:** all-MiniLM-L6-v2
- **Data:** 150K+ CRISPR-related records

---

## 🛠️ For Developers

### VSCode Setup:
```bash
# Open project
cd C:\Users\Ende\VirtualCRISPR

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate

# Install
pip install -r requirements.txt

# Generate embeddings (once)
python save_embeddings.py

# Run
python app.py
```

### Configuration:

**Change memory limits** (app.py line 24):
```python
max_rows_per_dataset=5000  # Lower if memory issues
```

**Change model** (app.py line 330):
```python
model="openai/gpt-oss-120b"  # Try other models
```

**Change embedding batch size** (save_embeddings.py):
```python
batch_size=64  # Lower if GPU/memory issues
```

---

## 🐛 Troubleshooting

### "tf-keras" Error
```bash
pip install tf-keras
```

### Out of Memory
Edit `app.py` line 24:
```python
max_rows_per_dataset=2000  # Lower this
```

### Slow First Run
**Solution:** Run `save_embeddings.py` first!
```bash
python save_embeddings.py  # Takes 30-60 min once
python app.py              # Then loads in 30 sec
```

### "Please enter HuggingFace token"
- Scroll down in the interface
- Find "Additional Inputs" section
- Look for "HuggingFace Token" password field
- Paste token from https://huggingface.co/settings/tokens
- Make sure token has "Read access to repos" permission

### No Predictions / Errors
- Verify HF token is valid
- Check that data loaded (see startup logs)
- Try simpler questions first
- Lower temperature for more consistent answers

### Evaluation Script Errors
Make sure `app.py` is running in another terminal first:
```bash
# Terminal 1
python app.py

# Terminal 2
python evaluate.py
```

---

## 📊 Benchmark Comparison

### Paper Results (Various Models):

| Model | F1 Score | FPR |
|-------|----------|-----|
| Llama-2-7B | 0.58 | 0.97 |
| GPT-4o | 0.47 | 0.22 |
| GPT-4 | 0.44 | 0.38 |
| o1 | 0.16 | 0.03 |

### Your Results:
Run `evaluate.py` to get your metrics with GPT-OSS-120B!

---

## 🎯 Use Cases

### 1. Experimental Design
*"Will this perturbation have the effect I want?"*

### 2. Hypothesis Generation
*"What genes should I screen for phenotype X?"*

### 3. Literature Mining
*"What's known about gene Y in context Z?"*

### 4. Pathway Analysis
*"How does this gene regulate this process?"*

### 5. Prediction Validation
*"Does my prediction match known results?"*

---

## 📝 Files Explained

### Core Files:
- **app.py** - Main chatbot interface with Gradio
- **save_embeddings.py** - Generate embeddings once for fast loading
- **evaluate.py** - Test model accuracy (F1, AUROC, etc.)
- **requirements.txt** - All Python dependencies

### Supporting Files:
- **test_setup.py** - Verify installation
- **README.md** - This documentation
- **.gitignore** - Git ignore rules

### Generated Files:
- **data/embeddings/*.npy** - Pre-computed vectors
- **evaluation_results.json** - Evaluation metrics
- **data/hf_cache/** - Cached HuggingFace datasets

---

## 🔮 Roadmap

- [ ] Add more CRISPR screen datasets
- [ ] Fine-tune on PerturbQA dataset
- [ ] Implement confidence scoring
- [ ] Add visualization of pathways
- [ ] Support batch predictions
- [ ] Export predictions to CSV
- [ ] Add more models (Llama, Claude, etc.)
- [ ] Web API for programmatic access
- [ ] Docker container for easy deployment

---

## 🤝 Contributing

Want to add features? Found bugs?

1. Fork the repo
2. Create feature branch
3. Test thoroughly  
4. Submit PR

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Acknowledgments

- **Research:** Virtual CRISPR paper authors
- **Datasets:** BIOGRID-ORCS, HuggingFace community
- **Tools:** Gradio, Sentence Transformers, FAISS
- **Models:** OpenAI GPT-OSS-120B

---

## 📞 Support

Issues? Questions?

1. Check troubleshooting section above
2. Run `python test_setup.py`
3. Check startup logs in terminal
4. Verify data files are present
5. Open GitHub issue

---

## ✅ Complete Setup Checklist

Before first use:

- [ ] Python 3.8+ installed
- [ ] Clone/download this repo
- [ ] `pip install -r requirements.txt`
- [ ] Run `python test_setup.py` (all tests pass)
- [ ] Put data in `data/` folder (optional)
- [ ] Run `python save_embeddings.py` (once)
- [ ] HF token obtained (https://huggingface.co/settings/tokens)
- [ ] Run `python app.py`
- [ ] Paste token in interface
- [ ] Test with example query
- [ ] (Optional) Run `python evaluate.py` for metrics

---

## 🎉 Quick Commands Summary

```bash
# First time setup
pip install -r requirements.txt
python save_embeddings.py    # Takes 30-60 min, run once

# Normal usage
python app.py                # Opens at localhost:7860

# Test accuracy
python app.py                # Terminal 1: keep running
python evaluate.py           # Terminal 2: run evaluation

# Verify setup
python test_setup.py
```

---

**Built for CRISPR researchers** 🧬

*Making CRISPR screen prediction accessible through AI*

---

## 📊 Citation

If you use this in research, please cite:

```bibtex
@article{virtual_crispr_2025,
  title={Virtual CRISPR: Can LLMs Predict CRISPR Screen Results?},
  author={[Authors]},
  journal={arXiv},
  year={2025}
}
```

And mention this implementation:
```
Virtual CRISPR Assistant - https://github.com/yourusername/VirtualCRISPR
```
