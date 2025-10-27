"""
Virtual CRISPR Evaluation - Compare predictions to real experimental results
Load questions with known answers from literature/databases and test accuracy
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from tqdm import tqdm
import json
from pathlib import Path
from huggingface_hub import InferenceClient
import time


class VirtualCRISPREvaluator:
    """Evaluate predictions against real experimental results"""
    
    def __init__(self, assistant, hf_token: str, model: str = "meta-llama/Llama-3.1-70B-Instruct"):
        self.assistant = assistant
        self.client = InferenceClient(token=hf_token)
        self.model_name = model
        
    def parse_prediction(self, response: str) -> int:
        """
        Extract Yes/No from LLM response
        Returns: 1 for Yes, 0 for No
        """
        response_lower = response.lower()
        
        # Look for explicit answers
        if "short answer: yes" in response_lower or "answer: yes" in response_lower:
            return 1
        if "short answer: no" in response_lower or "answer: no" in response_lower:
            return 0
        
        # Look in first sentence
        first_sentence = response_lower.split('.')[0]
        
        if 'yes' in first_sentence and 'no' not in first_sentence:
            return 1
        if 'no' in first_sentence and 'yes' not in first_sentence:
            return 0
        
        # Count occurrences in first 200 chars
        first_part = response_lower[:200]
        yes_count = first_part.count('yes')
        no_count = first_part.count('no')
        
        if yes_count > no_count:
            return 1
        elif no_count > yes_count:
            return 0
        
        # Default to No if unclear
        return 0
    
    def predict_single(self, question: str, max_tokens: int = 512, temperature: float = 0.7) -> dict:
        """Make a single prediction with RAG"""
        
        # Search knowledge base
        try:
            results = self.assistant.search(question, top_k=5)
            context = self.assistant.format_context(results, max_results=3)
        except:
            context = "No relevant information found."
        
        # Build prompt
        system_message = f"""You are an expert in CRISPR screens and gene perturbation prediction.

{context}

Instructions:
- Answer with "Short answer: yes" or "Short answer: no" FIRST
- Then provide detailed reasoning
- Cite sources from the database
- Be scientific and precise"""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        
        # Generate response
        response = ""
        try:
            output = self.client.chat_completion(
                messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                stream=False
            )
            
            if output and output.choices:
                response = output.choices[0].message.content
                
        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠️ Error: {error_msg[:100]}")
            
            # Retry once on rate limit
            if "429" in error_msg or "rate" in error_msg.lower():
                print("  Waiting 10 seconds...")
                time.sleep(10)
                try:
                    output = self.client.chat_completion(
                        messages,
                        model=self.model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False
                    )
                    if output and output.choices:
                        response = output.choices[0].message.content
                except:
                    pass
            
            return {"response": response, "prediction": 0, "error": error_msg}
        
        prediction = self.parse_prediction(response)
        
        return {
            "response": response,
            "prediction": prediction,
            "error": None
        }
    
    def evaluate_dataset(self, test_data: pd.DataFrame, output_file: str = "evaluation_results.json"):
        """
        Evaluate predictions against known experimental results
        
        Expected columns in test_data:
        - question: Binary CRISPR question
        - ground_truth: Known experimental result (1=Yes, 0=No)
        - source: Where the answer came from (optional)
        - gene, cell_type, phenotype: Optional metadata
        """
        
        print(f"\n🧪 Evaluating on {len(test_data)} questions with known answers")
        print(f"📝 Model: {self.model_name}")
        print("="*60)
        
        predictions = []
        ground_truths = []
        results_list = []
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Testing"):
            question = row['question']
            ground_truth = int(row['ground_truth'])
            source = row.get('source', 'Unknown')
            
            # Make prediction
            result = self.predict_single(question)
            
            predictions.append(result['prediction'])
            ground_truths.append(ground_truth)
            
            # Store detailed result
            results_list.append({
                'question': question,
                'ground_truth': ground_truth,
                'prediction': result['prediction'],
                'correct': result['prediction'] == ground_truth,
                'response': result['response'],
                'source': source,
                'error': result['error'],
                'gene': row.get('gene', ''),
                'cell_type': row.get('cell_type', ''),
                'phenotype': row.get('phenotype', '')
            })
            
            # Rate limiting
            time.sleep(2)
        
        # Calculate metrics
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        
        accuracy = accuracy_score(ground_truths, predictions)
        
        # Check class distribution
        unique_true = np.unique(ground_truths)
        unique_pred = np.unique(predictions)
        
        print(f"\n📊 Data distribution:")
        print(f"  Ground truth: Yes={sum(ground_truths)}, No={len(ground_truths)-sum(ground_truths)}")
        print(f"  Predictions:  Yes={sum(predictions)}, No={len(predictions)-sum(predictions)}")
        
        # Calculate metrics with error handling
        try:
            f1 = f1_score(ground_truths, predictions, zero_division=0)
        except:
            f1 = 0.0
            
        try:
            precision = precision_score(ground_truths, predictions, zero_division=0)
        except:
            precision = 0.0
            
        try:
            recall = recall_score(ground_truths, predictions, zero_division=0)
        except:
            recall = 0.0
        
        # AUROC and AUPRC
        auroc = None
        auprc = None
        if len(unique_true) > 1 and len(unique_pred) > 1:
            try:
                auroc = roc_auc_score(ground_truths, predictions)
                auprc = average_precision_score(ground_truths, predictions)
            except:
                pass
        
        # Confusion matrix
        try:
            cm = confusion_matrix(ground_truths, predictions, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        except:
            # Handle edge cases
            tp = sum((ground_truths == 1) & (predictions == 1))
            tn = sum((ground_truths == 0) & (predictions == 0))
            fp = sum((ground_truths == 0) & (predictions == 1))
            fn = sum((ground_truths == 1) & (predictions == 0))
        
        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Save results
        results = {
            'model': self.model_name,
            'num_samples': len(test_data),
            'timestamp': pd.Timestamp.now().isoformat(),
            'metrics': {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'precision': float(precision),
                'recall': float(recall),
                'auroc': float(auroc) if auroc else None,
                'auprc': float(auprc) if auprc else None,
                'fpr': float(fpr),
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            },
            'detailed_results': results_list
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self.print_results(results)
        
        # Show some examples
        self.show_examples(results_list)
        
        return results
    
    def print_results(self, results: dict):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        
        metrics = results['metrics']
        
        print(f"\n🎯 Model: {results['model']}")
        print(f"📝 Questions tested: {results['num_samples']}")
        
        print(f"\n📈 Performance vs Real Results:")
        print(f"  • Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  • F1 Score:  {metrics['f1_score']:.3f}")
        print(f"  • Precision: {metrics['precision']:.3f}")
        print(f"  • Recall:    {metrics['recall']:.3f}")
        
        if metrics['auroc']:
            print(f"  • AUROC:     {metrics['auroc']:.3f}")
        if metrics['auprc']:
            print(f"  • AUPRC:     {metrics['auprc']:.3f}")
        
        print(f"  • FPR:       {metrics['fpr']:.3f}")
        
        print(f"\n🔢 Confusion Matrix:")
        print(f"  • True Positives:  {metrics['true_positives']} (correctly predicted Yes)")
        print(f"  • True Negatives:  {metrics['true_negatives']} (correctly predicted No)")
        print(f"  • False Positives: {metrics['false_positives']} (predicted Yes, actually No)")
        print(f"  • False Negatives: {metrics['false_negatives']} (predicted No, actually Yes)")
        
        print("\n" + "="*60)
    
    def show_examples(self, results_list: list, n: int = 3):
        """Show example predictions"""
        print(f"\n📝 Example Predictions:")
        print("-"*60)
        
        # Show correct predictions
        correct = [r for r in results_list if r['correct']]
        if correct:
            print(f"\n✅ Correct Predictions ({len(correct)} total):")
            for r in correct[:n]:
                print(f"\nQ: {r['question']}")
                print(f"Real Answer: {'Yes' if r['ground_truth'] else 'No'}")
                print(f"AI Predicted: {'Yes' if r['prediction'] else 'No'} ✓")
                if r['source']:
                    print(f"Source: {r['source']}")
        
        # Show incorrect predictions
        incorrect = [r for r in results_list if not r['correct']]
        if incorrect:
            print(f"\n❌ Incorrect Predictions ({len(incorrect)} total):")
            for r in incorrect[:n]:
                print(f"\nQ: {r['question']}")
                print(f"Real Answer: {'Yes' if r['ground_truth'] else 'No'}")
                print(f"AI Predicted: {'Yes' if r['prediction'] else 'No'} ✗")
                if r['source']:
                    print(f"Source: {r['source']}")


def load_test_data_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load test data from CSV with known answers
    
    CSV format:
    question, ground_truth, source, gene, cell_type, phenotype
    
    Example:
    "Does knockout of TP53 result in decreased apoptosis?",1,"PMID:12345678","TP53","HeLa","apoptosis"
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if 'question' not in df.columns or 'ground_truth' not in df.columns:
        raise ValueError("CSV must have 'question' and 'ground_truth' columns")
    
    print(f"✅ Loaded {len(df)} test cases from {csv_path}")
    return df


def create_example_test_data():
    """Create example test dataset with known answers from literature"""
    
    test_cases = [
        {
            'question': 'Does activation of MAP4K1 in primary CD4+ human T cells result in increased TNF secretion?',
            'ground_truth': 0,  # No - MAP4K1 is negative regulator
            'source': 'Virtual CRISPR paper',
            'gene': 'MAP4K1',
            'cell_type': 'CD4+ T cells',
            'phenotype': 'increased TNF'
        },
        {
            'question': 'Does knockout of TP53 in cancer cells result in decreased apoptosis?',
            'ground_truth': 1,  # Yes - TP53 promotes apoptosis
            'source': 'Well-established',
            'gene': 'TP53',
            'cell_type': 'cancer cells',
            'phenotype': 'decreased apoptosis'
        },
        {
            'question': 'Does knockout of BRCA1 result in increased DNA damage sensitivity?',
            'ground_truth': 1,  # Yes - BRCA1 is DNA repair
            'source': 'Well-established',
            'gene': 'BRCA1',
            'cell_type': 'cells',
            'phenotype': 'DNA damage sensitivity'
        },
        {
            'question': 'Does overexpression of MYC result in increased cell proliferation?',
            'ground_truth': 1,  # Yes - MYC drives proliferation
            'source': 'Well-established',
            'gene': 'MYC',
            'cell_type': 'cells',
            'phenotype': 'proliferation'
        },
        {
            'question': 'Does knockout of PTEN in cells result in decreased cell growth?',
            'ground_truth': 0,  # No - PTEN loss increases growth
            'source': 'Well-established',
            'gene': 'PTEN',
            'cell_type': 'cells',
            'phenotype': 'decreased growth'
        },
        {
            'question': 'Does knockout of GAPDH in cells result in cell death?',
            'ground_truth': 1,  # Yes - essential gene
            'source': 'Well-established',
            'gene': 'GAPDH',
            'cell_type': 'cells',
            'phenotype': 'cell death'
        },
    ]
    
    return pd.DataFrame(test_cases)


def create_csv_template():
    """Create a template CSV for users to fill in"""
    template = pd.DataFrame({
        'question': [
            'Does knockout of [GENE] in [CELL_TYPE] result in [PHENOTYPE]?',
            'Does activation of [GENE] in [CELL_TYPE] result in [PHENOTYPE]?'
        ],
        'ground_truth': [1, 0],  # 1=Yes, 0=No
        'source': ['PMID:xxxxx or DOI or paper citation', 'DepMap or database name'],
        'gene': ['GENE_NAME', 'GENE_NAME'],
        'cell_type': ['CELL_TYPE', 'CELL_TYPE'],
        'phenotype': ['PHENOTYPE_DESCRIPTION', 'PHENOTYPE_DESCRIPTION']
    })
    
    template.to_csv('test_template.csv', index=False)
    print("✅ Created test_template.csv")
    print("📝 Fill this in with your questions and known answers from literature/databases")
    return template


# Main execution
if __name__ == "__main__":
    print("🧬 Virtual CRISPR Evaluation")
    print("Compare AI predictions to real experimental results")
    print("="*60)
    
    # Get HF token
    hf_token = input("\nEnter your HuggingFace token: ").strip()
    
    if not hf_token:
        print("❌ Token required!")
        exit(1)
    
    # Load assistant
    print("\n📥 Loading Virtual CRISPR Assistant...")
    from app import assistant
    
    # Choose model
    print("\n🤖 Choose model:")
    print("1. Llama-3.1-70B-Instruct (FREE, BEST quality)")
    print("2. Llama-3.2-3B-Instruct (FREE, faster)")
    print("3. Custom model")
    
    model_choice = input("\nChoice (1/2/3): ").strip() or "1"
    
    if model_choice == "1":
        model = "meta-llama/Llama-3.1-70B-Instruct"
    elif model_choice == "2":
        model = "meta-llama/Llama-3.2-3B-Instruct"
    elif model_choice == "3":
        model = input("Enter model name: ").strip()
    else:
        model = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Create evaluator
    evaluator = VirtualCRISPREvaluator(
        assistant=assistant,
        hf_token=hf_token,
        model=model
    )
    
    # Choose test data source
    print("\n📊 Choose test data:")
    print("1. Example questions with known answers (6 examples)")
    print("2. Load from CSV file (your questions + real answers)")
    print("3. Create CSV template to fill in")
    
    choice = input("\nChoice (1/2/3): ").strip() or "1"
    
    if choice == "1":
        test_data = create_example_test_data()
    elif choice == "2":
        csv_path = input("Enter CSV path: ").strip()
        test_data = load_test_data_from_csv(csv_path)
    elif choice == "3":
        create_csv_template()
        print("\n📝 Next steps:")
        print("1. Open test_template.csv")
        print("2. Add your questions with known answers from papers/databases")
        print("3. Run this script again and choose option 2")
        exit(0)
    else:
        test_data = create_example_test_data()
    
    if test_data is None or len(test_data) == 0:
        print("❌ No test data!")
        exit(1)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_data, output_file="evaluation_results.json")
    
    print("\n✅ Evaluation complete!")
    print("📄 Full results saved to: evaluation_results.json")