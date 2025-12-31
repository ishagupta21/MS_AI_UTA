#!/usr/bin/env python3
"""Script to collect error examples for Part 3 error analysis."""

import json
from factchecking_main import read_passages, read_fact_examples
from factcheck import EntailmentFactChecker, EntailmentModel, WordRecallThresholdFactChecker
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from tqdm import tqdm

def collect_errors(fact_checker, examples, output_file="error_analysis.json"):
    """Collect false positives and false negatives from predictions."""
    gold_label_indexer = ["S", "NS"]
    
    false_positives = []  # Predicted S, but true label is NS
    false_negatives = []  # Predicted NS, but true label is S
    
    for i, example in enumerate(tqdm(examples, desc="Collecting errors")):
        converted_label = "NS" if example.label == 'IR' else example.label
        gold_label = gold_label_indexer.index(converted_label)
        
        raw_pred = fact_checker.predict(example.fact, example.passages)
        pred_label = gold_label_indexer.index(raw_pred)
        
        if gold_label != pred_label:
            error_example = {
                "index": i,
                "fact": example.fact,
                "true_label": converted_label,
                "predicted_label": raw_pred,
                "passages": [
                    {
                        "title": p.get("title", ""),
                        "text": p["text"][:500]  # Truncate for readability
                    }
                    for p in example.passages[:3]  # Only include first 3 passages
                ]
            }
            
            if pred_label == 0 and gold_label == 1:  # False positive
                false_positives.append(error_example)
            elif pred_label == 1 and gold_label == 0:  # False negative
                false_negatives.append(error_example)
    
    results = {
        "false_positives": false_positives[:10],  # Limit to 10 examples
        "false_negatives": false_negatives[:10],  # Limit to 10 examples
        "total_false_positives": len(false_positives),
        "total_false_negatives": len(false_negatives),
        "total_errors": len(false_positives) + len(false_negatives)
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nError Analysis Results:")
    print(f"Total False Positives: {len(false_positives)}")
    print(f"Total False Negatives: {len(false_negatives)}")
    print(f"Total Errors: {len(false_positives) + len(false_negatives)}")
    print(f"\nSaved {len(results['false_positives'])} false positive examples to {output_file}")
    print(f"Saved {len(results['false_negatives'])} false negative examples to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['entailment', 'word_overlap'], default='entailment',
                       help='Which model to analyze errors for')
    parser.add_argument('--labels_path', type=str, default="data/dev_labeled_ChatGPT.jsonl")
    parser.add_argument('--passages_path', type=str, default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    parser.add_argument('--output', type=str, default="error_analysis.json")
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    print("Loading data...")
    fact_to_passage_dict = read_passages(args.passages_path)
    examples = read_fact_examples(args.labels_path, fact_to_passage_dict)
    print(f"Loaded {len(examples)} examples")
    
    if args.mode == 'entailment':
        print("Loading entailment model...")
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        ent_tokenizer = AutoTokenizer.from_pretrained(model_name)
        roberta_ent_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if args.cuda:
            roberta_ent_model.to('cuda')
        ent_model = EntailmentModel(roberta_ent_model, ent_tokenizer, args.cuda)
        fact_checker = EntailmentFactChecker(ent_model, threshold=0.7, overlap_threshold=0.1)
    else:
        print("Using word overlap model...")
        fact_checker = WordRecallThresholdFactChecker(threshold=0.25)
    
    print(f"Collecting errors for {args.mode} model...")
    collect_errors(fact_checker, examples, args.output)

if __name__ == "__main__":
    main()

