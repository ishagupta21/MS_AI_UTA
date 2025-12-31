#!/usr/bin/env python3
"""Script to tune threshold for entailment fact checker."""

import json
from factchecking_main import read_passages, read_fact_examples
from factcheck import EntailmentFactChecker, EntailmentModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

def evaluate_threshold(fact_checker, examples, threshold_name="threshold"):
    """Evaluate fact checker on examples and return accuracy."""
    from tqdm import tqdm
    correct = 0
    total = 0
    confusion_mat = [[0, 0], [0, 0]]
    gold_label_indexer = ["S", "NS"]
    
    for example in tqdm(examples, desc="Evaluating"):
        converted_label = "NS" if example.label == 'IR' else example.label
        gold_label = gold_label_indexer.index(converted_label)
        
        raw_pred = fact_checker.predict(example.fact, example.passages)
        pred_label = gold_label_indexer.index(raw_pred)
        
        confusion_mat[gold_label][pred_label] += 1
        if gold_label == pred_label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, confusion_mat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_path', type=str, default="data/dev_labeled_ChatGPT.jsonl")
    parser.add_argument('--passages_path', type=str, default="data/passages_bm25_ChatGPT_humfacts.jsonl")
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--thresholds', type=str, default="0.5,0.6,0.7,0.75,0.8,0.85,0.9", help='Comma-separated list of thresholds to test')
    args = parser.parse_args()
    
    print("Loading data...")
    fact_to_passage_dict = read_passages(args.passages_path)
    examples = read_fact_examples(args.labels_path, fact_to_passage_dict)
    print(f"Loaded {len(examples)} examples")
    
    print("Loading entailment model...")
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    ent_tokenizer = AutoTokenizer.from_pretrained(model_name)
    roberta_ent_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if args.cuda:
        roberta_ent_model.to('cuda')
    ent_model = EntailmentModel(roberta_ent_model, ent_tokenizer, args.cuda)
    
    thresholds = [float(t) for t in args.thresholds.split(',')]
    print(f"\nTuning EntailmentFactChecker with thresholds: {thresholds}")
    
    best_accuracy = 0
    best_threshold = 0
    best_confusion = None
    
    for threshold in thresholds:
        print(f"\nTesting threshold {threshold:.2f}...")
        fact_checker = EntailmentFactChecker(ent_model, threshold=threshold, overlap_threshold=0.1)
        accuracy, confusion = evaluate_threshold(fact_checker, examples)
        print(f"Threshold {threshold:.2f}: Accuracy = {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_confusion = confusion
    
    print(f"\nBest threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.4f}")
    if best_confusion:
        print("Confusion matrix:")
        print(f"  True S  -> Pred S: {best_confusion[0][0]}, Pred NS: {best_confusion[0][1]}")
        print(f"  True NS -> Pred S: {best_confusion[1][0]}, Pred NS: {best_confusion[1][1]}")

if __name__ == "__main__":
    main()

