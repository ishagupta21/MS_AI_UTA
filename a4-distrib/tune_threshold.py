#!/usr/bin/env python3
"""Script to tune threshold for word overlap fact checker."""

import json
from factchecking_main import read_passages, read_fact_examples
from factcheck import WordRecallThresholdFactChecker, WordOverlapFactChecker
import argparse

def evaluate_threshold(fact_checker, examples, threshold_name="threshold"):
    """Evaluate fact checker on examples and return accuracy."""
    correct = 0
    total = 0
    confusion_mat = [[0, 0], [0, 0]]
    gold_label_indexer = ["S", "NS"]
    
    for example in examples:
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
    parser.add_argument('--method', type=str, choices=['jaccard', 'tfidf'], default='jaccard')
    args = parser.parse_args()
    
    fact_to_passage_dict = read_passages(args.passages_path)
    examples = read_fact_examples(args.labels_path, fact_to_passage_dict)
    print(f"Loaded {len(examples)} examples")
    
    if args.method == 'jaccard':
        print("\nTuning WordRecallThresholdFactChecker (Jaccard similarity):")
        best_accuracy = 0
        best_threshold = 0
        best_confusion = None
        
        # Test a range of thresholds
        for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            fact_checker = WordRecallThresholdFactChecker(threshold=threshold)
            accuracy, confusion = evaluate_threshold(fact_checker, examples)
            print(f"Threshold {threshold:.2f}: Accuracy = {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                best_confusion = confusion
        
        print(f"\nBest threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.4f}")
        print("Confusion matrix:")
        print(f"  True S  -> Pred S: {best_confusion[0][0]}, Pred NS: {best_confusion[0][1]}")
        print(f"  True NS -> Pred S: {best_confusion[1][0]}, Pred NS: {best_confusion[1][1]}")
    
    elif args.method == 'tfidf':
        print("\nTuning WordOverlapFactChecker (TF-IDF cosine similarity):")
        best_accuracy = 0
        best_threshold = 0
        best_confusion = None
        
        # Test a range of thresholds
        for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            try:
                fact_checker = WordOverlapFactChecker(threshold=threshold)
                accuracy, confusion = evaluate_threshold(fact_checker, examples)
                print(f"Threshold {threshold:.2f}: Accuracy = {accuracy:.4f}")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
                    best_confusion = confusion
            except Exception as e:
                print(f"Threshold {threshold:.2f}: Error - {e}")
        
        print(f"\nBest threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.4f}")
        if best_confusion:
            print("Confusion matrix:")
            print(f"  True S  -> Pred S: {best_confusion[0][0]}, Pred NS: {best_confusion[0][1]}")
            print(f"  True NS -> Pred S: {best_confusion[1][0]}, Pred NS: {best_confusion[1][1]}")

if __name__ == "__main__":
    main()

