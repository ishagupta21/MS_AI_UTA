# Assignment 4 Implementation Summary

## Overview
This implementation provides fact-checking capabilities for ChatGPT outputs using word overlap and textual entailment methods.

## Part 1: Word Overlap (✅ Complete - 76.47% accuracy)

### Implementation
- **Class**: `WordRecallThresholdFactChecker`
- **Method**: Jaccard similarity (intersection over union) with precision weighting
- **Threshold**: 0.25 (tuned on dev set)
- **Features**:
  - Text cleaning (removes `<s>` HTML tags)
  - Stopword removal
  - Tokenization with NLTK
  - Weighted combination of Jaccard similarity and precision

### Performance
- **Accuracy**: 76.47% (exceeds 75% requirement)
- **Precision (S)**: 75.0%
- **Recall (S)**: 80.36%
- **F1 (S)**: 77.59%
- **Precision (NS)**: 78.22%
- **Recall (NS)**: 72.48%
- **F1 (NS)**: 75.24%

### Usage
```bash
python factchecking_main.py --mode word_overlap
```

## Part 2: Textual Entailment (🔄 Ready for testing)

### Implementation
- **Class**: `EntailmentFactChecker`
- **Model**: DeBERTa-v3-base-mnli-fever-anli
- **Threshold**: 0.7 (default, can be tuned)
- **Features**:
  - NLTK sentence tokenization (with fallback)
  - Text cleaning (removes HTML tags)
  - Word overlap pruning (10% threshold) for speed
  - Max entailment score across all sentences
  - Proper memory management (gc.collect())

### Improvements Made
1. **Better Sentence Splitting**: Uses NLTK `sent_tokenize` instead of naive `.split('.')`
2. **Text Cleaning**: Removes `<s>` and `</s>` tags from passages
3. **Error Handling**: Graceful fallback if sentence tokenization fails
4. **Memory Management**: Explicit garbage collection to prevent OOM

### Usage
```bash
# CPU mode
python factchecking_main.py --mode entailment

# GPU mode (if available)
python factchecking_main.py --mode entailment --cuda
```

### Threshold Tuning
To find the optimal threshold:
```bash
python tune_entailment.py --cuda  # Use --cuda if GPU available
```

**Note**: The threshold may need adjustment to reach ≥83% accuracy. The current default is 0.7, but optimal value should be determined by running the tuning script.

## Part 3: Error Analysis (✅ Infrastructure Ready)

### Error Analysis Script
A script is provided to collect error examples:
```bash
# For entailment model
python error_analysis.py --mode entailment --cuda

# For word overlap model
python error_analysis.py --mode word_overlap
```

This generates `error_analysis.json` with:
- 10 false positive examples (predicted S, true NS)
- 10 false negative examples (predicted NS, true S)
- Statistics on error counts

### Manual Analysis Required
After running the error analysis script, you should:
1. Examine the error examples in `error_analysis.json`
2. Categorize errors into 2-4 fine-grained categories
3. Write up the analysis in `error_analysis.md` with:
   - Category definitions
   - Statistics per category
   - 3 detailed examples with explanations

## File Structure

```
a4-distrib/
├── factcheck.py              # Core fact-checking implementations
├── factchecking_main.py      # Main execution script
├── tune_threshold.py         # Threshold tuning for word overlap
├── tune_entailment.py        # Threshold tuning for entailment
├── error_analysis.py         # Error collection script
├── requirements.txt          # Python dependencies
└── data/                     # Data files
    ├── dev_labeled_ChatGPT.jsonl
    └── passages_bm25_ChatGPT_humfacts.jsonl
```

## Key Improvements Made

### Word Overlap
1. ✅ Text cleaning (HTML tag removal)
2. ✅ Improved tokenization
3. ✅ Threshold tuning (0.25 gives 76.47% accuracy)
4. ✅ Combined Jaccard + precision scoring

### Entailment
1. ✅ NLTK sentence tokenization
2. ✅ Text cleaning
3. ✅ Better error handling
4. ✅ Memory management
5. ✅ Word overlap pruning for speed
6. ⚠️ Threshold tuning needed (use `tune_entailment.py`)

## Testing Checklist

- [x] Word overlap mode works
- [x] Word overlap achieves ≥75% accuracy
- [ ] Entailment mode runs without errors
- [ ] Entailment achieves ≥83% accuracy
- [ ] Entailment runs within 10 minutes
- [ ] Error analysis script collects examples
- [ ] Error analysis writeup completed

## Next Steps

1. **Test Entailment Method**: Run the entailment mode and verify it works
2. **Tune Threshold**: Use `tune_entailment.py` to find optimal threshold for ≥83% accuracy
3. **Run Error Analysis**: Execute `error_analysis.py` to collect error examples
4. **Write Error Analysis**: Complete the written analysis in `error_analysis.md`

## Notes

- The code uses lazy imports to avoid numpy compatibility issues
- NLTK resources are downloaded automatically (with error handling)
- Memory management is implemented to prevent OOM errors
- The entailment model may take several minutes to download on first run
- GPU acceleration is optional but recommended for faster entailment inference

