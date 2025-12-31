# Submission Information

## Single File Submission

As per the assignment requirements, **only one file needs to be uploaded to Gradescope**: `factchecking_main.py`

This file contains:
- All fact-checking classes (from the original `factcheck.py`)
- All main execution code (from the original `factchecking_main.py`)
- Complete implementation for Part 1 (Word Overlap) and Part 2 (Entailment)

## Verification

Before submitting, verify that both commands work:

```bash
python factchecking_main.py --mode word_overlap
python factchecking_main.py --mode entailment
```

## File Structure for Submission

**Required file:**
- `factchecking_main.py` - Single combined file with all implementations

**Not required for submission (but useful for reference):**
- `factcheck.py` - Original separate file (now combined into main)
- `tune_threshold.py` - Utility script for threshold tuning
- `tune_entailment.py` - Utility script for threshold tuning
- `error_analysis.py` - Utility script for error analysis
- `IMPLEMENTATION_SUMMARY.md` - Documentation
- `error_analysis.md` - Error analysis writeup (for Part 3)

## Part 1: Word Overlap ✅
- **Status**: Complete
- **Accuracy**: 76.47% (exceeds 75% requirement)
- **Implementation**: `WordRecallThresholdFactChecker` class
- **Threshold**: 0.25 (tuned)

## Part 2: Entailment 🔄
- **Status**: Ready for testing
- **Implementation**: `EntailmentFactChecker` class
- **Threshold**: 0.7 (default, may need tuning)
- **Note**: Test to ensure ≥83% accuracy and <10 minute runtime

## Part 3: Error Analysis 📝
- **Status**: Infrastructure ready
- **Required**: Written submission (separate from code)
- **Use**: `error_analysis.py` to collect examples (optional utility)

## Important Notes

1. The single file `factchecking_main.py` is self-contained and does not require `factcheck.py`
2. All imports are handled within the file
3. NLTK resources are downloaded automatically
4. The file works independently without any other Python files

