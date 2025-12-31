# factchecking_main.py
# Combined file for Assignment 4 submission

import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
from tqdm import tqdm
import argparse
import torch
import numpy as np
import gc

# Lazy imports for NLTK to avoid numpy compatibility issues
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        """
        Check if hypothesis is entailed by premise.
        Returns entailment score (higher = more likely to be entailed).
        For MNLI models, typically: [entailment, neutral, contradiction]
        """
        with torch.no_grad():
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True, max_length=512)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            # For MNLI models: [entailment, neutral, contradiction]
            # We return the entailment probability
            # Alternative: could use (entailment - contradiction) or entailment / (entailment + contradiction)
            entailment_prob = probabilities[0]
            contradiction_prob = probabilities[2] if len(probabilities) > 2 else 0
            
            # Use a score that favors high entailment and low contradiction
            # This helps distinguish between neutral and supported cases
            score = entailment_prob  # Simple: just use entailment probability
        
        del inputs, outputs, logits
        gc.collect()

        return score
    
    def check_entailment_batch(self, premises: List[str], hypothesis: str, batch_size=None):
        """
        Check entailment for multiple premises at once (batched for speed).
        Returns list of entailment scores.
        """
        if batch_size is None:
            batch_size = len(premises)  # Process all at once if batch_size not specified
        
        scores = []
        for i in range(0, len(premises), batch_size):
            batch_premises = premises[i:i+batch_size]
            with torch.no_grad():
                # Tokenize the batch
                inputs = self.tokenizer(
                    batch_premises, 
                    [hypothesis] * len(batch_premises),
                    return_tensors='pt', 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                if self.cuda:
                    inputs = {key: value.to('cuda') for key, value in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                # Get entailment probability (first column)
                batch_scores = probabilities[:, 0]
                scores.extend(batch_scores.tolist())
            
            del inputs, outputs, logits
        
        gc.collect()
        return scores


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold=0.25):
        self.threshold = threshold
        self.stop_words = set(stopwords.words('english'))
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing HTML-like tags and extra whitespace."""
        # Remove <s> and </s> tags
        text = text.replace('<s>', ' ').replace('</s>', ' ')
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _tokenize(self, text: str) -> set:
        """Tokenize text and return set of meaningful tokens."""
        text = self._clean_text(text)
        tokens = [word.lower() for word in word_tokenize(text) 
                 if word.isalnum() and word.lower() not in self.stop_words]
        return set(tokens)

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = self._tokenize(fact)
        if len(fact_tokens) == 0:
            return "NS"
        
        max_overlap = 0
        for passage in passages:
            passage_tokens = self._tokenize(passage['text'])
            if len(passage_tokens) == 0:
                continue
            
            # Use Jaccard similarity (intersection over union)
            intersection = len(fact_tokens & passage_tokens)
            union = len(fact_tokens | passage_tokens)
            if union > 0:
                overlap = intersection / union
                max_overlap = max(max_overlap, overlap)
            
            # Also try precision (recall of fact words in passage)
            # This helps when passages are much longer than facts
            if len(fact_tokens) > 0:
                precision = intersection / len(fact_tokens)
                # Use weighted combination: favor precision slightly for short facts
                combined_score = 0.6 * overlap + 0.4 * precision if union > 0 else precision
                max_overlap = max(max_overlap, combined_score)

        return "S" if max_overlap >= self.threshold else "NS"

class WordOverlapFactChecker(FactChecker):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        # Lazy import to avoid numpy compatibility issues when not used
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def predict(self, fact: str, passages: List[dict]) -> str:
        from sklearn.metrics.pairwise import cosine_similarity
        corpus = [fact] + [passage['text'] for passage in passages]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        fact_vector = tfidf_matrix[0]
        passage_vectors = tfidf_matrix[1:]

        max_similarity = max(cosine_similarity(fact_vector, passage_vectors)[0])
        return "S" if max_similarity >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model, threshold=0.7, overlap_threshold=0.15, max_sentences_per_passage=5, max_passages=10, batch_size=8, early_stop_threshold=0.95):
        self.ent_model = ent_model
        self.threshold = threshold
        self.overlap_threshold = overlap_threshold  # Increased from 0.1 to 0.15 for more aggressive pruning
        self.max_sentences_per_passage = max_sentences_per_passage  # Limit sentences per passage
        self.max_passages = max_passages  # Limit number of passages to process
        self.batch_size = batch_size  # Batch size for model inference
        self.early_stop_threshold = early_stop_threshold  # Stop early if score is very high
        self.stop_words = set(stopwords.words('english'))
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing HTML-like tags and extra whitespace."""
        # Remove <s> and </s> tags
        text = text.replace('<s>', ' ').replace('</s>', ' ')
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _split_sentences_fast(self, text: str) -> List[str]:
        """Fast sentence splitting using simple heuristics."""
        text = self._clean_text(text)
        # Simple splitting by periods, but keep sentences that are reasonable length
        sentences = text.split('.')
        # Filter and clean sentences
        cleaned = []
        for s in sentences:
            s = s.strip()
            # Filter: length between 20 and 400 characters (reasonable sentence length)
            if 20 <= len(s) <= 400 and len(s.split()) >= 5:
                cleaned.append(s)
        return cleaned
    
    def _score_sentence_relevance(self, sentence: str, fact_tokens: set) -> float:
        """Score how relevant a sentence is to the fact (for ranking)."""
        sentence_lower = sentence.lower()
        sentence_tokens = set(word_tokenize(sentence_lower)) - self.stop_words
        if len(fact_tokens) == 0:
            return 0.0
        # Use Jaccard similarity as relevance score
        intersection = len(fact_tokens & sentence_tokens)
        union = len(fact_tokens | sentence_tokens)
        return intersection / union if union > 0 else 0.0

    def predict(self, fact: str, passages: List[dict]) -> str:
        # Clean fact text
        fact = self._clean_text(fact)
        
        # Tokenize fact once
        fact_tokens = set(word_tokenize(fact.lower())) - self.stop_words
        if len(fact_tokens) == 0:
            return "NS"
        
        # Step 1: Prune passages based on word overlap (more aggressive)
        passage_scores = []
        for passage in passages:
            passage_text = self._clean_text(passage['text'])
            passage_tokens = set(word_tokenize(passage_text.lower())) - self.stop_words
            overlap_ratio = len(fact_tokens & passage_tokens) / len(fact_tokens) if len(fact_tokens) > 0 else 0
            if overlap_ratio >= self.overlap_threshold:
                passage_scores.append((overlap_ratio, passage_text))
        
        if len(passage_scores) == 0:
            return "NS"
        
        # Step 2: Sort passages by overlap and take top N
        passage_scores.sort(key=lambda x: x[0], reverse=True)
        filtered_passages = [text for _, text in passage_scores[:self.max_passages]]
        
        # Step 3: Extract and rank sentences from filtered passages
        all_sentences = []
        for passage_text in filtered_passages:
            sentences = self._split_sentences_fast(passage_text)
            # Score each sentence by relevance to fact
            sentence_scores = [(self._score_sentence_relevance(s, fact_tokens), s) for s in sentences]
            # Sort by relevance and take top N sentences per passage
            sentence_scores.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [s for _, s in sentence_scores[:self.max_sentences_per_passage]]
            all_sentences.extend(top_sentences)
        
        if len(all_sentences) == 0:
            return "NS"
        
        # Step 4: Batch process sentences for entailment
        max_score = float('-inf')
        
        # Process in batches
        for i in range(0, len(all_sentences), self.batch_size):
            batch_sentences = all_sentences[i:i+self.batch_size]
            try:
                # Batch inference
                if hasattr(self.ent_model, 'check_entailment_batch'):
                    scores = self.ent_model.check_entailment_batch(batch_sentences, fact, batch_size=len(batch_sentences))
                else:
                    # Fallback to individual processing if batch method not available
                    scores = [self.ent_model.check_entailment(s, fact) for s in batch_sentences]
                
                batch_max = max(scores) if scores else float('-inf')
                max_score = max(max_score, batch_max)
                
                # Early stopping: if we find a very high score, we can stop
                if max_score >= self.early_stop_threshold:
                    break
                    
            except Exception as e:
                # Skip batch if error occurs
                continue

        return "S" if max_score >= self.threshold else "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        import spacy
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help="choose from [random', 'always_entail', 'word_overlap', 'parsing', 'entailment]")
    # parser.add_argument('--labels_path', type=str, default="data/labeled_ChatGPT.jsonl", help="path to the labels")
    parser.add_argument('--labels_path', type=str, default="data/dev_labeled_ChatGPT.jsonl", help="path to the labels")
    parser.add_argument('--passages_path', type=str, default="data/passages_bm25_ChatGPT_humfacts.jsonl", help="path to the passages retrieved for the ChatGPT human-labeled facts")
    parser.add_argument('--cuda', dest='cuda', default=False, action='store_true', help='skip printing output on the test set')
    args = parser.parse_args()
    return args


def read_passages(path: str):
    """
    Reads the retrieved passages and puts then in a dictionary mapping facts to passages.
    :param path: path to the cached passages
    :return: dict mapping facts (strings) to passages
    """
    fact_to_passage_dict = {}
    with open(path, 'r') as file:
        all_lines = file.readlines()
        for nextline in all_lines:
            dict = json.loads(nextline)
            name = dict["name"]
            for passage in dict["passages"]:
                if passage["title"] != name:
                    raise Exception("Couldn't find a match for: " + name + " " + passage["title"])
            fact_to_passage_dict[dict["sent"]] = dict["passages"]
    return fact_to_passage_dict


def read_fact_examples(labeled_facts_path: str, fact_to_passage_dict: Dict):
    """
    Reads the labeled fact examples and constructs FactExample objects associating labeled, human-annotated facts
    with their corresponding passages
    :param labeled_facts_path: path to the list of labeled
    :param fact_to_passage_dict: the dict mapping facts to passages (see load_passages)
    :return: a list of FactExample objects to use as our dataset
    """
    examples = []
    with open(labeled_facts_path, 'r') as file:
        all_lines = file.readlines()
        for nextline in all_lines:
            dict = json.loads(nextline)
            if dict["annotations"] is not None:
                for sent in dict["annotations"]:
                    if sent["human-atomic-facts"] is None:
                        # Should never be the case, but just in case
                        print("No facts! Skipping this one: " + repr(sent))
                    else:
                        for fact in sent["human-atomic-facts"]:
                            if fact["text"] not in fact_to_passage_dict:
                                # Should never be the case, but just in case
                                print("Missing fact: " + fact["text"])
                            else:
                                examples.append(FactExample(fact["text"], fact_to_passage_dict[fact["text"]], fact["label"]))
    return examples


def predict_two_classes(examples: List[FactExample], fact_checker):
    """
    Compares against ground truth which is just the labels S and NS (IR is mapped to NS).
    Makes predictions and prints evaluation statistics on this setting.
    :param examples: a list of FactExample objects
    :param fact_checker: the FactChecker object to use for prediction
    """
    gold_label_indexer = ["S", "NS"]
    confusion_mat = [[0, 0], [0, 0]]
    ex_count = 0

    for i, example in enumerate(tqdm(examples)):
        converted_label = "NS" if example.label == 'IR' else example.label
        gold_label = gold_label_indexer.index(converted_label)

        raw_pred = fact_checker.predict(example.fact, example.passages)
        pred_label = gold_label_indexer.index(raw_pred)

        confusion_mat[gold_label][pred_label] += 1
        ex_count += 1
    print_eval_stats(confusion_mat, gold_label_indexer)


def print_eval_stats(confusion_mat, gold_label_indexer):
    """
    Takes a confusion matrix and the label indexer and prints accuracy and per-class F1
    :param confusion_mat: The confusion matrix, indexed as [gold_label, pred_label]
    :param gold_label_indexer: The Indexer for the labels as a List, not an Indexer
    """
    for row in confusion_mat:
        print("\t".join([repr(item) for item in row]))
    correct_preds = sum([confusion_mat[i][i] for i in range(0, len(gold_label_indexer))])
    total_preds = sum([confusion_mat[i][j] for i in range(0, len(gold_label_indexer)) for j in range(0, len(gold_label_indexer))])
    print("Accuracy: " + repr(correct_preds) + "/" + repr(total_preds) + " = " + repr(correct_preds/total_preds))
    for idx in range(0, len(gold_label_indexer)):
        num_correct = confusion_mat[idx][idx]
        num_gold = sum([confusion_mat[idx][i] for i in range(0, len(gold_label_indexer))])
        num_pred = sum([confusion_mat[i][idx] for i in range(0, len(gold_label_indexer))])
        rec = num_correct / num_gold
        if num_pred > 0:
            prec = num_correct / num_pred
            f1 = 2 * prec * rec/(prec + rec)
        else:
            prec = "undefined"
            f1 = "undefined"
        print("Prec for " + gold_label_indexer[idx] + ": " + repr(num_correct) + "/" + repr(num_pred) + " = " + repr(prec))
        print("Rec for " + gold_label_indexer[idx] + ": " + repr(num_correct) + "/" + repr(num_gold) + " = " + repr(rec))
        print("F1 for " + gold_label_indexer[idx] + ": " + repr(f1))


if __name__=="__main__":
    args = _parse_args()
    print(args)

    fact_to_passage_dict = read_passages(args.passages_path)

    examples = read_fact_examples(args.labels_path, fact_to_passage_dict)
    print("Read " + repr(len(examples)) + " examples")
    print("Fact and length of passages for each fact:")
    for example in examples:
        print(example.fact + ": " + repr([len(p["text"]) for p in example.passages]))

    assert args.mode in ['random', 'always_entail', 'word_overlap', 'parsing', 'entailment'], "invalid method"
    print(f"Method: {args.mode}")

    fact_checker = None
    if args.mode == "random":
        fact_checker = RandomGuessFactChecker()
    elif args.mode == "always_entail":
        fact_checker = AlwaysEntailedFactChecker()
    elif args.mode == "word_overlap":
        # Using improved WordRecallThresholdFactChecker with tuned threshold
        # Threshold tuned to optimize accuracy on dev set: 0.25 gives 76.47% accuracy
        fact_checker = WordRecallThresholdFactChecker(threshold=0.25)
    elif args.mode == "parsing":
        fact_checker = DependencyRecallThresholdFactChecker()
    elif args.mode == "entailment":
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        # model_name = "roberta-large-mnli"   # alternative model that you can try out if you want
        ent_tokenizer = AutoTokenizer.from_pretrained(model_name)
        roberta_ent_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if args.cuda:
            roberta_ent_model.to('cuda')
        ent_model = EntailmentModel(roberta_ent_model, ent_tokenizer, args.cuda)
        # Optimized settings for speed:
        # - Higher overlap_threshold (0.15) for more aggressive passage pruning
        # - Limit to 10 passages max
        # - Limit to 5 sentences per passage
        # - Batch size 8 for faster inference
        # - Early stop if score >= 0.95
        fact_checker = EntailmentFactChecker(
            ent_model, 
            threshold=0.7, 
            overlap_threshold=0.15,
            max_sentences_per_passage=5,
            max_passages=10,
            batch_size=8,
            early_stop_threshold=0.95
        )
    else:
        raise NotImplementedError

    predict_two_classes(examples, fact_checker)
