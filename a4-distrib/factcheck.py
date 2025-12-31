# factcheck.py

import torch
from typing import List
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
    def __init__(self, ent_model, threshold=0.7, overlap_threshold=0.1):
        self.ent_model = ent_model
        self.threshold = threshold
        self.overlap_threshold = overlap_threshold
        self.stop_words = set(stopwords.words('english'))
        # Use NLTK sentence tokenizer for better sentence splitting
        try:
            from nltk.tokenize import sent_tokenize
            self.sent_tokenize = sent_tokenize
        except:
            # Fallback to simple splitting if NLTK tokenizer not available
            self.sent_tokenize = None
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing HTML-like tags and extra whitespace."""
        # Remove <s> and </s> tags
        text = text.replace('<s>', ' ').replace('</s>', ' ')
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK tokenizer if available."""
        text = self._clean_text(text)
        if self.sent_tokenize:
            try:
                sentences = self.sent_tokenize(text)
                # Filter out very short sentences (likely artifacts)
                return [s.strip() for s in sentences if len(s.strip()) > 10]
            except:
                pass
        # Fallback to simple splitting
        sentences = text.split('.')
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def predict(self, fact: str, passages: List[dict]) -> str:
        # Clean fact text
        fact = self._clean_text(fact)
        
        # Prune passages based on word overlap
        fact_tokens = set(word_tokenize(fact.lower())) - self.stop_words
        if len(fact_tokens) == 0:
            return "NS"
        
        filtered_passages = []
        for passage in passages:
            passage_text = self._clean_text(passage['text'])
            passage_tokens = set(word_tokenize(passage_text.lower())) - self.stop_words
            overlap_ratio = len(fact_tokens & passage_tokens) / len(fact_tokens) if len(fact_tokens) > 0 else 0
            if overlap_ratio >= self.overlap_threshold:
                filtered_passages.append(passage_text)
        
        if len(filtered_passages) == 0:
            return "NS"

        max_score = float('-inf')
        for passage_text in filtered_passages:
            sentences = self._split_sentences(passage_text)
            for sentence in sentences:
                if sentence.strip() and len(sentence.strip()) > 10:
                    try:
                        # premise is the passage sentence, hypothesis is the fact
                        score = self.ent_model.check_entailment(sentence.strip(), fact)
                        max_score = max(max_score, score)
                    except Exception as e:
                        # Skip sentences that cause errors
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
