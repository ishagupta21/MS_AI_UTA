# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from collections import defaultdict


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class DeepAveragingNetwork(nn.Module):
    """
    Deep Averaging Network implementation for sentiment classification.
    
    The network takes word indices as input, embeds them, averages the embeddings,
    and passes through feedforward layers to return log probabilities.
    """
    def __init__(self, embedding_layer, hidden_dims, dropout_rate=0.1, activation='relu'):
        super(DeepAveragingNetwork, self).__init__()
        
        self.embedding_layer = embedding_layer
        self.embedding_dim = embedding_layer.embedding_dim
        
        # Build the feedforward layers
        layers = []
        prev_dim = self.embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final classification layer (outputs raw scores)
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, word_indices, sentence_lengths):
        """
        Forward pass of the network
        
        Args:
            word_indices: Tensor of shape (batch_size, max_length) - word indices
            sentence_lengths: Tensor of shape (batch_size,) with actual sentence lengths
        
        Returns:
            log_probs: Tensor of shape (batch_size, 2) - log probabilities
        """
        # Get embeddings from word indices
        # word_indices: (batch_size, max_length)
        sentence_embeddings = self.embedding_layer(word_indices)  # (batch_size, max_length, embedding_dim)
        
        # Average the embeddings for each sentence
        # Create mask for padding tokens (PAD token has index 0)
        mask = torch.arange(sentence_embeddings.size(1), device=sentence_embeddings.device)[None, :] < sentence_lengths[:, None]
        mask = mask.float().unsqueeze(-1)  # (batch_size, max_length, 1)
        
        # Apply mask and sum
        masked_embeddings = sentence_embeddings * mask
        summed_embeddings = torch.sum(masked_embeddings, dim=1)  # (batch_size, embedding_dim)
        
        # Average by sentence length
        averaged_embeddings = summed_embeddings / sentence_lengths.float().unsqueeze(-1)
        
        # Pass through the network to get raw scores
        raw_scores = self.network(averaged_embeddings)
        
        # Convert to log probabilities using log softmax
        log_probs = torch.log_softmax(raw_scores, dim=1)
        
        return log_probs


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Neural sentiment classifier that wraps the DeepAveragingNetwork
    """
    def __init__(self, network, embeddings, device='cpu', use_prefix=False):
        self.network = network
        self.embeddings = embeddings
        self.device = device
        self.use_prefix = use_prefix
        self.network.to(device)
        self.network.eval()
    
    def predict(self, ex_words, has_typos=False):
        """
        Predict sentiment for a single example
        """
        # Convert words to indices
        word_indices = []
        for word in ex_words:
            if self.use_prefix:
                # Use prefix embeddings
                prefix = self.embeddings.get_prefix(word)
                idx = self.embeddings.prefix_indexer.index_of(prefix)
                if idx == -1:
                    idx = self.embeddings.prefix_indexer.index_of("UNK")
            else:
                # Use word embeddings
                idx = self.embeddings.word_indexer.index_of(word)
                if idx == -1:
                    idx = self.embeddings.word_indexer.index_of("UNK")
            word_indices.append(idx)
        
        # Convert to tensor
        word_tensor = torch.tensor([word_indices], dtype=torch.long, device=self.device)
        length_tensor = torch.tensor([len(ex_words)], dtype=torch.long, device=self.device)
        
        # Get predictions
        with torch.no_grad():
            log_probs = self.network(word_tensor, length_tensor)
            prediction = torch.argmax(log_probs, dim=1).item()
        
        return prediction
    
    def predict_all(self, all_ex_words, has_typos=False):
        """
        Predict sentiment for multiple examples (batched version for efficiency)
        """
        if len(all_ex_words) == 0:
            return []
        
        # Convert all examples to indices
        all_word_indices = []
        all_lengths = []
        
        for ex_words in all_ex_words:
            word_indices = []
            for word in ex_words:
                if self.use_prefix:
                    # Use prefix embeddings
                    prefix = self.embeddings.get_prefix(word)
                    idx = self.embeddings.prefix_indexer.index_of(prefix)
                    if idx == -1:
                        idx = self.embeddings.prefix_indexer.index_of("UNK")
                else:
                    # Use word embeddings
                    idx = self.embeddings.word_indexer.index_of(word)
                    if idx == -1:
                        idx = self.embeddings.word_indexer.index_of("UNK")
                word_indices.append(idx)
            
            all_word_indices.append(word_indices)
            all_lengths.append(len(ex_words))
        
        # Pad sequences to same length
        max_length = max(all_lengths)
        padded_indices = []
        for word_indices in all_word_indices:
            padded = word_indices + [0] * (max_length - len(word_indices))  # Pad with PAD token (index 0)
            padded_indices.append(padded)
        
        # Convert to tensors
        word_tensor = torch.tensor(padded_indices, dtype=torch.long, device=self.device)
        length_tensor = torch.tensor(all_lengths, dtype=torch.long, device=self.device)
        
        # Get predictions
        with torch.no_grad():
            log_probs = self.network(word_tensor, length_tensor)
            predictions = torch.argmax(log_probs, dim=1).cpu().numpy().tolist()
        
        return predictions


class PrefixEmbeddings:
    """
    Character-based prefix embeddings for handling typos.
    Uses the first 3 characters of words to create embeddings that are robust to typos.
    """
    def __init__(self, prefix_indexer, vectors):
        self.prefix_indexer = prefix_indexer
        self.vectors = vectors

    def get_initialized_embedding_layer(self, frozen=False, padding_idx=None):
        """
        :param frozen: True if you want the embedding layer to stay frozen, false to fine-tune embeddings
        :param padding_idx: Set to a value that you want to be labeled as "padding" in the embedding space
        :return: torch.nn.Embedding layer you can use in your network
        """
        return torch.nn.Embedding.from_pretrained(torch.FloatTensor(self.vectors), freeze=frozen, padding_idx=padding_idx)

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, prefix):
        """
        Returns the embedding for a given prefix
        :param prefix: The prefix to look up
        :return: The UNK vector if the prefix is not in the Indexer or the vector otherwise
        """
        prefix_idx = self.prefix_indexer.index_of(prefix)
        if prefix_idx != -1:
            return self.vectors[prefix_idx]
        else:
            return self.vectors[self.prefix_indexer.index_of("UNK")]

    def get_prefix(self, word):
        """
        Extract the prefix from a word (first 3 characters)
        :param word: The word to extract prefix from
        :return: The 3-character prefix (or shorter if word is shorter)
        """
        return word[:3] if len(word) >= 3 else word


def create_prefix_embeddings(word_embeddings, embedding_dim):
    """
    Create prefix embeddings by averaging word embeddings for words that start with the same prefix.
    """
    from utils import Indexer
    
    # Create prefix indexer
    prefix_indexer = Indexer()
    prefix_indexer.add_and_get_index("PAD")  # Position 0
    prefix_indexer.add_and_get_index("UNK")  # Position 1
    
    # Collect all prefixes and their corresponding word embeddings
    prefix_to_embeddings = defaultdict(list)
    
    # Go through all words in the word embeddings
    for word_idx in range(2, len(word_embeddings.word_indexer)):  # Skip PAD and UNK
        word = word_embeddings.word_indexer.get_object(word_idx)
        if word is not None:
            prefix = word[:3] if len(word) >= 3 else word
            word_embedding = word_embeddings.vectors[word_idx]
            prefix_to_embeddings[prefix].append(word_embedding)
    
    # Create prefix vectors by averaging word embeddings
    prefix_vectors = []
    prefix_vectors.append(np.zeros(embedding_dim))  # PAD vector
    prefix_vectors.append(np.zeros(embedding_dim))  # UNK vector
    
    for prefix, embeddings_list in prefix_to_embeddings.items():
        prefix_indexer.add_and_get_index(prefix)
        # Average the embeddings for words that start with this prefix
        avg_embedding = np.mean(embeddings_list, axis=0)
        prefix_vectors.append(avg_embedding)
    
    print(f"Created {len(prefix_indexer)} prefix embeddings from {len(word_embeddings.word_indexer)} word embeddings")
    return PrefixEmbeddings(prefix_indexer, np.array(prefix_vectors))


def create_batches(examples, batch_size):
    """
    Create batches of examples for training
    """
    batches = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        batches.append(batch)
    return batches


def prepare_batch(batch_examples, embeddings, device, use_prefix=False):
    """
    Prepare a batch of examples for training
    
    Args:
        batch_examples: List of SentimentExample objects
        embeddings: WordEmbeddings or PrefixEmbeddings object
        device: torch device
        use_prefix: If True, use prefix embeddings instead of word embeddings
    
    Returns:
        word_tensor: (batch_size, max_length) - word/prefix indices
        length_tensor: (batch_size,) - sentence lengths  
        labels_tensor: (batch_size,) - labels
    """
    all_word_indices = []
    all_lengths = []
    all_labels = []
    
    for ex in batch_examples:
        word_indices = []
        for word in ex.words:
            if use_prefix:
                # Use prefix embeddings
                prefix = embeddings.get_prefix(word)
                idx = embeddings.prefix_indexer.index_of(prefix)
                if idx == -1:
                    idx = embeddings.prefix_indexer.index_of("UNK")
            else:
                # Use word embeddings
                idx = embeddings.word_indexer.index_of(word)
                if idx == -1:
                    idx = embeddings.word_indexer.index_of("UNK")
            word_indices.append(idx)
        
        all_word_indices.append(word_indices)
        all_lengths.append(len(ex.words))
        all_labels.append(ex.label)
    
    # Pad sequences to same length (use dynamic padding for efficiency)
    max_length = max(all_lengths)
    # Cap at reasonable length to avoid memory issues with very long sentences
    max_length = min(max_length, 128)
    
    padded_indices = []
    for word_indices in all_word_indices:
        if len(word_indices) > max_length:
            # Truncate if too long
            word_indices = word_indices[:max_length]
            all_lengths[len(padded_indices)] = max_length
        
        padded = word_indices + [0] * (max_length - len(word_indices))  # Pad with PAD token (index 0)
        padded_indices.append(padded)
    
    # Convert to tensors
    word_tensor = torch.tensor(padded_indices, dtype=torch.long, device=device)
    length_tensor = torch.tensor(all_lengths, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=device)
    
    return word_tensor, length_tensor, labels_tensor


def evaluate_model(model, eval_examples, embeddings, device, use_prefix=False):
    """
    Evaluate model on a set of examples
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        batches = create_batches(eval_examples, batch_size=32)
        for batch in batches:
            word_tensor, length_tensor, labels_tensor = prepare_batch(batch, embeddings, device, use_prefix)
            
            # Forward pass - model now takes word indices directly
            log_probs = model(word_tensor, length_tensor)
            predictions = torch.argmax(log_probs, dim=1)
            
            correct += (predictions == labels_tensor).sum().item()
            total += labels_tensor.size(0)
    
    return correct / total


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    Train a deep averaging network for sentiment classification
    
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Choose embeddings based on typo setting
    if train_model_for_typo_setting:
        print("Training with prefix embeddings for typo robustness...")
        embeddings = create_prefix_embeddings(word_embeddings, word_embeddings.get_embedding_length())
        use_prefix = True
        frozen_embeddings = False  # Fine-tune prefix embeddings
    else:
        print("Training with word embeddings...")
        embeddings = word_embeddings
        use_prefix = False
        frozen_embeddings = True  # Keep word embeddings frozen
    
    # Model configuration - use command line arguments
    embedding_dim = embeddings.get_embedding_length()
    hidden_size = getattr(args, 'hidden_size', 100)
    hidden_dims = [hidden_size]  # Single hidden layer as default
    dropout_rate = 0.2  # Moderate dropout
    activation = 'relu'  # Fast activation
    
    # Create embedding layer with padding_idx=0 for efficient padding
    embedding_layer = embeddings.get_initialized_embedding_layer(frozen=frozen_embeddings, padding_idx=0)
    
    # Create model
    model = DeepAveragingNetwork(
        embedding_layer=embedding_layer,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        activation=activation
    )
    model.to(device)
    
    # Training configuration - use command line arguments
    batch_size = getattr(args, 'batch_size', 1)
    learning_rate = getattr(args, 'lr', 0.001)
    num_epochs = getattr(args, 'num_epochs', 10)
    weight_decay = getattr(args, 'weight_decay', 1e-4)
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.NLLLoss()  # Use NLLLoss since we return log probabilities
    
    print(f"Training with {len(train_exs)} examples")
    print(f"Model architecture: {embedding_dim} -> {' -> '.join(map(str, hidden_dims))} -> 2")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Epochs: {num_epochs}")
    
    # Training loop with early stopping
    best_dev_acc = 0.0
    patience = 3  # Allow some patience for better convergence
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create batches
        batches = create_batches(train_exs, batch_size)
        random.shuffle(batches)  # Shuffle for each epoch
        
        for batch in batches:
            optimizer.zero_grad()
            
            # Prepare batch
            word_tensor, length_tensor, labels_tensor = prepare_batch(batch, embeddings, device, use_prefix)
            
            # Forward pass - model now takes word indices directly
            log_probs = model(word_tensor, length_tensor)
            loss = criterion(log_probs, labels_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Evaluate on dev set
        dev_acc = evaluate_model(model, dev_exs, embeddings, device, use_prefix)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Dev Acc = {dev_acc:.4f}")
        
        # Early stopping
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"Training completed. Best dev accuracy: {best_dev_acc:.4f}")
    
    # Create classifier wrapper
    classifier = NeuralSentimentClassifier(model, embeddings, device, use_prefix)
    
    return classifier

