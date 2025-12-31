# models.py

import numpy as np


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index):
        import torch
        self.model = model
        self.vocab_index = vocab_index
        self.model.eval()

    def get_next_char_log_probs(self, context):
        import torch
        # Convert context to indices
        context_indices = [self.vocab_index.index_of(c) for c in context]
        
        # Add start token (space) and pad/truncate to 20 characters
        input_indices = [self.vocab_index.index_of(' ')] + context_indices[-19:]
        
        # Convert to tensor
        input_tensor = torch.LongTensor(input_indices)
        
        # Get log probabilities for the last position
        with torch.no_grad():
            log_probs, _ = self.model(input_tensor)
            last_char_log_probs = log_probs[-1, :]  # Last position, all classes
        
        return last_char_log_probs.numpy()

    def get_log_prob_sequence(self, next_chars, context):
        total_log_prob = 0.0
        current_context = context
        
        for char in next_chars:
            # Get log probabilities for next character
            next_char_log_probs = self.get_next_char_log_probs(current_context)
            
            # Get log probability for the actual character
            char_idx = self.vocab_index.index_of(char)
            char_log_prob = next_char_log_probs[char_idx]
            
            total_log_prob += char_log_prob
            
            # Update context
            current_context += char
        
        return total_log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    import torch
    import torch.nn as nn
    from torch import optim
    import random
    from transformer import Transformer
    
    # Use GPU if available (CUDA for Gradescope, MPS for Mac), otherwise CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Model hyperparameters - optimized for GPU performance
    vocab_size = len(vocab_index)
    num_positions = 20  # sequence length
    d_model = 256  # larger embedding dimension for better performance
    d_internal = 128  # larger internal attention dimension
    num_classes = vocab_size  # 27 characters
    num_layers = 3  # use 3 layers for better performance
    
    # Create model with causal masking
    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, causal=True)
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # higher learning rate for faster convergence
    loss_fcn = nn.NLLLoss()

    # Training parameters - optimized for GPU performance
    chunk_size = 20
    batch_size = 128  # larger batch size for GPU
    num_epochs = 5  # more epochs for better performance
    max_chunks = 1000  # more training data for better performance
    
    print(f"Training language model with {len(train_text)} characters...")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_chunks = 0
        
        # Process text in chunks (limited for speed)
        max_start = min(len(train_text) - chunk_size, max_chunks * batch_size)
        for i in range(0, max_start, batch_size):
            batch_inputs = []
            batch_targets = []
            batch_size_actual = min(batch_size, len(train_text) - i - chunk_size)
            
            for j in range(batch_size_actual):
                start_idx = i + j
                if start_idx + chunk_size >= len(train_text):
                    break
                    
                # Get chunk of characters
                chunk = train_text[start_idx:start_idx + chunk_size]
                
                # Convert to indices
                input_indices = [vocab_index.index_of(c) for c in chunk]
                
                # Target is the next character for each position
                target_indices = [vocab_index.index_of(c) for c in train_text[start_idx + 1:start_idx + chunk_size + 1]]
                
                batch_inputs.append(input_indices)
                batch_targets.append(target_indices)
            
            if batch_inputs:
                # Process batch
                total_batch_loss = 0.0
                model.zero_grad()
                
                for input_indices, target_indices in zip(batch_inputs, batch_targets):
                    # Convert to tensors and move to device
                    input_tensor = torch.LongTensor(input_indices).to(device)
                    target_tensor = torch.LongTensor(target_indices).to(device)
                    
                    # Forward pass
                    log_probs, _ = model(input_tensor)
                    
                    # Compute loss
                    loss = loss_fcn(log_probs, target_tensor)
                    total_batch_loss += loss
                
                # Average loss and backward pass
                avg_batch_loss = total_batch_loss / len(batch_inputs)
                avg_batch_loss.backward()
                optimizer.step()
                
                total_loss += avg_batch_loss.item()
                num_chunks += len(batch_inputs)
        
        avg_epoch_loss = total_loss / num_chunks if num_chunks > 0 else 0
        print(f"Epoch {epoch}: Average loss = {avg_epoch_loss:.4f}")
    
    model.eval()
    return NeuralLanguageModel(model, vocab_index)
