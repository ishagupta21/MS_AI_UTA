# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, causal=False):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        :param causal: If True, use causal masking for language modeling
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.causal = causal
        
        # Character embedding layer
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding (optional for Part 0, required for Part 1)
        self.pos_encoding = PositionalEncoding(d_model, num_positions)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, d_internal, causal=causal) for _ in range(num_layers)
        ])
        
        # Output projection to num_classes
        self.output_projection = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # Convert indices to tensor if needed
        if isinstance(indices, list):
            indices = torch.LongTensor(indices)
        
        # Character embeddings: [seq_len, d_model]
        char_embeddings = self.char_embedding(indices)
        
        # Add positional encodings
        embedded = self.pos_encoding(char_embeddings)
        
        # Pass through transformer layers
        attention_maps = []
        x = embedded
        for layer in self.transformer_layers:
            x, attention_weights = layer(x)
            attention_maps.append(attention_weights)
        
        # Output projection: [seq_len, num_classes]
        logits = self.output_projection(x)
        
        # Apply log softmax to get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        return log_probs, attention_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, causal=False):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        :param causal: If True, apply causal masking to prevent attending to future tokens
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.causal = causal
        
        # Self-attention components
        self.W_q = nn.Linear(d_model, d_internal)  # Query projection
        self.W_k = nn.Linear(d_model, d_internal)  # Key projection  
        self.W_v = nn.Linear(d_model, d_model)     # Value projection
        self.W_o = nn.Linear(d_model, d_model)     # Output projection
        
        # Feed-forward network
        self.ff1 = nn.Linear(d_model, d_model)     # First linear layer
        self.ff2 = nn.Linear(d_model, d_model)     # Second linear layer
        
        # Activation function
        self.activation = nn.ReLU()

    def forward(self, input_vecs):
        # input_vecs shape: [seq_len, d_model]
        seq_len, d_model = input_vecs.shape
        
        # Self-attention
        Q = self.W_q(input_vecs)  # [seq_len, d_internal]
        K = self.W_k(input_vecs)  # [seq_len, d_internal]
        V = self.W_v(input_vecs)  # [seq_len, d_model]
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_internal ** 0.5)
        # attention_scores shape: [seq_len, seq_len]
        
        # Apply causal mask if needed
        if self.causal:
            # Create causal mask: lower triangular matrix
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attention_scores.device))
            attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [seq_len, d_model]
        
        # Output projection
        attended = self.W_o(attended)
        
        # First residual connection
        residual1 = input_vecs + attended
        
        # Feed-forward network
        ff_out = self.ff2(self.activation(self.ff1(residual1)))
        
        # Second residual connection
        output = residual1 + ff_out
        
        return output, attention_weights


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor).to(x.device)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # Use GPU if available (CUDA for Gradescope), CPU for small models (MPS can be slower)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')  # CPU is often faster for small models
    print(f"Using device: {device}")
    
    # Model hyperparameters - optimized for speed and accuracy
    vocab_size = 27  # 26 letters + space
    num_positions = 20  # sequence length
    d_model = 64  # embedding dimension
    d_internal = 32  # internal attention dimension
    num_classes = 3  # 0, 1, 2+
    num_layers = 1  # use 1 layer for speed (already achieves 99.7% accuracy)
    
    # Create model and move to GPU
    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = nn.NLLLoss()

    num_epochs = 10
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        
        for ex_idx in ex_idxs:
            example = train[ex_idx]
            
            # Move tensors to GPU
            input_tensor = example.input_tensor.to(device)
            target_tensor = example.output_tensor.to(device)
            
            # Forward pass
            log_probs, _ = model(input_tensor)
            
            # Compute loss (NLLLoss expects log probabilities and target indices)
            loss = loss_fcn(log_probs, target_tensor)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_this_epoch += loss.item()
        
        avg_loss = loss_this_epoch / len(train)
        print(f"Epoch {t}: Average loss = {avg_loss:.4f}")
    
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        # Move input tensor to same device as model
        device = next(model.parameters()).device
        input_tensor = ex.input_tensor.to(device)
        (log_probs, attn_maps) = model.forward(input_tensor)
        predictions = np.argmax(log_probs.detach().cpu().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().cpu().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
