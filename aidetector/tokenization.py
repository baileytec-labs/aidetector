"""
This module contains the functionality focused around tokenization.

Imports:
    spacy: Library for advanced Natural Language Processing (NLP).
    spacy.cli: Command-line interface for spaCy library.
    torchtext.vocab: Library for creating vocabulary from iterators.
    torch: PyTorch library for tensor computations and neural networks.
    nn: Neural network modules from PyTorch library.
"""

import spacy
import spacy.cli
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch import nn

def tokenize(iterator,tokenizer):
    """
    Tokenize the given iterator using a specified tokenizer.

    Parameters:
    iterator (iterator): An iterator of text data.
    tokenizer (function): The function used for tokenizing the text.

    Yields:
    iterator: An iterator of tokenized text.
    """
    
    for text in iterator:
        yield tokenizer(text)

def get_vocab(trained_text,tokenizer):
    """
    Build vocabulary from trained text data using a tokenizer.

    Parameters:
    trained_text (iterable): An iterable of trained text data.
    tokenizer (function): The function used for tokenizing the text.

    Returns:
    Vocab: A torchtext Vocab object.
    """
    vocab = build_vocab_from_iterator(tokenize(trained_text,tokenizer), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def process_data(raw_text_iter, vocab, tokenizer):
    """
    Process raw text data: tokenize, convert to tensor and pad sequences.

    Parameters:
    raw_text_iter (iterable): An iterable of raw text data.
    vocab (Vocab): A torchtext Vocab object.
    tokenizer (function): The function used for tokenizing the text.

    Returns:
    Tensor: A tensor of processed text data.
    """
    data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                         dtype=torch.long) for item in raw_text_iter]
    data = nn.utils.rnn.pad_sequence(data, batch_first=True)
    return data

def tokenize_data(trained_text, tested_text,tokenizer):
    """
    Tokenize and process trained and tested text data.

    Parameters:
    trained_text (iterable): An iterable of trained text data.
    tested_text (iterable): An iterable of tested text data.
    tokenizer (function): The function used for tokenizing the text.

    Returns:
    Tuple: A tuple containing vocabulary, processed trained sequences and processed tested sequences.
    """
    vocab = get_vocab(trained_text,tokenizer)
    trained_sequences = process_data(trained_text, vocab,tokenizer)
    tested_sequences = process_data(tested_text, vocab,tokenizer)

    return vocab, trained_sequences, tested_sequences

def get_tokenizer(tokenmodel='xx_ent_wiki_sm',download=False):
    """
    Get a tokenizer based on a specified spaCy model.

    Parameters:
    tokenmodel (str, optional): The name of the spaCy model to use for tokenization. Default is 'xx_ent_wiki_sm'.
    download (bool, optional): Whether to download the spaCy model or not. Default is False.

    Returns:
    function: A function for tokenizing text.
    """
    if download:
        spacy.cli.download(tokenmodel)
    nlp = spacy.load(tokenmodel) 
    def spacy_tokenizer(text):
        return [tok.text for tok in nlp.tokenizer(text)]
    tokenizer = spacy_tokenizer
    return tokenizer