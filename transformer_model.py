import math
import torch
import torch.nn as nn
import re

# Special token definitions
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

class SimpleTokenizer:
    def __init__(self, vocab):
        """
        vocab: a dict mapping tokens (words) to indices.
        It must include the following special tokens:
        "<pad>", "<bos>", "<eos>", and "<unk>".
        """
        self.vocab = vocab
        self.id2word = {i: w for w, i in vocab.items()}

    def tokenize(self, text):
        # Convert to lowercase
        text = text.lower()
    
        # Replace punctuation with spaces (remove commas, question marks, etc.)
        # So "Hallo, wie geht es dir?" -> "hallo  wie geht es dir "
        text = re.sub(r'[^a-z0-9äöüß]+', ' ', text)
    
        # Now split on whitespace
        tokens = text.split()

        token_ids = [self.vocab["<bos>"]]
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab["<unk>"])
        token_ids.append(self.vocab["<eos>"])
        return token_ids



    def detokenize(self, token_ids):
        """
        Converts a list of token IDs back to a string.
        It skips special tokens (PAD, BOS, EOS).
        """
        words = []
        for token_id in token_ids:
            if token_id in (self.vocab["<pad>"], self.vocab["<bos>"], self.vocab["<eos>"]):
                continue
            words.append(self.id2word.get(token_id, "<unk>"))
        return " ".join(words)

def generate_square_subsequent_mask(sz):
    """
    Generates an upper-triangular matrix of -inf values with zeros on the diagonal.
    Used to mask future tokens during decoding.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4,
             num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=1024, dropout=0.1):
        super(TransformerNMT, self).__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers,
                                          num_decoder_layers, dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # src and tgt are expected to be of shape (seq_len, batch_size)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        # Apply positional encoding (Transformer expects shape: [batch, seq_len, d_model])
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask, 
                                  tgt_key_padding_mask=tgt_padding_mask, 
                                  memory_key_padding_mask=memory_key_padding_mask)
        output = self.fc_out(output)
        return output
