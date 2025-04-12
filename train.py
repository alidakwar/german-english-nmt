import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

from transformer_model import (
    TransformerNMT, SimpleTokenizer,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    generate_square_subsequent_mask
)

########################################
# 1) Data & Vocabulary Setup
########################################

def build_vocab(hf_dataset, lang, max_vocab_size):
    """
    Build a vocabulary from the given Hugging Face dataset for the specified language.
    Includes special tokens: <pad>, <bos>, <eos>, <unk>.
    """
    counter = Counter()
    for item in hf_dataset:
        sentence = item["translation"][lang].lower()
        tokens = sentence.split()
        counter.update(tokens)

    vocab = {
        "<pad>": PAD_TOKEN,
        "<bos>": BOS_TOKEN,
        "<eos>": EOS_TOKEN,
        "<unk>": UNK_TOKEN
    }
    idx = 4
    for word, _ in counter.most_common(max_vocab_size - 4):
        vocab[word] = idx
        idx += 1
    return vocab

class HuggingfaceTranslationDataset(Dataset):
    def __init__(self, hf_dataset, src_lang="de", tgt_lang="en",
                 src_tokenizer=None, tgt_tokenizer=None):
        self.dataset = hf_dataset
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_sentence = item["translation"][self.src_lang]
        tgt_sentence = item["translation"][self.tgt_lang]

        src_tokens = self.src_tokenizer.tokenize(src_sentence)
        tgt_tokens = self.tgt_tokenizer.tokenize(tgt_sentence)

        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt_tokens, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lengths = [s.size(0) for s in src_batch]
    tgt_lengths = [t.size(0) for t in tgt_batch]
    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)

    padded_src = []
    padded_tgt = []

    for s in src_batch:
        pad_size = max_src_len - s.size(0)
        padded_src.append(torch.cat([s, torch.full((pad_size,), PAD_TOKEN, dtype=torch.long)]))
    for t in tgt_batch:
        pad_size = max_tgt_len - t.size(0)
        padded_tgt.append(torch.cat([t, torch.full((pad_size,), PAD_TOKEN, dtype=torch.long)]))

    padded_src = torch.stack(padded_src)  # (batch_size, max_src_len)
    padded_tgt = torch.stack(padded_tgt)  # (batch_size, max_tgt_len)

    # Transpose to shape (seq_len, batch_size)
    return padded_src.transpose(0, 1), padded_tgt.transpose(0, 1)

########################################
# 2) Training Utilities & Metrics
########################################

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on a given DataLoader and return the average loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.to(device)  # (src_seq_len, batch_size)
            tgt_batch = tgt_batch.to(device)  # (tgt_seq_len, batch_size)

            tgt_input = tgt_batch[:-1, :]
            tgt_output = tgt_batch[1:, :]

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
            output = model(src_batch, tgt_input, tgt_mask=tgt_mask)
            output_dim = output.shape[-1]

            loss = criterion(
                output.reshape(-1, output_dim),
                tgt_output.reshape(-1)
            )
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def noam_lr_lambda(step, d_model=256, warmup=4000):
    """
    Noam schedule: lr = d_model**(-0.5) * min(step**(-0.5), step*(warmup**-1.5))
    """
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))

def greedy_decode(model, src, max_len, start_symbol, device):
    """
    Given a source sequence, generate a translation using greedy decoding.
    """
    model.eval()
    src = src.to(device)
    src = src.unsqueeze(1)  # (src_seq_len, 1)
    src_emb = model.src_embedding(src) * math.sqrt(model.d_model)
    src_emb = model.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)
    memory = model.transformer.encoder(src_emb)
    
    ys = torch.tensor([start_symbol], dtype=torch.long).to(device).unsqueeze(1)  # (1, 1)
    for i in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).to(device)
        tgt_emb = model.tgt_embedding(ys) * math.sqrt(model.d_model)
        tgt_emb = model.pos_decoder(tgt_emb.transpose(0, 1)).transpose(0, 1)
        out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        out = model.fc_out(out)
        next_token_logits = out[-1, 0]
        _, next_token = torch.max(next_token_logits, dim=0)
        next_token = next_token.item()
        ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=0)
        if next_token == EOS_TOKEN:
            break
    return ys.squeeze(1)

def compute_bleu(model, dataset, src_tokenizer, tgt_tokenizer, device, max_len=50):
    """
    Compute the corpus-level BLEU score over a dataset using NLTK's corpus_bleu.
    """
    from nltk.translate.bleu_score import corpus_bleu
    references = []
    hypotheses = []
    for i in range(len(dataset)):
        src_tensor, tgt_tensor = dataset[i]
        ref_sentence = tgt_tokenizer.detokenize(tgt_tensor.tolist())
        ref_words = ref_sentence.split()
        pred_tokens = greedy_decode(model, src_tensor, max_len, BOS_TOKEN, device)
        pred_sentence = tgt_tokenizer.detokenize(pred_tokens.tolist())
        pred_words = pred_sentence.split()
        references.append([ref_words])
        hypotheses.append(pred_words)
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

########################################
# 3) Main Training Script with New Parameters
########################################

if __name__ == "__main__":
    # --------------------------------------------------------
    # Updated Hyperparameters for Even Better Performance
    # --------------------------------------------------------
    MAX_VOCAB_SIZE = 10000
    BATCH_SIZE = 32    
    NUM_EPOCHS = 40        
    LEARNING_RATE = 0.05   
    WARMUP_STEPS = 4000        
    D_MODEL = 256               
    FEEDFORWARD_DIM = 1024     
    NHEAD = 4                   
    NUM_LAYERS = 2             
    DROPOUT = 0.1               
    PATIENCE = 10               

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # 1) Load and Split Dataset
    # --------------------------------------------------------
    full_data = load_dataset("opus_books", "de-en", split="train[:100%]")
    print("Loaded dataset with", len(full_data), "examples.")

    full_data = full_data.shuffle(seed=42)
    total_len = len(full_data)
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size

    train_data = full_data.select(range(train_size))
    val_data = full_data.select(range(train_size, train_size + val_size))
    test_data = full_data.select(range(train_size + val_size, total_len))

    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    # --------------------------------------------------------
    # 2) Build Vocabulary
    # --------------------------------------------------------
    src_vocab = build_vocab(train_data, "de", MAX_VOCAB_SIZE)
    tgt_vocab = build_vocab(train_data, "en", MAX_VOCAB_SIZE)

    with open("src_vocab.json", "w", encoding="utf-8") as f:
        json.dump(src_vocab, f, ensure_ascii=False, indent=2)
    with open("tgt_vocab.json", "w", encoding="utf-8") as f:
        json.dump(tgt_vocab, f, ensure_ascii=False, indent=2)

    src_tokenizer = SimpleTokenizer(src_vocab)
    tgt_tokenizer = SimpleTokenizer(tgt_vocab)

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print("src_vocab_size:", src_vocab_size)
    print("tgt_vocab_size:", tgt_vocab_size)

    # --------------------------------------------------------
    # 3) Create Datasets & Dataloaders
    # --------------------------------------------------------
    train_dataset = HuggingfaceTranslationDataset(
        train_data, src_lang="de", tgt_lang="en",
        src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
    )
    val_dataset = HuggingfaceTranslationDataset(
        val_data, src_lang="de", tgt_lang="en",
        src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
    )
    test_dataset = HuggingfaceTranslationDataset(
        test_data, src_lang="de", tgt_lang="en",
        src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # 4) Instantiate Model
    # --------------------------------------------------------
    model = TransformerNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        dim_feedforward=FEEDFORWARD_DIM,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_TOKEN,
    label_smoothing=0.1)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,  # Learning rate further shaped by the LR scheduler
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: noam_lr_lambda(step, d_model=D_MODEL, warmup=WARMUP_STEPS)
    )

    # --------------------------------------------------------
    # 5) Training Loop with BLEU & Perplexity Metrics
    # --------------------------------------------------------
    best_val_loss = float("inf")
    no_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        # Wrap the training dataloader with tqdm for a progress bar
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch:02d}") as pbar:
            for src_batch, tgt_batch in train_dataloader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)

                tgt_input = tgt_batch[:-1, :]
                tgt_output = tgt_batch[1:, :]

                tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
                optimizer.zero_grad()

                output = model(src_batch, tgt_input, tgt_mask=tgt_mask)
                output_dim = output.shape[-1]
                loss = criterion(
                    output.reshape(-1, output_dim),
                    tgt_output.reshape(-1)
                )
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_train_loss += loss.item()

                # Update the progress bar for each batch processed
                pbar.update(1)

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = evaluate(model, val_dataloader, criterion, device)
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 100 else float('inf')
        val_ppl = math.exp(avg_val_loss) if avg_val_loss < 100 else float('inf')
        #val_bleu = compute_bleu(model, val_dataset, src_tokenizer, tgt_tokenizer, device, max_len=50)
        val_bleu = 0;
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f} | Val BLEU: {val_bleu:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improvement += 1
            if no_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break

    # --------------------------------------------------------
    # 6) Final Test Evaluation (BLEU & Perplexity)
    # --------------------------------------------------------
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    test_loss = evaluate(model, test_dataloader, criterion, device)
    test_ppl = math.exp(test_loss) if test_loss < 100 else float('inf')
    #test_bleu = compute_bleu(model, test_dataset, src_tokenizer, tgt_tokenizer, device, max_len=50)
    test_bleu = 0
    print(f"\nTraining complete. Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.2f} | Test BLEU: {test_bleu:.4f}\n")
