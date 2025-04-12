import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformer_model import TransformerNMT, SimpleTokenizer, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, generate_square_subsequent_mask
from train import HuggingfaceTranslationDataset, collate_fn, evaluate

# ---------------------- Configuration ----------------------
BATCH_SIZE = 8
MAX_LEN = 50
MODEL_PATH = "best_model.pt"
SRC_VOCAB_PATH = "src_vocab.json"
TGT_VOCAB_PATH = "tgt_vocab.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Load Vocabularies ----------------------
with open(SRC_VOCAB_PATH, "r", encoding="utf-8") as f:
    src_vocab = json.load(f)
with open(TGT_VOCAB_PATH, "r", encoding="utf-8") as f:
    tgt_vocab = json.load(f)

src_tokenizer = SimpleTokenizer(src_vocab)
tgt_tokenizer = SimpleTokenizer(tgt_vocab)

# ---------------------- Load Test Dataset ----------------------
full_data = load_dataset("opus_books", "de-en", split="train[:100%]")
total_len = len(full_data)
train_size = int(0.8 * total_len)
val_size = int(0.1 * total_len)
test_data = full_data.select(range(train_size + val_size, total_len))

test_dataset = HuggingfaceTranslationDataset(
    test_data, src_lang="de", tgt_lang="en",
    src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# ---------------------- Load Model ----------------------
model = TransformerNMT(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=128,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=512,
    dropout=0.1
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------------- Evaluate on Test Set ----------------------
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
test_loss = evaluate(model, test_loader, criterion, DEVICE)
print(f"\n✅ Test Loss: {test_loss:.4f}")

# ---------------------- Sample Translation ----------------------
def greedy_decode(model, src, max_len, start_symbol, device):
    model.eval()
    src = src.unsqueeze(1).to(device)
    ys = torch.tensor([start_symbol], dtype=torch.long).unsqueeze(1).to(device)
    for _ in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).to(device)
        out = model(src, ys, tgt_mask=tgt_mask)
        next_token = torch.argmax(out[-1, 0, :]).item()
        ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=0)
        if next_token == EOS_TOKEN:
            break
    return ys.squeeze()

# Try one sentence
example = "Hallo, wie geht es dir?"
src_ids = torch.tensor(src_tokenizer.tokenize(example), dtype=torch.long)
out_ids = greedy_decode(model, src_ids, MAX_LEN, BOS_TOKEN, DEVICE)
translation = tgt_tokenizer.detokenize(out_ids.tolist())

print(f"\n📝 Input: {example}")
print(f"📘 Translation: {translation}")
