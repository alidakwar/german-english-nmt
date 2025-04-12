import streamlit as st
import torch
import json
from transformer_model import TransformerNMT, SimpleTokenizer, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, generate_square_subsequent_mask

def load_vocab(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab

def translate_text(model, text, src_tokenizer, tgt_tokenizer, device, max_length=50):
    model.eval()
    src_tokens = src_tokenizer.tokenize(text)
    src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(1).to(device)
    
    tgt_tokens = [BOS_TOKEN]
    for _ in range(max_length):
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long).unsqueeze(1).to(device)
        tgt_mask = generate_square_subsequent_mask(len(tgt_tokens)).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, tgt_mask=tgt_mask)
        next_token_logits = output[-1, 0, :]
        next_token = next_token_logits.argmax().item()
        tgt_tokens.append(next_token)
        if next_token == EOS_TOKEN:
            break
    translation = tgt_tokenizer.detokenize(tgt_tokens)
    return translation

def main():
    st.title("German–English Neural Machine Translation")
    st.markdown("### Enter German text below and click **Translate** to see the English translation!")

    # Sidebar with adjustable settings.
    st.sidebar.header("Settings")
    max_length = st.sidebar.slider("Max Translation Length", min_value=10, max_value=100, value=50)

    # Load the trained model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabularies and create tokenizers.
    try:
        src_vocab = load_vocab("src_vocab.json")
        tgt_vocab = load_vocab("tgt_vocab.json")
    except Exception as e:
        st.error("Vocabularies not found. Please train the model to generate the vocab files.")
        return

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    model = TransformerNMT(
    src_vocab_size,
    tgt_vocab_size,
    d_model=256,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=1024,
    dropout=0.1)

    try:
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return
    model.to(device)

    src_tokenizer = SimpleTokenizer(src_vocab)
    tgt_tokenizer = SimpleTokenizer(tgt_vocab)

    st.markdown("#### German Input")
    german_text = st.text_area("Enter German text here...", height=150)

    if st.button("Translate"):
        if not german_text.strip():
            st.warning("Please enter some German text.")
        else:
            st.write("Tokenized input:", src_tokenizer.tokenize(german_text))  # <- Debug line
            with st.spinner("Translating..."):
                translation = translate_text(model, german_text, src_tokenizer, tgt_tokenizer, device, max_length)
            st.markdown("#### English Translation")
            st.success(translation)


if __name__ == "__main__":
    main()
