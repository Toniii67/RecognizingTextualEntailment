import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = './model'
# model_path = 'model.safetensors'

try:
    import sentencepiece
except ImportError:
    st.error("The sentencepiece library is not installed. Please install it using 'pip install sentencepiece'.")
    st.stop()

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

def predict(sentence1, sentence2):
    inputs = tokenizer.encode_plus(sentence1, sentence2, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)
    labels = ["Netral", "Kontradiksi", "Keterlibatan"]
    return labels[predictions.item()], probabilities

# Streamlit app
st.title("Recognizing Textual Entailment (RTE)")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    kalimat1 = st.text_input("Masukkan kalimat pertama:")

with col2:
    kalimat2 = st.text_input("Masukkan kalimat kedua:")

if st.button("Bandingkan"):
    if kalimat1 and kalimat2:
        label, probabilities = predict(kalimat1, kalimat2)
        st.write(f"Predicted relationship: {label}")
        # st.write(f"Probabilities: {probabilities.numpy()}")
    else:
        st.write("Please enter both sentences to get a comparison.")

