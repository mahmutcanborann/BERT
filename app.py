import streamlit as st
import torch
import json
from transformers import BertTokenizerFast
from model import IntentSlotBERT  # Model sınıfını çağırıyoruz
import nest_asyncio
import os

nest_asyncio.apply()

# 📌 **Intent ve Slot Etiketlerini JSON Dosyalarından Yükle**
with open("bert-turkish-intent-slot/intent_labels.json", "r") as f:
    intent_labels = json.load(f)

with open("bert-turkish-intent-slot/slot_labels.json", "r") as f:
    slot_labels = json.load(f)

# 📌 **Modeli ve Tokenizer'ı Yükle**
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
tokenizer = BertTokenizerFast.from_pretrained("bert-turkish-intent-slot")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model_path = "bert-turkish-intent-slot/model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError("❗ model.pth dosyası eksik. Lütfen README'deki Google Drive linkinden indirip 'bert-turkish-intent-slot' klasörüne yerleştirin.")

model = IntentSlotBERT(MODEL_NAME, num_intents=len(intent_labels), num_slots=len(slot_labels)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()



# 📌 **Streamlit Arayüzü**
st.title("🚗 Türkçe BERT - Intent & Slot Tanıma")
st.write("Kendi eğittiğimiz BERT modeli ile kullanıcı girdisini analiz ediyoruz!")

# Kullanıcıdan giriş al
sentence = st.text_input("Bir komut girin:", "Farları aç")

# Eğer giriş varsa, modeli çalıştır
if st.button("Tahmin Yap"):
    with torch.no_grad():
        # Tokenize et
        inputs = tokenizer(sentence, padding="max_length", truncation=True, max_length=50, return_tensors="pt").to(device)

        # Model tahmini yap
        intent_logits, slot_logits = model(inputs["input_ids"], inputs["attention_mask"])

        # 📌 **Intent Sonucu**
        intent_pred = torch.argmax(intent_logits, dim=1).item()
        predicted_intent = [key for key, value in intent_labels.items() if value == intent_pred][0]

        # 📌 **Slot Sonucu**
        slot_preds = torch.argmax(slot_logits, dim=2).squeeze(0).tolist()
        tokenized_words = tokenizer.tokenize(sentence)

        predicted_slots = []
        for word, slot_index in zip(tokenized_words, slot_preds[:len(tokenized_words)]):
            slot_name = [key for key, value in slot_labels.items() if value == slot_index][0]
            if slot_name != "O":  # "O" olanları göstermeyelim
                predicted_slots.append(f"{slot_name}: {word}")

        # 📌 **Sonucu Göster**
        st.subheader("📌 Tahmin Sonuçları:")
        st.write(f"**Intent:** `{predicted_intent}`")
        st.write(f"**Slots:**")
        if predicted_slots:
            for slot in predicted_slots:
                st.write(f"🔹 {slot}")
        else:
            st.write("🚫 **Slot bulunamadı!**")

        # 📌 **Komuta Göre Araba Animasyonunu Güncelle**
        st.write("---")  # Görsel ayrım için
        if predicted_intent == "far_ac":
            st.write("💡 **Farlar açıldı!**")
        elif predicted_intent == "far_kapat":
            st.write("🌑 **Farlar kapatıldı!**")
        elif predicted_intent == "kapı_ac":
            st.write("🚪 **Kapılar açıldı!**")
        elif predicted_intent == "kapı_kapat":
            st.write("🔒 **Kapılar kapatıldı!**")
        elif predicted_intent == "cam_ac":
            st.write("🪟 **Camlar açıldı!**")
        elif predicted_intent == "cam_kapat":
            st.write("🪟 **Camlar kapatıldı!**")


# final version - test
