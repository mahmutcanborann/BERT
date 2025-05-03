### 🔗 Download model
Due to GitHub's file size limitations, the model is hosted externally.

➡️ [Download model.pth from Google Drive](https://drive.google.com/uc?id=1XSVfJOxGwF704yCsiVCz9SSJCNdqJlJz)

After downloading, place the file in the project root directory.


# BERT-MODEL
This project utilizes a Turkish BERT model to perform intent detection and slot filling for text commands in the automotive domain. It classifies user inputs such as "Turn on the headlights" or "Open the windows" by identifying the underlying intent and extracting relevant entities for downstream processing.



# 🚗 Turkish BERT - Intent & Slot Detection for Automotive Commands

This project uses a fine-tuned **Turkish BERT** model to detect **intents** and extract **slots** from user commands in the automotive domain.

For example:
- **"Farları aç"** → Intent: turn_on_lights, Slot: farlar
- **"Camları kapat"** → Intent: close_windows, Slot: camlar

---

## 🧠 Model

- Based on: dbmdz/bert-base-turkish-cased
- Fine-tuned on domain-specific Turkish vehicle-related commands
- Trained with PyTorch & Hugging Face Transformers
- Stored using **Git LFS** because the model file (model.pth) exceeds 100MB

---

## 📁 Project Structure


---

## 🚀 Inference Example

python
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = torch.load("model.pth", map_location=torch.device("cpu"))
model.eval()

text = "Camları aç"
# Tokenize and predict intent + slots (custom logic here)


 Requirements
Python 3.8+

torch

transformers

scikit-learn

Git LFS

git lfs install
git clone https://github.com/yourusername/bert-turkish-intent-slot.git

Notes
The model is trained to understand Turkish automotive commands

Can be integrated into voice assistants, infotainment systems, or smart vehicle interfaces

 Author
Mahmut Can Boran
AI Engineer | NLP Developer | Automotive Software Enthusiast
