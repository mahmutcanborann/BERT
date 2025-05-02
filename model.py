import torch
import torch.nn as nn
from transformers import BertModel


class IntentSlotBERT(nn.Module):
    def __init__(self, model_name, num_intents, num_slots):
        super(IntentSlotBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output  # [CLS] token'ı için

        intent_logits = self.intent_classifier(pooled_output)  # Intent sınıflandırması
        slot_logits = self.slot_classifier(sequence_output)  # Slot etiketleme

        return intent_logits, slot_logits
