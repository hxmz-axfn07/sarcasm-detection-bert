from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "final_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()


def predict(text, context=""):
    combined = context + " [SEP] " + text

    inputs = tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    sarcasm_prob = probs[0][1].item()

    if sarcasm_prob > 0.35:
        label = "Sarcastic 😏"
    else:
        label = "Not Sarcastic 🙂"
    
    return {"label": label, "confidence": round(sarcasm_prob, 3)}

# TEST
if __name__ == "__main__":
    print(predict("Oh great, another bug in my code"))
    print(predict("I am going to the store"))
    print(predict("Area man absolutely thrilled to work overtime again"))
    print(predict("Local man wins lottery for third time this week"))
    print(predict("Area man thrilled to pay taxes again"))
    print(predict("Scientists discover water on Mars"))
    print(predict(
        "\nOh great, another bug",
        context="I have been debugging for 5 hours"
    ))
