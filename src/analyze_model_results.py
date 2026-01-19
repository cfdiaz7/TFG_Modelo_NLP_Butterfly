import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter

# Paths y configuraciones
MODEL_PATH = "./models/final_model"
DATASET_PATH = r"C:\Users\imlud\PycharmProjects\TFG_Modelo_NLP\data\DS_ABL_BUTTERFLY_V2"
BATCH_SIZE = 32  # Ajusta según tu RAM o GPU

label_map = {0: "Ant", 1: "Bee", 2: "Leech", 3: "Butterfly"}

# Detectar si hay GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Using device: {device}")

# Cargar modelo y tokenizer
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Cargar dataset de test
dataset = load_from_disk(DATASET_PATH)
test_dataset = dataset["test"]

print(f"✅ Test samples loaded: {len(test_dataset)}")

# DataLoader para batches
dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Listas para métricas
y_true = []
y_pred = []

# Iterar por batches
for batch in dataloader:
    texts = batch["text"]
    labels = batch["labels"]

    # Tokenizar batch
    inputs = tokenizer(list(texts), return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inferencia
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)

    y_true.extend(labels)
    y_pred.extend(preds.cpu().tolist())

from datasets import load_from_disk
ds = load_from_disk(r"C:\Users\imlud\PycharmProjects\TFG_Modelo_NLP\data\DS_ABL_BUTTERFLY_V2")
print(ds)

train = ds["train"]
test = ds["test"]

print("Train size:", len(train))
print("Test size:", len(test))

print("Butterfly en train:", sum(1 for x in train if x["labels"] == 3))
print("Butterfly en test:", sum(1 for x in test if x["labels"] == 3))


# Métricas y análisis
print("\n📈 Classification Report:")
report = classification_report(y_true, y_pred, target_names=label_map.values(), digits=3)
print(report)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.values(),
            yticklabels=label_map.values())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – ABL + Butterfly Model")
plt.tight_layout()
plt.show()

# Accuracy global
acc = np.trace(cm) / np.sum(cm)
print(f"✅ Global Accuracy: {acc:.3f}")

