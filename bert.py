import inspect
import json
import random

import matplotlib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 4
OUTPUT_DIR = "./results"
FINAL_MODEL_DIR = "final_model"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================
# LOAD DATA
# =========================
df = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)

df = df.rename(
    columns={
        "headline": "text",
        "is_sarcastic": "label",
    }
)

df = df[["text", "label"]].dropna()
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)

# Do not add previous headline as context. It is unrelated noise for this dataset.
train_df, eval_df = train_test_split(
    df,
    test_size=0.10,
    random_state=SEED,
    stratify=df["label"],
)

train_df = train_df.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)


# =========================
# DATASET
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
eval_dataset = Dataset.from_pandas(eval_df, preserve_index=False)

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# =========================
# MODEL
# =========================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


# =========================
# CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_df["label"].values,
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights:", class_weights.tolist())


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# =========================
# METRICS
# =========================
def metric_dict(labels, preds):
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return metric_dict(labels, preds)


# =========================
# TRAINING
# =========================
steps_per_epoch = int(np.ceil(len(train_dataset) / TRAIN_BATCH_SIZE))
warmup_steps = int(0.10 * steps_per_epoch * NUM_EPOCHS)

training_kwargs = {
    "output_dir": OUTPUT_DIR,
    "per_device_train_batch_size": TRAIN_BATCH_SIZE,
    "per_device_eval_batch_size": EVAL_BATCH_SIZE,
    "num_train_epochs": NUM_EPOCHS,
    "save_strategy": "epoch",
    "logging_strategy": "epoch",
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": warmup_steps,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_f1",
    "greater_is_better": True,
    "save_total_limit": 2,
    "report_to": "none",
    "seed": SEED,
}

training_args_params = inspect.signature(TrainingArguments).parameters
strategy_arg = "eval_strategy" if "eval_strategy" in training_args_params else "evaluation_strategy"
training_kwargs[strategy_arg] = "epoch"

training_args = TrainingArguments(**training_kwargs)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights,
)

trainer.train()


# =========================
# PREDICT ON EVAL SET
# =========================
predictions = trainer.predict(eval_dataset)

logits = predictions.predictions
true_labels = predictions.label_ids

probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
probs_positive = probs[:, 1]

default_preds = (probs_positive >= 0.50).astype(int)

precision_vals, recall_vals, pr_thresholds = precision_recall_curve(
    true_labels,
    probs_positive,
)
f1_vals = (
    2
    * precision_vals[:-1]
    * recall_vals[:-1]
    / (precision_vals[:-1] + recall_vals[:-1] + 1e-12)
)
best_idx = int(np.argmax(f1_vals))
best_threshold = float(pr_thresholds[best_idx])
best_preds = (probs_positive >= best_threshold).astype(int)

fpr, tpr, _ = roc_curve(true_labels, probs_positive)
roc_auc = float(auc(fpr, tpr))
pr_auc = float(average_precision_score(true_labels, probs_positive))

default_metrics = metric_dict(true_labels, default_preds)
best_metrics = metric_dict(true_labels, best_preds)


# =========================
# PRINT FINAL METRICS
# =========================
print("\n===== DEFAULT THRESHOLD 0.50 =====")
for name, value in default_metrics.items():
    print(f"{name.title()}: {value:.4f}")

print(f"\n===== BEST F1 THRESHOLD {best_threshold:.4f} =====")
for name, value in best_metrics.items():
    print(f"{name.title()}: {value:.4f}")

print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC: {pr_auc:.4f}")

print("\nClassification report at best threshold:")
print(
    classification_report(
        true_labels,
        best_preds,
        target_names=["not_sarcastic", "sarcastic"],
        zero_division=0,
    )
)


# =========================
# PLOTS
# =========================
def save_confusion_matrix(labels, preds, path, title):
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_labels = ["Not Sarcastic", "Sarcastic"]
    plt.xticks([0, 1], tick_labels, rotation=20)
    plt.yticks([0, 1], tick_labels)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


save_confusion_matrix(
    true_labels,
    default_preds,
    "confusion_matrix_default_threshold.png",
    "Confusion Matrix (Threshold 0.50)",
)
save_confusion_matrix(
    true_labels,
    best_preds,
    "confusion_matrix.png",
    f"Confusion Matrix (Best Threshold {best_threshold:.3f})",
)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals, label=f"PR AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("pr_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(pr_thresholds, f1_vals)
plt.axvline(best_threshold, linestyle="--", color="red", label=f"Best = {best_threshold:.3f}")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("Threshold vs F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig("threshold_vs_f1.png", dpi=200)
plt.close()

plt.figure(figsize=(6, 5))
plt.bar(
    ["Accuracy", "Precision", "Recall", "F1"],
    [
        best_metrics["accuracy"],
        best_metrics["precision"],
        best_metrics["recall"],
        best_metrics["f1"],
    ],
)
plt.ylim(0, 1)
plt.title("Model Performance (Best Threshold)")
plt.tight_layout()
plt.savefig("model_performance.png", dpi=200)
plt.close()

log_history = pd.DataFrame(trainer.state.log_history)
log_history.to_csv("training_log.csv", index=False)

plt.figure(figsize=(7, 5))
if "epoch" in log_history and "loss" in log_history:
    train_logs = log_history.dropna(subset=["loss"])
    plt.plot(train_logs["epoch"], train_logs["loss"], marker="o", label="Train loss")
if "epoch" in log_history and "eval_loss" in log_history:
    eval_logs = log_history.dropna(subset=["eval_loss"])
    plt.plot(eval_logs["epoch"], eval_logs["eval_loss"], marker="o", label="Eval loss")
if "epoch" in log_history and "eval_f1" in log_history:
    eval_logs = log_history.dropna(subset=["eval_f1"])
    plt.plot(eval_logs["epoch"], eval_logs["eval_f1"], marker="o", label="Eval F1")
plt.xlabel("Epoch")
plt.title("Training Curves")
plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=200)
plt.close()


# =========================
# SAVE
# =========================
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

final_report = {
    "model_name": MODEL_NAME,
    "seed": SEED,
    "max_length": MAX_LENGTH,
    "class_weights": class_weights.tolist(),
    "default_threshold": 0.50,
    "best_threshold": best_threshold,
    "default_threshold_metrics": default_metrics,
    "best_threshold_metrics": best_metrics,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "artifacts": [
        "confusion_matrix.png",
        "confusion_matrix_default_threshold.png",
        "roc_curve.png",
        "pr_curve.png",
        "threshold_vs_f1.png",
        "model_performance.png",
        "training_curves.png",
        "training_log.csv",
    ],
}

with open("final_metrics.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2)

with open(f"{FINAL_MODEL_DIR}/threshold.txt", "w", encoding="utf-8") as f:
    f.write(str(best_threshold))

print("\nTraining complete. Model saved to final_model.")
print("Saved: final_metrics.json, confusion_matrix.png, roc_curve.png, pr_curve.png")
