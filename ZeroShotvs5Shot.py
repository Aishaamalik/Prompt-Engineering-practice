# save as sentiment_shot_demo.py
import os
import time
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from groq import Groq

# Load API key
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.7

# -------------------------
# Utility Functions
# -------------------------

def canonical_label(raw: str):
    """
    Normalize model outputs into standard labels.
    """
    s = raw.strip().lower()
    if "positive" in s:
        return "Positive"
    if "negative" in s:
        return "Negative"
    if "neutral" in s:
        return "Neutral"
    if s in ["pos", "p", "+"]:
        return "Positive"
    if s in ["neg", "n", "-"]:
        return "Negative"
    return None  # if uncertain

# Zero-shot instruction
ZERO_SHOT_INSTRUCTION = (
    "You are a helpful sentiment labeling assistant.\n"
    "Classify the sentiment of the sentence into exactly one of: Positive, Negative, Neutral.\n"
    "Return ONLY the label (one word): Positive, Negative, or Neutral.\n\n"
)

# Few-shot examples
FIVE_SHOT_EXAMPLES = [
    ('I love the battery life on this phone, it lasts all day!', 'Positive'),
    ('The waiter ignored us and the food was cold.', 'Negative'),
    ('The movie was okay, not bad but not great either.', 'Neutral'),
    ('What a fantastic performance—absolutely stunning!', 'Positive'),
    ('My package arrived later than promised and the box was damaged.', 'Negative'),
]

def build_five_shot_prompt(sentence: str) -> str:
    """
    Build a 5-shot prompt with labeled examples.
    """
    lines = [ZERO_SHOT_INSTRUCTION, "Examples:"]
    for i, (s, label) in enumerate(FIVE_SHOT_EXAMPLES, start=1):
        lines.append(f"{i}. \"{s}\" -> {label}")
    lines.append("\nNow classify the following sentence in the same format:\n")
    lines.append(f"Sentence: \"{sentence}\"\nLabel:")
    return "\n".join(lines)

def build_zero_shot_prompt(sentence: str) -> str:
    """
    Build a zero-shot prompt (instructions only, no examples).
    """
    return ZERO_SHOT_INSTRUCTION + f"Sentence: \"{sentence}\"\nLabel:"

def query_model(prompt: str, max_tokens=6) -> str:
    """
    Query the Groq model with a given prompt.
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        n=1,
    )
    return resp.choices[0].message.content.strip()

# -------------------------
# Experiment Setup
# -------------------------

TEST_DATA = [
    ("I had a great time at the restaurant last night.", "Positive"),
    ("The app keeps crashing every time I try to open it.", "Negative"),
    ("It was an average experience.", "Neutral"),
    ("Amazing customer support — helped me solve the issue quickly!", "Positive"),
    ("I wouldn't buy this product again.", "Negative"),
    ("The package was delivered on time.", "Neutral"),
    ("The plot was predictable and boring.", "Negative"),
    ("Wow, that concert was the best night of my life!", "Positive"),
    ("It does what it says.", "Neutral"),
    ("The trainer was unhelpful and rude.", "Negative"),
]

# -------------------------
# Main Experiment
# -------------------------

def run_experiment():
    zs_preds, five_preds = [], []
    gold = [label for (_, label) in TEST_DATA]

    print("\n=== Running Sentiment Classification Experiment ===\n")

    for sentence, _ in TEST_DATA:
        # Zero-shot
        z_prompt = build_zero_shot_prompt(sentence)
        raw_z = query_model(z_prompt)
        print(f"[Zero-Shot RAW] {sentence} -> {raw_z}")
        z_label = canonical_label(raw_z) or raw_z
        zs_preds.append(z_label)
        time.sleep(0.3)

        # Five-shot
        f_prompt = build_five_shot_prompt(sentence)
        raw_f = query_model(f_prompt)
        print(f"[Five-Shot RAW] {sentence} -> {raw_f}")
        f_label = canonical_label(raw_f) or raw_f
        five_preds.append(f_label)
        time.sleep(0.3)

    # Results
    print("\n--- Zero-Shot Results ---")
    print(classification_report(gold, zs_preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(gold, zs_preds, labels=["Positive", "Neutral", "Negative"]))

    print("\n--- Five-Shot Results ---")
    print(classification_report(gold, five_preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(gold, five_preds, labels=["Positive", "Neutral", "Negative"]))

    # Visualization
    plot_results(gold, zs_preds, five_preds)

# -------------------------
# Visualization
# -------------------------

def plot_results(gold, zs_preds, five_preds):
    """
    Compare Zero-shot vs Few-shot with accuracy & F1-score bar charts.
    """
    methods = ["Zero-Shot", "Five-Shot"]

    accs = [
        accuracy_score(gold, zs_preds),
        accuracy_score(gold, five_preds),
    ]

    from sklearn.metrics import f1_score
    f1s = [
        f1_score(gold, zs_preds, average="macro"),
        f1_score(gold, five_preds, average="macro"),
    ]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    rects1 = ax.bar(x - width/2, accs, width, label="Accuracy")
    rects2 = ax.bar(x + width/2, f1s, width, label="F1-score")

    ax.set_ylabel("Score")
    ax.set_title("Zero-Shot vs Five-Shot Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    # Label bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

# -------------------------
# Entry Point
# -------------------------

if __name__ == "__main__":
    run_experiment()
