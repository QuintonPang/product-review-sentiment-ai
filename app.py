import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "quintonpyx/distilbert-amazon-polarity"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def analyze_sentiment(user_text):
    if not user_text.strip():
        return {"Positive 😊": 0.0, "Negative 😞": 0.0}, "⚠️ Please enter a review."

    inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    pos = float(probs[1])
    neg = float(probs[0])

    if pos > 0.9:
        verdict = "😍 Strongly Positive"
    elif pos > 0.6:
        verdict = "🙂 Mildly Positive"
    elif neg > 0.9:
        verdict = "😡 Strongly Negative"
    elif neg > 0.6:
        verdict = "😕 Mildly Negative"
    else:
        verdict = "😐 Mixed / Neutral"

    return {"Positive 😊": pos, "Negative 😞": neg}, verdict

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Type or paste a product review here...",
        label="Product Review"
    ),
    outputs=[
        gr.Label(num_top_classes=2, label="Sentiment Scores"),
        gr.Textbox(label="Verdict")
    ],
    title="🛍️ AI Product Review Sentiment Analyzer",
    description="Fine-tuned DistilBERT on Amazon product reviews.",
    examples=[
        ["This phone case is amazing! Fits perfectly and looks great."],
        ["Terrible quality, broke within a week. Waste of money."],
        ["It arrived on time but the color was slightly different from the photo."],
        ["Best headphones I've ever owned. Crystal clear sound and great battery life."],
        ["Stopped working after 3 days. Customer service was no help at all."]
    ],
)

if __name__ == "__main__":
    demo.launch(
        share=True,
        # flagging_mode="never",  # moved here from Interface
        theme=gr.themes.Soft()  # moved here from Interface
    )