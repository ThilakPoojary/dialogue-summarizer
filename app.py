import streamlit as st
import re
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.set_page_config(page_title="Dialogue Summarizer", page_icon="💬", layout="centered")

st.markdown("""
<style>
    .title-block {
        background: linear-gradient(135deg, #1B2A4A, #2E5FA3);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .title-block h1 { color: white; margin: 0; font-size: 2rem; }
    .title-block p  { color: #ccd9f0; margin: 0.4rem 0 0 0; font-size: 0.95rem; }
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: white;
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 0.78rem;
        margin: 6px 4px 0 0;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .summary-box {
        background: #eaf4ea;
        border-left: 5px solid #2e8b57;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 1.05rem;
        color: #1a1a1a;
        margin-top: 0.5rem;
    }
    .stat-card {
        background: white;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-top: 0.5rem;
    }
    .stat-card .number { font-size: 1.6rem; font-weight: 700; color: #2E5FA3; }
    .stat-card .label  { font-size: 0.75rem; color: #888; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-block">
    <h1>💬 Dialogue Summarizer</h1>
    <p>Powered by fine-tuned <strong>T5-small</strong> trained on the SAMSum dataset</p>
    <div style="margin-top:10px">
        <span class="badge">🤗 Transformers</span>
        <span class="badge">T5-small</span>
        <span class="badge">SAMSum Dataset</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Same clean_data as notebook ──────────────────────────────────────
def clean_data(text):
    text = re.sub(r"[\r\n]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\d]", " ", text)
    text = text.strip().lower()
    return text

# ── Load model ───────────────────────────────────────────────────────
MODEL_PATH = "./final_summary_model"

@st.cache_resource(show_spinner=False)
def load_model():
    # Always load tokenizer from t5-small hub — avoids the "not a string"
    # bug caused by corrupted tokenizer_config.json in saved folders
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Load your fine-tuned weights from the saved folder
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device

if not os.path.isdir(MODEL_PATH):
    st.error(f"❌ Folder `{MODEL_PATH}` not found next to app.py")
    st.code(
        "# Run these two lines at the end of your notebook:\n"
        "model.save_pretrained('./final_summary_model')\n"
        "Tokenizer.save_pretrained('./final_summary_model')"
    )
    st.stop()

with st.spinner("⏳ Loading model..."):
    try:
        model, tokenizer, device = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

st.success(f"✅ Model ready | Device: **{device.upper()}**")

# ── Summarize ────────────────────────────────────────────────────────
def summarize(dialogue: str, num_beams: int, max_len: int) -> str:
    cleaned = clean_data(dialogue)
    inputs  = tokenizer(
        cleaned,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            num_beams=num_beams,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ── Examples ─────────────────────────────────────────────────────────
EXAMPLES = {
    "📚 Exam Schedule": (
        "A: Our exam schedule was announced today.\n"
        "B: Really? When is the NLP exam?\n"
        "A: It is next Monday morning.\n"
        "B: Then we should start revising from tomorrow itself."
    ),
    "🍕 Food Order": (
        "A: Hey, are you coming to the office today?\n"
        "B: Yes, around noon. Should I bring lunch?\n"
        "A: That would be great! Can you get some pizza?\n"
        "B: Sure, how many boxes?\n"
        "A: Three should be enough for everyone.\n"
        "B: Got it, I will order before I leave."
    ),
    "✈️ Travel Plans": (
        "A: I booked the flights for our trip to Goa!\n"
        "B: Amazing! When do we leave?\n"
        "A: This Friday evening at 7 PM.\n"
        "B: Perfect. Did you also book the hotel?\n"
        "A: Yes, beachside resort for 3 nights.\n"
        "B: Can't wait! I will start packing tonight."
    ),
}

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    num_beams = st.slider("Beam Search Width", 1, 8, 4,
                          help="Higher = better quality, slightly slower")
    max_len   = st.slider("Max Summary Length (tokens)", 30, 200, 150, step=10)
    st.divider()
    st.header("📋 Load Example")
    for label, text in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["dialogue_input"] = text
            st.rerun()
    st.divider()
    st.caption("T5-small · SAMSum · Streamlit")

# ── Input area ───────────────────────────────────────────────────────
st.subheader("📝 Enter Dialogue")
dialogue = st.text_area(
    "Paste a conversation below:",
    value=st.session_state.get("dialogue_input", ""),
    height=220,
    placeholder="A: Hi!\nB: Hey, what's up?\nA: ...",
    key="dialogue_input"
)

col_btn, col_clear, col_wc = st.columns([2, 1, 1])
with col_btn:
    run = st.button("✨ Summarize", type="primary", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["dialogue_input"] = ""
        st.rerun()
with col_wc:
    wc = len(dialogue.split()) if dialogue.strip() else 0
    st.markdown(f"""<div class="stat-card">
        <div class="number">{wc}</div>
        <div class="label">words</div>
    </div>""", unsafe_allow_html=True)

# ── Output ───────────────────────────────────────────────────────────
if run:
    if not dialogue.strip():
        st.warning("⚠️ Please enter a dialogue first.")
    else:
        with st.spinner("Generating summary..."):
            result = summarize(dialogue, num_beams, max_len)

        st.subheader("📄 Summary")
        st.markdown(f'<div class="summary-box">📌 {result}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        iw = len(dialogue.split())
        sw = len(result.split())
        cp = round((1 - sw / max(iw, 1)) * 100)
        for col, num, label in [(c1, iw, "Input Words"), (c2, sw, "Summary Words"), (c3, f"{cp}%", "Compression")]:
            with col:
                st.markdown(f"""<div class="stat-card">
                    <div class="number">{num}</div>
                    <div class="label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download Summary",
            data=f"DIALOGUE:\n{dialogue}\n\nSUMMARY:\n{result}",
            file_name="summary.txt",
            mime="text/plain"
        )
