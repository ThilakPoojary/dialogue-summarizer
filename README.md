cat > README.md << 'EOF'
# 💬 Dialogue Summarizer — T5-small

A Streamlit web app that summarizes conversations using a fine-tuned **T5-small** model trained on the SAMSum dataset.

## Features
- Real-time dialogue summarization
- Adjustable beam search and summary length
- 3 built-in example dialogues
- Compression stats and download option

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model
Fine-tuned T5-small on SAMSum dataset using HuggingFace Transformers.
Place your `final_summary_model/` folder in the project root before running.

