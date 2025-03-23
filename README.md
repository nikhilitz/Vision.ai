# Vision.ai

**Vision.ai** is an AI-powered assistive technology designed for visually impaired individuals. It generates real-time captions for uploaded images using state-of-the-art Transformer architectures and converts them into speech, enabling users to understand their surroundings effortlessly.

## 🚀 Features

- **Real-Time Image Captioning**  
  Uses **CNN + Transformer architecture** for accurate and context-aware caption generation.

- **Speech Output (Text-to-Speech)**  
  Converts generated captions into speech using `pyttsx3` for enhanced accessibility.

- **AI-Driven Language Enhancement**  
  Natural Language Processing techniques refine the output for grammatical and contextual correctness.

- **User-Friendly Interface**  
  Simple, intuitive image upload and output interface designed for accessibility.

## 🛠️ Tech Stack

- **Deep Learning**
  - CNN (Image Feature Extraction)
  - Transformer (Caption Generation)
  - GAN (Optional) for refinement/enhancement

- **Machine Learning**
  - Model tuning and caption optimization

- **Natural Language Processing (NLP)**
  - Grammar and context correction for fluent captions

- **Text-to-Speech (TTS)**
  - `pyttsx3` for speech synthesis

- **Frontend**
  - Streamlit-based User Interface

- **Backend**
  - Python (Modular Structure)

- **Libraries & Tools**
  - PyTorch, OpenCV, NumPy, scikit-learn, NLTK, torchvision, pyttsx3, transformers

## 📂 Project Structure

\`\`\`
Vision.ai/
├── Data/
│   └── captions/
│   └── images/
├── src/
│   └── captioning/
│       ├── transforms.py
│       ├── utils.py
│       ├── vocabulary.py
├── test/
│   ├── test_transforms.py
│   ├── test_utils.py
│   └── testVocab.py
├── train/
│   └── train_captioning.py
├── config/
│   ├── config.py
│   └── logger.py
├── models/
│   └── encoder_decoder.py
├── frontend/
│   └── streamlit_app.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── Docs/
├── requirements.txt
├── .gitignore
├── README.md
\`\`\`

## ✔️ Completed Modules

- `transforms.py` – Image preprocessing  
- `utils.py` – Caption cleaning & loading  
- `vocabulary.py` – Vocabulary creation, stoi/itos mapping, numericalization  
- `test_transforms.py` – Unit tests for transforms  
- `test_utils.py` – Unit tests for utils  
- `testVocab.py` – Unit tests for vocabulary module

## 🧪 Testing

All core modules are tested using `unittest`.

Run tests individually:
\`\`\`bash
python -m unittest test/test_transforms.py
python -m unittest test/test_utils.py
python -m unittest test/testVocab.py
\`\`\`

Or run all tests (optional if `pytest` installed):
\`\`\`bash
pytest test/
\`\`\`

## 📦 Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/nikhilitz/Vision.ai.git
cd Vision.ai
\`\`\`

2. Create and activate virtual environment:
\`\`\`bash
python -m venv imgcap
source imgcap/bin/activate  # On Windows: imgcap\Scripts\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. Download NLTK tokenizer data (once only):
\`\`\`bash
python -m nltk.downloader punkt
\`\`\`

## 📊 Upcoming Work

- `dataset.py` – PyTorch Dataset for images + captions  
- `encoder_decoder.py` – CNN + Transformer model  
- `train_captioning.py` – Training loop  
- `evaluate.py` – BLEU score evaluation  
- `inference.py` – Caption generation from uploaded image  
- `tts_engine.py` – Convert text to speech  
- `translator.py` – Optional caption translation  
- `streamlit_app.py` – Frontend UI  
- `Dockerfile` – Containerization

## 📬 Contact

**Author:** Nikhil Gupta  
GitHub: https://github.com/nikhilitz

## 📝 License

MIT License – feel free to use, contribute, or modify.
