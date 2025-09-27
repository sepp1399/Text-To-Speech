# TTS Training & Service

This project contains two main components:

1. **Training** (`TTS_training.py`)  
   Script for fine-tuning a **CSM (Conditional Speech Model)** using the [Common Voice Italian dataset](https://commonvoice.mozilla.org/it).  
   The model is trained to generate speech from text and saves both the fine-tuned weights and processor.

2. **API Service** (`TTS_service.py`)  
   A **FastAPI**-based REST service exposing a `/tts` endpoint to perform Text-to-Speech conversion.  
   It loads a pre-trained model and returns generated speech as a **WAV** audio file.

---

## 🚀 Requirements

- Python >= 3.9
- CUDA (optional, recommended for faster inference/training)
- Main dependencies:
  - `transformers`
  - `torch`
  - `datasets`
  - `soundfile`
  - `fastapi`
  - `uvicorn`
  - `python-dotenv`

Install dependencies with:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch transformers datasets soundfile fastapi uvicorn python-dotenv
```

---

## 📘 Model Training

To start training with the **Common Voice IT** dataset:

```bash
python TTS_training.py
```

This script will:
- Load the `sesame/csm-1b` model
- Fine-tune it on the Italian dataset
- Save the model and processor under:
  ```
  commonvoice-it-finetuned-model/
  commonvoice-it-finetuned-processor/
  ```

---

## 🎙️ Running the TTS Service

The FastAPI service is defined in `TTS_service.py`.

Run it locally with:

```bash
uvicorn TTS_service:app --host 0.0.0.0 --port 8000 --reload
```

- The `--reload` option is useful in development (auto-reload).  
- The API will be available at:  
  👉 [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

---

## 🔊 API Usage

### Endpoint

#### `POST /tts`  
Convert text into speech.

**Request body (JSON):**
```json
{
  "text": "Hello, how are you?"
}
```

**Response:**  
A WAV file containing the generated voice.

Test with `curl`:

```bash
curl -X POST "http://localhost:8000/tts"      -H "Content-Type: application/json"      -d '{"text":"Hello, how are you?"}'      --output output.wav
```

This will save the generated speech into `output.wav`.

---

## ⚙️ Environment Variables

The service uses a middleware session secret loaded from `.env`.  
Create a `.env` file in the project root:

```
MIDDELWARE_SECRET=supersecret123
```

---

## 📂 Project Structure

```
.
├── TTS_training.py         # Training script
├── TTS_service.py          # FastAPI TTS service
├── requirements.txt        # Dependencies (optional)
├── .env                    # Secret configuration
└── README.md               # This file
```
