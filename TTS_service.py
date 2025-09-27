# llm-api/main.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor
import soundfile as sf
from io import BytesIO
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
secret_key = os.getenv("MIDDELWARE_SECRET")

app = FastAPI(title="TTS API")
app.add_middleware(SessionMiddleware, secret_key=secret_key)

processor = AutoProcessor.from_pretrained("cartesia/azzurra-voice")
model = CsmForConditionalGeneration.from_pretrained("cartesia/azzurra-voice").to(device)

class TTSRequest(BaseModel):
    text: str


@app.post("/tts")
def text_to_speech(request: TTSRequest):
    try:
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": request.text}]},
        ]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(device)
        audio_output = model.generate(**inputs, output_audio=True)
        waveform = audio_output[0].cpu().numpy()
        
        # Write to an in-memory buffer instead of disk
        buf = BytesIO()
        sf.write(buf, waveform, 24_000, format='WAV')
        buf.seek(0)
        
        # Option 1: StreamingResponse
        return StreamingResponse(buf, media_type="audio/wav")
        
        # To download the file instead of streaming, uncomment below:
        # headers = {"Content-Disposition": "attachment; filename=output.wav"}
        # return Response(content=buf.read(), media_type="audio/wav", headers=headers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))