from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import os
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
import re
from kokoro import KPipeline
import logging
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from rvc_python.infer import RVCInference
import io
import soundfile as sf
import librosa  # Per resampling


load_dotenv()

RVC_SAMPLE_RATE = 48000

# -----------------------------
# Configurazione Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# -----------------------------
# FastAPI + session middleware
# -----------------------------
secret_key = os.getenv("MIDDELWARE_SECRET", "changeme")
app = FastAPI(title="KOKORO TTS API")
app.add_middleware(SessionMiddleware, secret_key=secret_key)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Impostazioni dispositivo
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32


os.environ["XDG_CACHE_HOME"] = "/data/eleanor/samuel"
os.environ['HF_HOME'] = '/data/eleanor/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/data/eleanor/cache'
os.environ["HF_HOME"] = "/data/eleanor/cache"


# -----------------------------
# Modello Kokoro
# -----------------------------
pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code='it')


# -----------------------------
# RVC Setup (Elodie Voice)
# -----------------------------
RVC_MODELS_DIR = "/data/eleanor/cache/rvc_models"
voice = "caterina"
rvc_model_path = os.path.join(RVC_MODELS_DIR, voice, f"{voice}.pth")
rvc_index_path = os.path.join(RVC_MODELS_DIR, voice,  f"{voice}.index")

# -----------------------------
# Schema richiesta
# -----------------------------
class TTSRequest(BaseModel):
    prompt: str


# Mappa delle pause (in secondi) per tipo di punteggiatura
PAUSE_MAP = {
    ',': 0,
    ';': 0.1,
    ':': 0.1,
    '.': 0.2,
    '?': 0.15,
    '!': 0.15,
    '…': 0.2,
    '-': 0.05,
}


# Pre-generazione silenzi per punteggiature (48kHz per compatibilità RVC)
PREGENERATED_SILENCE = {p: np.zeros(int(RVC_SAMPLE_RATE * d), dtype=np.float32).tobytes()
                        for p, d in PAUSE_MAP.items() if d > 0}


# Funzione per spezzare testo mantenendo punteggiatura
def split_text_with_punctuation_successivo(text: str, min_words=3):
    """
    Divide il testo in segmenti mantenendo la punteggiatura.
    Accorpa segmenti troppo brevi (<=min_words) al successivo.
    
    Args:
        text: testo da dividere
        min_words: numero minimo di parole per segmento (default: 3)
    
    Returns:
        Lista di tuple (segmento, punteggiatura)
    """
    pattern = r'([^,;:.!?…\-]+)([,:;.!?…\-]?)'
    raw_segments = re.findall(pattern, text)
    
    # Filtra segmenti vuoti
    raw_segments = [(s.strip(), p) for s, p in raw_segments if s.strip()]
    
    # Accorpa segmenti brevi
    merged_segments = []
    buffer = ""
    buffer_punct = ""
    
    for i, (segment, punct) in enumerate(raw_segments):
        # Conta parole nel segmento corrente
        word_count = len(segment.split())
        
        # Aggiungi al buffer
        if buffer:
            buffer += buffer_punct + " " + segment
        else:
            buffer = segment
        buffer_punct = punct
        
        # Conta parole totali nel buffer
        total_words = len(buffer.split())
        
        # Decide se mantenere o continuare ad accorpare
        is_last = (i == len(raw_segments) - 1)
        
        if total_words > min_words or is_last:
            # Buffer sufficientemente lungo o ultimo segmento
            merged_segments.append((buffer, buffer_punct))
            buffer = ""
            buffer_punct = ""
    
    return merged_segments

def split_text_with_punctuation_pre1(text: str, min_words: int = 3):
    """
    Divide il testo in segmenti mantenendo la punteggiatura.
    Accorpa segmenti con ≤ min_words parole al segmento precedente.
    
    Args:
        text: Testo da dividere
        min_words: Numero minimo di parole per segmento (default: 3)
    
    Returns:
        Lista di tuple (segmento, punteggiatura)
    """
    pattern = r'([^,;:.!?…\-]+)([,:;.!?…\-]?)'
    raw_segments = re.findall(pattern, text)
    
    # Filtra segmenti vuoti
    raw_segments = [(seg.strip(), punct) for seg, punct in raw_segments if seg.strip()]
    
    if not raw_segments:
        return []
    
    merged_segments = []
    current_segment = ""
    current_punct = ""
    
    for segment, punct in raw_segments:
        # Conta le parole nel segmento corrente
        word_count = len(segment.split())
        
        if word_count <= min_words and current_segment:
            # Accorpa al segmento precedente
            # Mantieni la punteggiatura precedente se presente
            if current_punct:
                current_segment += current_punct + " " + segment
            else:
                current_segment += " " + segment
            current_punct = punct  # Usa la punteggiatura del nuovo segmento
        else:
            # Salva il segmento precedente se esiste
            if current_segment:
                merged_segments.append((current_segment, current_punct))
            
            # Inizia un nuovo segmento
            current_segment = segment
            current_punct = punct
    
    # Aggiungi l'ultimo segmento
    if current_segment:
        merged_segments.append((current_segment, current_punct))
    
    return merged_segments

def split_text_with_punctuation(text: str, min_words: int = 6):
    """
    Divide il testo in segmenti mantenendo la punteggiatura.
    Accorpa segmenti brevi (≤ min_words) al precedente, 
    tranne se separati da interpunzione forte (. ! ? …).
    
    Args:
        text: Testo da segmentare
        min_words: Numero minimo di parole per segmento (default: 3)
    
    Returns:
        Lista di tuple (testo, punteggiatura)
    """
    # Pattern originale
    pattern = r'([^,;:.!?…\-]+)([,:;.!?…\-]?)'
    segments = re.findall(pattern, text)
    
    # Segni di interpunzione forte che NON permettono l'accorpamento
    strong_punctuation = {'.', '!', '?', '…'}
    
    # Lista risultante con accorpamenti
    merged_segments = []
    
    for segment_text, punct in segments:
        # Rimuovi spazi e conta le parole
        words = segment_text.strip().split()
        num_words = len(words)
        
        # Se il segmento è vuoto, saltalo
        if not segment_text.strip():
            continue
        
        # Se è il primo segmento o se il precedente aveva interpunzione forte
        # oppure se ha abbastanza parole, aggiungilo normalmente
        if (not merged_segments or 
            merged_segments[-1][1] in strong_punctuation or 
            num_words > min_words):
            merged_segments.append((segment_text, punct))
        
        # Altrimenti, accorpalo al precedente
        else:
            prev_text, prev_punct = merged_segments[-1]
            # Unisci mantenendo la punteggiatura debole del precedente
            merged_text = prev_text + prev_punct + segment_text
            # La punteggiatura finale è quella del segmento corrente
            merged_segments[-1] = (merged_text, punct)
    
    return merged_segments



## Parametri per la voce
RVC_SAMPLE_RATE = 48000
speed = 1.02
f0up_key=0               # Abbassa di 2 semitoni (pitch)
index_rate=1.0             # MASSIMA similarità al clone italiano
filter_radius=7            # Smoothing moderato per naturalezza
resample_sr=0              # Mantieni sample rate originale
rms_mix_rate=7          # Usa dinamica vocale del clone italiano
protect=0.5


# Durata minima per RVC (evita chunk troppo corti che causano errori)
MIN_AUDIO_DURATION = 0.5  # secondi

# -----------------------------
# Inizializza RVC con parametri espliciti
# -----------------------------
rvc = None
try:
    rvc = RVCInference(device=device)
    
    if os.path.exists(rvc_model_path):
        rvc.load_model(rvc_model_path)
        
        # Usa set_params per i parametri supportati dalla classe
        rvc.set_params(
            f0method="rmvpe",           # Migliore per parlato italiano naturale
            f0up_key=12,               # Abbassa di 2 semitoni (pitch)
            index_rate=1,             # MASSIMA similarità al clone italiano
            filter_radius=7,            # Smoothing moderato per naturalezza
            resample_sr=0,              # Mantieni sample rate originale
            rms_mix_rate=1,          # Usa dinamica vocale del clone italiano
            protect=0.5                 # MINIMA protezione voce originale
        )
        
        logging.info("Modello RVC caricato con parametri ottimizzati")
        
    else:
        logging.warning(f"Modello RVC non trovato in {rvc_model_path}")
        rvc = None
except Exception as e:
    logging.error(f"Errore inizializzazione RVC: {e}")
    rvc = None

# -----------------------------
# Funzione conversione RVC migliorata
# -----------------------------
def convert_with_rvc(audio_array: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Converte l'audio usando RVC con gestione robusta degli errori"""
    
    # Calcola durata audio
    duration = len(audio_array) / sample_rate
    
    # Se l'audio è troppo corto, salta RVC (evita tensor mismatch)
    if duration < MIN_AUDIO_DURATION:
        logging.info(f"Audio troppo corto ({duration:.2f}s), skip RVC")
        if sample_rate != RVC_SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=RVC_SAMPLE_RATE)
        return audio_array.astype(np.float32).tobytes()
    
    if rvc is None:
        # Fallback: resample senza RVC
        if sample_rate != RVC_SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=RVC_SAMPLE_RATE)
        return audio_array.astype(np.float32).tobytes()
    
    try:
        # Resample a 48kHz prima di RVC
        if sample_rate != RVC_SAMPLE_RATE:
            audio_resampled = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=RVC_SAMPLE_RATE)
        else:
            audio_resampled = audio_array
        
        # File temporanei unici
        temp_input = f"/data/eleanor/cache/tmp/rvc_input_{os.getpid()}_{id(audio_array)}_{int(duration*1000)}.wav"
        temp_output = f"/data/eleanor/cache/tmp/rvc_output_{os.getpid()}_{id(audio_array)}_{int(duration*1000)}.wav"
        
        # Salva input
        sf.write(temp_input, audio_resampled, RVC_SAMPLE_RATE, format='WAV')
        
        # Verifica file input creato
        if not os.path.exists(temp_input) or os.path.getsize(temp_input) == 0:
            raise ValueError(f"Input file non valido: {temp_input}")
        
        # Applica RVC
        logging.debug(f"RVC processing: duration={duration:.2f}s, samples={len(audio_resampled)}")
        
        rvc.infer_file(
            input_path=temp_input,
            output_path=temp_output
        )
        
        # Verifica che il file output esista e sia valido
        if not os.path.exists(temp_output):
            logging.error(f"RVC non ha generato output: {temp_output}")
            raise FileNotFoundError(f"Output file not created")
        
        if os.path.getsize(temp_output) == 0:
            logging.error(f"File output RVC vuoto")
            raise ValueError("RVC generated empty output file")
        
        # Leggi output con gestione sicura della tupla
        result = sf.read(temp_output, dtype='float32')
        
        if isinstance(result, tuple):
            converted_audio, sr = result
            logging.info(f"RVC OK: shape={converted_audio.shape}, sr={sr}, duration={duration:.2f}s")
        else:
            # Fallback improbabile
            converted_audio = result
            logging.warning(f"sf.read returned unexpected type: {type(result)}")
        
        # Verifica validità
        if not isinstance(converted_audio, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(converted_audio)}")
        
        if len(converted_audio) == 0:
            raise ValueError("Converted audio is empty")
        
        # Cleanup
        for temp_file in [temp_input, temp_output]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as cleanup_error:
                    logging.debug(f"Cleanup warning {temp_file}: {cleanup_error}")
        
        return converted_audio.astype(np.float32).tobytes()
    
    except Exception as e:
        logging.error(f"Errore conversione RVC (duration={duration:.2f}s): {e}", exc_info=True)
        
        # Cleanup su errore
        for temp_file in [temp_input, temp_output]:
            try:
                if 'temp_input' in locals() and os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        
        # Fallback: ritorna audio resampled senza RVC
        if sample_rate != RVC_SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=RVC_SAMPLE_RATE)
        return audio_array.astype(np.float32).tobytes()
# -----------------------------
# Endpoint TTS con RVC
# -----------------------------
@app.post("/tts_fast")
def text_to_speech(request: TTSRequest):
    try:
        segments = split_text_with_punctuation(request.prompt)

        def audio_generator():
            for segment, punct in segments:
                if not segment.strip():
                    continue

                # Generazione audio TTS
                try:
                    gs, ps, audio = next(pipeline(segment.strip(), voice="im_nicola", speed=1.18))
                except Exception as e:
                    logging.error(f"Errore generazione audio per segmento '{segment}': {e}")
                    continue

                # Converti audio
                audio_np = audio.cpu().numpy()
                
                # Applica RVC per conversione vocale Elodie
                converted_audio = convert_with_rvc(audio_np, sample_rate=24000)
                
                yield converted_audio

        return StreamingResponse(audio_generator(), media_type="application/octet-stream")

    except Exception as e:
        logging.error(f"Errore endpoint /tts_fast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Endpoint senza RVC (fallback veloce)
# -----------------------------
@app.post("/tts_fast_no_rvc")
def text_to_speech_no_rvc(request: TTSRequest):
    try:
        segments = split_text_with_punctuation(request.prompt)

        def audio_generator():
            for segment, punct in segments:
                if not segment.strip():
                    continue

                # Yield silenzio pre-generato (24kHz per Kokoro)
                silence_24k = PAUSE_MAP.get(punct, 0)
                if silence_24k > 0:
                    silence_samples = np.zeros(int(24000 * silence_24k), dtype=np.float32)
                    yield silence_samples.tobytes()

                try:
                    gs, ps, audio = next(pipeline(segment.strip(), voice="if_sara"))
                except Exception as e:
                    logging.error(f"Errore generazione audio per segmento '{segment}': {e}")
                    continue

                yield audio.cpu().numpy().astype(np.float32).tobytes()

        return StreamingResponse(audio_generator(), media_type="application/octet-stream")

    except Exception as e:
        logging.error(f"Errore endpoint /tts_fast_no_rvc: {e}")
        raise HTTPException(status_code=500, detail=str(e))