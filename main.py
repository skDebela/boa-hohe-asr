import logging
import os
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import AutoModelForCTC, AutoProcessor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('hohe-asr')
MODEL_ID = os.getenv('MODEL_ID', 'snapwre/hohe-asr-amharic')
REVISION = os.getenv('MODEL_REVISION', 'main')
MAX_SECONDS = int(os.getenv('MAX_AUDIO_SECONDS', '60'))
MAX_BYTES = int(os.getenv('MAX_UPLOAD_MB', '25')) * 1024 * 1024
DTYPE_NAME = os.getenv('MODEL_DTYPE', 'float32')
DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16}
lock = threading.Lock()
processor = model = None
ready = False


def infer(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(device='cuda:0', dtype=DTYPES[DTYPE_NAME] if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
    with torch.inference_mode():
        ids = model(**inputs).logits.argmax(dim=-1).cpu()
    return processor.batch_decode(ids)[0].replace('[AMH]', '').strip()


@asynccontextmanager
async def lifespan(app):
    global model, processor, ready
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. CPU fallback is disabled.')
    if DTYPE_NAME not in DTYPES:
        raise RuntimeError('MODEL_DTYPE must be float32 or bfloat16')
    if DTYPE_NAME == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support bfloat16')
    torch.set_num_threads(2)
    # Prefer the complete local snapshot. Network is used only on a cache miss.
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        path = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModelForCTC.from_pretrained(path, local_files_only=True, torch_dtype=DTYPES[DTYPE_NAME])
    except (LocalEntryNotFoundError, OSError):
        path = snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'], ignore_patterns=['lm/*', 'eval/*'])
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModelForCTC.from_pretrained(path, local_files_only=True, torch_dtype=DTYPES[DTYPE_NAME])
    model = model.to('cuda:0').eval()
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError('All model parameters must be on CUDA')
    for _ in range(2):
        infer(np.zeros(16000, dtype=np.float32))
    torch.cuda.synchronize()
    ready = True
    log.info('READY: %s on %s; dtype=%s; snapshot=%s', MODEL_ID, torch.cuda.get_device_name(0), DTYPE_NAME, path)
    yield
    ready = False


app = FastAPI(title='Hohe Amharic ASR', lifespan=lifespan)


@app.get('/health')
def health():
    if not ready:
        raise HTTPException(503, 'Model is not ready')
    return {'status': 'ready', 'model': MODEL_ID, 'device': 'cuda:0', 'gpu': torch.cuda.get_device_name(0), 'dtype': DTYPE_NAME, 'decoder': 'greedy'}


@app.post('/transcribe')
def transcribe(audio: UploadFile = File(...)):
    if not ready:
        raise HTTPException(503, 'Model is not ready')
    if not lock.acquire(blocking=False):
        raise HTTPException(429, 'GPU is busy; retry shortly', headers={'Retry-After': '1'})
    start = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix='.audio') as tmp:
            size = 0
            while chunk := audio.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, 'Audio upload is too large')
                tmp.write(chunk)
            if size == 0:
                raise HTTPException(400, 'Empty audio upload')
            tmp.flush()
            try:
                result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', tmp.name, '-t', str(MAX_SECONDS + 1), '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'], capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                raise HTTPException(422, 'Audio decoding timed out')
        if result.returncode:
            raise HTTPException(422, 'Cannot decode this audio file')
        samples = np.frombuffer(result.stdout, dtype='<f4').copy()
        if len(samples) > MAX_SECONDS * 16000:
            raise HTTPException(413, f'Maximum audio duration is {MAX_SECONDS} seconds')
        if len(samples) < 1600 or not np.isfinite(samples).all():
            raise HTTPException(422, 'Audio must contain at least 0.1 seconds of valid samples')
        text = infer(samples)
        return {'text': text, 'language': 'am', 'duration_seconds': round(len(samples) / 16000, 3), 'processing_seconds': round(time.perf_counter() - start, 3), 'device': 'cuda:0'}
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(503, 'Insufficient GPU memory; retry with shorter audio')
    finally:
        audio.file.close()
        lock.release()
