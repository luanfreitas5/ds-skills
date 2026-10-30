"""Template: servir um classificador de imagens (preprocessamento idêntico, limites, ONNX Runtime).

Executa de ponta a ponta com dados sintéticos: ``python serving-vision-models_template.py``.
Verificado com: onnxruntime 1.31.1, torch/torchvision (CPU), Pillow, opencv-python-headless,
fastapi + httpx + python-multipart, Python 3.12.
"""
from __future__ import annotations

import io
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.testclient import TestClient
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

logger = logging.getLogger("serving_vision")

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 25_000_000  # limite de decodificação (bomba de descompressão)
MAX_BATCH = 16
Image.MAX_IMAGE_PIXELS = MAX_PIXELS  # Pillow levanta DecompressionBombError acima de 2x


def preprocess_pil(image: Image.Image) -> np.ndarray:
    """Replica o pipeline de treino (PIL, bilinear, RGB, ImageNet) sem torchvision.

    Parameters
    ----------
    image : PIL.Image.Image
        Imagem de entrada, qualquer modo.

    Returns
    -------
    np.ndarray
        Tensor ``(3, SIZE, SIZE)`` float32.

    Examples
    --------
    >>> preprocess_pil(Image.new("RGB", (64, 48))).shape
    (3, 224, 224)
    """
    rgb = image.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    array = (np.asarray(rgb, dtype=np.float32) / 255.0 - MEAN) / STD
    return array.transpose(2, 0, 1)


def decode_upload(data: bytes) -> Image.Image:
    """Decodifica bytes em imagem validando tamanho e dimensões.

    Raises
    ------
    HTTPException
        413 se o arquivo exceder o limite; 400 se não for imagem válida.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Arquivo maior que o limite permitido.")
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_PIXELS:
            raise HTTPException(413, "Imagem com dimensões acima do limite.")
        image.load()
        return image
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, "Imagem com dimensões acima do limite.") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "Imagem inválida ou corrompida.") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cria a sessão ONNX uma única vez; falha na subida se o artefato faltar."""
    path = app.state.model_path
    app.state.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logger.info("Sessão ONNX carregada de %s", path)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
def predict(request: Request, file: UploadFile) -> dict:
    """Classifica uma imagem. ``def`` (não ``async``): roda no threadpool e não trava o loop."""
    declared = request.headers.get("content-length")
    if declared and int(declared) > MAX_UPLOAD_BYTES + 4096:
        raise HTTPException(413, "Arquivo maior que o limite permitido.")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    x = preprocess_pil(decode_upload(data))[None]
    logits = request.app.state.session.run(None, {"input": x})[0]
    return {"class": int(logits.argmax(axis=1)[0])}


# ----------------------------- demonstração com dados sintéticos -----------------------------
def _synthetic_png(width: int = 300, height: int = 200, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def main() -> None:
    """Exporta um ResNet18 aleatório, testa paridade, limites e latência."""
    torch.manual_seed(0)
    model = models.resnet18(num_classes=5).eval()
    onnx_path = Path(tempfile.mkdtemp()) / "m.onnx"
    torch.onnx.export(model, torch.randn(2, 3, SIZE, SIZE), str(onnx_path), input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "b"}, "logits": {0: "b"}},
                      opset_version=17, dynamo=False)
    app.state.model_path = onnx_path

    train_tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                                   transforms.Normalize(MEAN.tolist(), STD.tolist())])
    png = _synthetic_png()
    img = Image.open(io.BytesIO(png))

    # 1) paridade de preprocessamento: serving == treino (torchvision/PIL)
    diff = np.abs(preprocess_pil(img) - train_tf(img.convert("RGB")).numpy()).max()
    print(f"paridade preprocess max|diff| = {diff:.2e}")
    assert diff < 1e-4

    # 2) cv2 diverge do PIL mesmo com 'o mesmo' resize bilinear
    import cv2
    bgr = cv2.resize(cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    cv_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    pil_rgb = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
    print(f"cv2 vs PIL resize max|diff| = {np.abs(cv_rgb - pil_rgb).max():.0f} (de 255)")

    with TestClient(app) as client:
        # 3) paridade de ponta a ponta: API == modelo torch com o pipeline de treino
        r = client.post("/predict", files={"file": ("a.png", png, "image/png")})
        assert r.status_code == 200, r.text
        with torch.no_grad():
            expected = int(model(train_tf(img.convert("RGB")).unsqueeze(0)).argmax())
        assert r.json()["class"] == expected
        # 4) limites
        big = b"0" * (MAX_UPLOAD_BYTES + 10)
        assert client.post("/predict", files={"file": ("a.png", big, "image/png")}).status_code == 413
        assert client.post("/predict", files={"file": ("a.png", b"nao-e-imagem", "image/png")}).status_code == 400
        bomb = io.BytesIO()
        Image.new("1", (20000, 20000)).save(bomb, "PNG")  # poucos KB, 400M pixels
        assert client.post("/predict", files={"file": ("a.png", bomb.getvalue(), "image/png")}).status_code == 413
        # 5) latência (aquecida, p50/p99)
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            client.post("/predict", files={"file": ("a.png", png, "image/png")})
            times.append((time.perf_counter() - t0) * 1000)
        print(f"latência p50={np.percentile(times, 50):.1f} ms p99={np.percentile(times, 99):.1f} ms")
    print("ok")


if __name__ == "__main__":
    main()
