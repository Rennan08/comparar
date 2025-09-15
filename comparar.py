# -*- coding: utf-8 -*-
"""
comparar.py
Autor: Você
Descrição: compara duas imagens (1:1) e decide se representam a mesma pessoa,
usando embeddings ArcFace (InsightFace) e similaridade por cosseno.

Requisitos (Python 3.11):
  - insightface==0.7.3
  - onnxruntime==1.17.3  (ou onnxruntime-gpu==1.17.3 se usar GPU NVIDIA)
  - opencv-python==4.10.0.84
  - numpy (instalado como dependência)

Uso:
  python comparar.py IMG1 IMG2 [--thr 0.60] [--ctx -1] [--det-size 640 640]
                          [--no-show] [--save SAIDA.jpg] [--model buffalo_l]

Observação operacional:
  - O limiar (--thr) deve ser calibrado localmente (ROC/FAR/FRR) para o seu AO.
  - ctx_id: -1=CPU, 0=GPU. Ajuste conforme disponibilidade/Política de TI.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import insightface


# ---------------------------- Utilidades ------------------------------------ #
def _select_primary_face(faces: list, strategy: str = "largest") -> "insightface.app.common.Face":
    """
    Seleciona um rosto quando há múltiplos detectados.
    strategy:
      - "largest": maior área de bounding-box (robusto para close-ups)
      - "bestscore": maior det_score do detector
    """
    if not faces:
        raise ValueError("Nenhum rosto detectado.")
    if strategy == "bestscore":
        return max(faces, key=lambda f: getattr(f, "det_score", 0.0))
    # padrão: largest
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return float(max(0, x2 - x1) * max(0, y2 - y1))
    return max(faces, key=area)


def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Similaridade por cosseno entre dois vetores (embeddings).
    As embeddings do InsightFace (normed_embedding) já vêm normalizadas (||v||=1),
    então o produto interno equivale ao cosseno.
    """
    # Garantir float32 para estabilidade / compatibilidade ONNX
    emb1 = emb1.astype(np.float32, copy=False)
    emb2 = emb2.astype(np.float32, copy=False)
    return float(np.dot(emb1, emb2))


def _draw_result(img1: np.ndarray, face1, img2: np.ndarray, face2, text: str) -> np.ndarray:
    """
    Desenha bounding boxes e texto consolidado; retorna imagem lado a lado.
    """
    for face, img in [(face1, img1), (face2, img2)]:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    h = max(img1.shape[0], img2.shape[0])
    w = img1.shape[1] + img2.shape[1]
    combined = np.zeros((h, w, 3), dtype=np.uint8)
    combined[: img1.shape[0], : img1.shape[1]] = img1
    combined[: img2.shape[0], img1.shape[1] :] = img2

    cv2.putText(
        combined,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0) if "MESMA" in text else (0, 0, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return combined


# ---------------------------- Núcleo ---------------------------------------- #
def comparar_fotos(
    path1: Path,
    path2: Path,
    app: "insightface.app.FaceAnalysis",
    threshold: float = 0.60,
    pick_strategy: str = "largest",
    show: bool = True,
    save_path: Path | None = None,
) -> Tuple[float, bool]:
    """
    Executa a comparação 1:1. Retorna (similaridade, mesma_pessoa).
    """
    # Carregar imagens
    img1 = cv2.imread(str(path1))
    img2 = cv2.imread(str(path2))
    if img1 is None:
        raise FileNotFoundError(f"Imagem inválida ou inacessível: {path1}")
    if img2 is None:
        raise FileNotFoundError(f"Imagem inválida ou inacessível: {path2}")

    # Detecção + extração
    faces1 = app.get(img1)
    faces2 = app.get(img2)
    if len(faces1) == 0 or len(faces2) == 0:
        raise ValueError("Não foi possível detectar rosto em uma das imagens.")

    # Seleciona um rosto por imagem
    face1 = _select_primary_face(faces1, strategy=pick_strategy)
    face2 = _select_primary_face(faces2, strategy=pick_strategy)

    emb1 = face1.normed_embedding
    emb2 = face2.normed_embedding

    sim = _cosine_similarity(emb1, emb2)
    same = sim >= threshold

    # Visualização / Exportação
    if show or save_path is not None:
        status = "MESMA PESSOA" if same else "PESSOAS DIFERENTES"
        text = f"Similaridade (cos): {sim:.3f} | Limiar: {threshold:.2f} -> {status}"
        frame = _draw_result(img1.copy(), face1, img2.copy(), face2, text)
        if show:
            cv2.imshow("Comparação de Rostos", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), frame)

    return sim, same


def build_app(
    model_name: str = "buffalo_l",
    ctx_id: int = -1,
    det_size: Tuple[int, int] = (640, 640),
) -> "insightface.app.FaceAnalysis":
    """
    Inicializa o pipeline de análise (detector + alinhamento + rec/embedding).
    ctx_id: -1=CPU, 0=GPU (NVIDIA), etc.
    det_size: tamanho alvo para o detector (robustez em imagens variadas).
    """
    app = insightface.app.FaceAnalysis(name=model_name)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


# ---------------------------- CLI ------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="comparar.py",
        description="Comparação 1:1 de rostos (ArcFace/InsightFace) com similaridade por cosseno.",
    )
    p.add_argument("img1", type=Path, help="Caminho da primeira imagem")
    p.add_argument("img2", type=Path, help="Caminho da segunda imagem")
    p.add_argument("--thr", type=float, default=0.60, help="Limiar de decisão (default: 0.60)")
    p.add_argument("--ctx", type=int, default=-1, help="Contexto: -1=CPU, 0=GPU (default: -1)")
    p.add_argument("--det-size", type=int, nargs=2, default=(640, 640), metavar=("W", "H"),
                   help="Tamanho do detector (default: 640 640)")
    p.add_argument("--model", type=str, default="buffalo_l",
                   help='Modelo do zoo (default: "buffalo_l")')
    p.add_argument("--no-show", action="store_true", help="Não abrir janela gráfica")
    p.add_argument("--save", type=Path, default=None, help="Salvar imagem comparativa em arquivo")
    p.add_argument("--pick", type=str, default="largest", choices=["largest", "bestscore"],
                   help='Critério de seleção de rosto quando houver múltiplos (default: "largest")')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Valida caminhos
    for p in (args.img1, args.img2):
        if not p.is_file():
            print(f"[ERRO] Arquivo não encontrado: {p}")
            return 2

    try:
        app = build_app(model_name=args.model, ctx_id=args.ctx, det_size=tuple(args.det_size))
        sim, same = comparar_fotos(
            path1=args.img1,
            path2=args.img2,
            app=app,
            threshold=args.thr,
            pick_strategy=args.pick,
            show=not args.no_show,
            save_path=args.save,
        )
        status = "MESMA PESSOA" if same else "PESSOAS DIFERENTES"
        print(f"Similaridade (cos): {sim:.4f} | limiar={args.thr:.2f} -> {status}")
        return 0 if same else 1  # opcional: código de saída semântica
    except Exception as e:
        print(f"[EXCEÇÃO] {type(e).__name__}: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
