"""Reconhecimento facial em fotos com OpenCV e DeepFace."""

import argparse
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
# Mantém os pesos no projeto, respeitando um cache configurado pelo usuário.
os.environ.setdefault("DEEPFACE_HOME", str(BASE))
# Os logs de download do DeepFace usam Unicode, inclusive no PowerShell.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

import cv2
from deepface import DeepFace


EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODELO = "Facenet512"


def imagens(pasta):
    return sorted(p for p in pasta.rglob("*") if p.suffix.lower() in EXTENSOES)


def detectar(imagem):
    # Retorna recortes BGR na escala 0..255, compatíveis com verify.
    return DeepFace.extract_faces(
        img_path=imagem,
        detector_backend="opencv",
        enforce_detection=True,
        align=True,
        color_face="bgr",
        normalize_face=False,
    )


def carregar_autorizados(pasta):
    referencias = []
    for caminho in imagens(pasta):
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            print(f"[AVISO] Não foi possível ler: {caminho}")
            continue
        try:
            rostos = detectar(imagem)
        except ValueError as erro:
            print(f"[AVISO] Cadastro ignorado em {caminho}: {erro}")
            continue
        if len(rostos) != 1:
            print(f"[AVISO] Cadastro deve conter exatamente um rosto: {caminho}")
            continue
        referencias.append((caminho.parent.name, rostos[0]["face"]))
    return referencias


def identificar(rosto, referencias):
    for nome, referencia in referencias:
        resultado = DeepFace.verify(
            img1_path=rosto,
            img2_path=referencia,
            model_name=MODELO,
            detector_backend="skip",
            enforce_detection=False,
            distance_metric="cosine",
        )
        # Usa o limiar calibrado pelo DeepFace para o modelo e a métrica.
        if resultado["verified"]:
            return nome
    return None


def desenhar(imagem, area, autorizado):
    altura, largura = imagem.shape[:2]
    x = max(0, int(area["x"]))
    y = max(0, int(area["y"]))
    x2 = min(largura - 1, int(area["x"] + area["w"]))
    y2 = min(altura - 1, int(area["y"] + area["h"]))
    cor = (0, 255, 0) if autorizado else (0, 0, 255)  # OpenCV usa BGR.
    texto = "acesso liberado" if autorizado else "acesso negado"
    cv2.rectangle(imagem, (x, y), (x2, y2), cor, 2)
    escala = min(0.65, max(0.15, (largura - 8) / 300))
    (texto_largura, texto_altura), _ = cv2.getTextSize(
        texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 1
    )
    tx = max(0, min(x, largura - texto_largura - 4))
    ty = min(altura - 3, max(texto_altura + 4, y - 8))
    cv2.rectangle(imagem, (tx, ty - texto_altura - 3),
                  (tx + texto_largura + 3, ty + 3), (0, 0, 0), -1)
    cv2.putText(imagem, texto, (tx + 1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                escala, cor, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autorizados", type=Path, default=BASE / "autorizados")
    parser.add_argument("--testes", type=Path, default=BASE / "testes")
    parser.add_argument("--saida", type=Path, default=BASE / "resultados")
    args = parser.parse_args()
    for pasta in (args.autorizados, args.testes):
        if not pasta.is_dir():
            parser.error(f"Pasta não encontrada: {pasta}")
    for origem in (args.autorizados.resolve(), args.testes.resolve()):
        destino = args.saida.resolve()
        if destino == origem or origem in destino.parents or destino in origem.parents:
            parser.error("A saída deve estar separada das pastas de entrada.")
    arquivos = imagens(args.testes)
    if not arquivos:
        print(f"Adicione fotos de teste em: {args.testes}")
        return
    referencias = carregar_autorizados(args.autorizados)
    if not referencias:
        print("[AVISO] Nenhum cadastro válido. Todos os rostos terão acesso negado.")
    for caminho in arquivos:
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            print(f"[AVISO] Imagem ilegível: {caminho}")
            continue
        try:
            rostos = detectar(imagem)
        except ValueError as erro:
            print(f"[AVISO] Sem rosto detectável em {caminho}: {erro}")
            # Sem rosto, marca a imagem inteira como acesso negado.
            rostos = []
            desenhar(imagem, {"x": 0, "y": 0, "w": imagem.shape[1],
                              "h": imagem.shape[0]}, False)
        for rosto in rostos:
            try:
                nome = identificar(rosto["face"], referencias)
            except Exception as erro:
                # Uma falha de comparação nunca libera acesso.
                print(f"[ERRO] Falha na comparação em {caminho}: {erro}")
                nome = None
            desenhar(imagem, rosto["facial_area"], nome is not None)
            print(f"{caminho.name}: " + (f"acesso liberado ({nome})" if nome else "acesso negado"))
        # Mantém subpastas e extensão original no nome para evitar colisões.
        destino = args.saida / caminho.relative_to(args.testes)
        destino = destino.with_name(destino.name + ".png")
        destino.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destino), imagem):
            raise OSError(f"Não foi possível salvar: {destino}")
        print(f"Resultado salvo em: {destino}")


if __name__ == "__main__":
    main()
