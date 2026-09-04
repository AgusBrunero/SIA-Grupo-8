"""Genera imágenes target simples, para no depender de descargas.

    python images/generate_samples.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
SIZE = 256


def japan() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(img)
    r = SIZE * 0.3
    d.ellipse([SIZE / 2 - r, SIZE / 2 - r, SIZE / 2 + r, SIZE / 2 + r], fill=(188, 0, 45))
    return img


def germany() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(img)
    for i, color in enumerate([(0, 0, 0), (221, 0, 0), (255, 206, 0)]):
        d.rectangle([0, i * SIZE / 3, SIZE, (i + 1) * SIZE / 3], fill=color)
    return img


def cross() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(img)
    w = SIZE * 0.18
    d.rectangle([SIZE / 2 - w, SIZE * 0.15, SIZE / 2 + w, SIZE * 0.85], fill=(200, 0, 0))
    d.rectangle([SIZE * 0.15, SIZE / 2 - w, SIZE * 0.85, SIZE / 2 + w], fill=(200, 0, 0))
    return img


if __name__ == "__main__":
    for name, fn in {"japan": japan, "germany": germany, "cross": cross}.items():
        path = HERE / f"{name}.png"
        fn().save(path)
        print(f"escrito {path}")
