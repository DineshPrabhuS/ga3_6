"""
photo_to_face_glb.py — turn a single face photo into a textured 3D face mesh (.glb)

Pipeline: MediaPipe Face Mesh (468 3D landmarks) -> Delaunay triangulation
-> photo projected as texture -> GLB, auto-shrunk to stay under 2 MB.

Setup (Python 3.9–3.12):
    pip install "mediapipe==0.10.14" opencv-python trimesh scipy pillow

Usage:
    python photo_to_face_glb.py photo.jpg face.glb
"""
import io
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import Delaunay

MAX_BYTES = 2_000_000


def landmarks(img_bgr):
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False
    ) as fm:
        res = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        sys.exit("No face detected — try a larger / sharper crop of the face.")
    h, w = img_bgr.shape[:2]
    lm = res.multi_face_landmarks[0].landmark
    # pixel coords; MediaPipe z is roughly in the same scale as x (image width)
    return np.array([[p.x * w, p.y * h, p.z * w] for p in lm])


def build_glb(img_path, tex_size):
    bgr = cv2.imread(img_path)
    if bgr is None:
        sys.exit(f"Could not read {img_path}")
    pts = landmarks(bgr)

    # --- crop texture to the face region (with margin) ---
    h, w = bgr.shape[:2]
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    m = 0.05 * max(x1 - x0, y1 - y0)
    x0, y0 = max(int(x0 - m), 0), max(int(y0 - m), 0)
    x1, y1 = min(int(x1 + m), w), min(int(y1 + m), h)
    crop = Image.fromarray(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    crop.thumbnail((tex_size, tex_size), Image.LANCZOS)

    uv = np.stack(
        [(pts[:, 0] - x0) / (x1 - x0), 1.0 - (pts[:, 1] - y0) / (y1 - y0)], axis=1
    )

    # --- geometry: centre, flip to Y-up / +Z toward viewer, scale to ~0.2 m tall ---
    v = pts.copy()
    v[:, 1] *= -1
    v[:, 2] *= -1
    v -= v.mean(axis=0)
    v *= 0.2 / (v[:, 1].max() - v[:, 1].min())

    # Triangulate in the (x, y) plane -> CCW winding, front faces point +Z
    faces = Delaunay(v[:, :2]).simplices

    # drop long sliver triangles along the convex hull
    e = np.linalg.norm(v[faces] - v[np.roll(faces, 1, axis=1)], axis=2).max(axis=1)
    faces = faces[e < np.percentile(e, 99) * 1.5]

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=crop,
        metallicFactor=0.0,
        roughnessFactor=0.85,
        doubleSided=True,
    )
    mesh = trimesh.Trimesh(
        vertices=v,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
        process=False,
    )
    # a few rounds of smoothing on z only would distort UVs; keep landmarks exact
    return trimesh.Scene(mesh).export(file_type="glb")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "face.glb"

    for tex in (2048, 1536, 1024, 768, 512, 384):
        data = build_glb(src, tex)
        if len(data) < MAX_BYTES:
            break
    with open(out, "wb") as f:
        f.write(data)
    size = os.path.getsize(out)
    print(f"Wrote {out}: {size:,} bytes (texture <= {tex}px)",
          "OK" if size < MAX_BYTES else "STILL TOO BIG")


if __name__ == "__main__":
    main()