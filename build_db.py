import argparse
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import config

IDENTITY_DATABASES = {
    "site_1_kolkata": config.KOLKATA_IDENTITY_DB,
    "site_2_pune": config.PUNE_IDENTITY_DB,
}

def normalize(embedding):
    norm = np.linalg.norm(embedding)
    return (embedding / norm).astype(np.float32) if norm >= 1e-12 else embedding

def build_identity_database(image_dir, output_path):
    print(f"Loading Face Model to build identity database from {image_dir}...")
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_thresh=0.5)

    employee_ids, embeddings = [], []

    for emp_id in os.listdir(image_dir):
        emp_folder = os.path.join(image_dir, emp_id)
        if not os.path.isdir(emp_folder): continue

        for filename in os.listdir(emp_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(emp_folder, filename))
                if img is None: continue
                
                faces = app.get(img)
                if faces:
                    best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    employee_ids.append(emp_id)
                    embeddings.append(normalize(best_face.embedding))
                    print(f"Processed: {emp_id}")

    if not embeddings:
        print("No faces processed.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, employee_ids=np.array(employee_ids, dtype=object), embeddings=np.array(embeddings, dtype=np.float32))
    print(f"Identity database saved to {output_path} with {len(embeddings)} identities.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("site", choices=IDENTITY_DATABASES)
    parser.add_argument("image_dir")
    arguments = parser.parse_args()
    build_identity_database(
        arguments.image_dir, IDENTITY_DATABASES[arguments.site]
    )