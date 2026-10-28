import os
import cv2
import numpy as np
import threading
from insightface.app import FaceAnalysis
import onnxruntime as ort
import config

class IdentityService:
    def __init__(self):
        self.lock = threading.Lock()
        self.active_providers = self._get_providers()
        
        self.face_app = FaceAnalysis(name=config.FACE_MODEL_NAME, providers=self.active_providers)
        ctx_id = 0 if "CUDAExecutionProvider" in self.active_providers else -1
        self.face_app.prepare(ctx_id=ctx_id, det_thresh=0.50, det_size=(640, 640))
        
        self.site_databases = self._load_db()

        self.identity_calls = 0
        self.last_error = None

    def _get_providers(self):
        try:
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            return providers
        except Exception:
            return ["CPUExecutionProvider"]

    def _normalize_rows(self, embeddings):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return (embeddings / norms).astype(np.float32, copy=False)

    def _normalize_embedding(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm < 1e-12:
            return embedding
        return (embedding / norm).astype(np.float32, copy=False)

    def _load_site_db(self, path):
        if not os.path.exists(path):
            print(f"[IDENTITY] Database not found at {path}")
            return np.empty((0,), dtype=object), np.empty((0, 512), dtype=np.float32)
        try:
            data = np.load(path, allow_pickle=True)
            employee_ids = data.get("employee_ids", data.get("ids", []))
            embeddings = data.get("embeddings", data.get("features", []))
            embeddings = self._normalize_rows(embeddings)
            print(f"[IDENTITY] Loaded {len(employee_ids)} identities from {path}.")
            return np.asarray(employee_ids, dtype=object), embeddings
        except Exception as e:
            print(f"[IDENTITY] Failed to load database: {e}")
            return np.empty((0,), dtype=object), np.empty((0, 512), dtype=np.float32)

    def _load_db(self):
        return {
            "site_1_kolkata": self._load_site_db(config.KOLKATA_IDENTITY_DB),
            "site_2_pune": self._load_site_db(config.PUNE_IDENTITY_DB),
        }

    @staticmethod
    def _site_for_camera(camera_id):
        if camera_id.startswith("S1_CAM_"):
            return "site_1_kolkata"
        if camera_id.startswith("PUNE_CAM_"):
            return "site_2_pune"
        raise ValueError(f"Unsupported camera ID for identity routing: {camera_id}")

    def _match_db(self, normalized_emb, ids, embeddings):
        if embeddings is None or len(embeddings) == 0:
            return None, 0.0
        scores = np.dot(embeddings, normalized_emb)
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        
        # Matches against configured strictness threshold
        if best_score >= config.SIMILARITY_THRESHOLD:
            return str(ids[best_index]), best_score
        return None, best_score

    def identify(self, frame, camera_id, bbox=None):
        site = self._site_for_camera(camera_id)
        with self.lock:
            try:
                self.identity_calls += 1
                self.last_error = None
                faces = self.face_app.get(frame)
                results = []
                for face in faces:
                    embedding = getattr(face, "embedding", None)
                    if embedding is None:
                        continue
                    
                    norm_emb = self._normalize_embedding(embedding)
                    
                    employee_ids, embeddings = self.site_databases[site]
                    emp_id, score = self._match_db(norm_emb, employee_ids, embeddings)
                    
                    if emp_id:
                        results.append({"identity": emp_id, "status": "KNOWN", "confidence": score, "bbox": face.bbox})
                    else:
                        # Unknown is a classification of this observation, not a
                        # new person. Track-level deduplication belongs to the
                        # surveillance engine.
                        results.append({"identity": None, "status": "UNKNOWN", "confidence": score, "bbox": face.bbox})
                return results
            except Exception as e:
                self.last_error = str(e)
                print(f"[IDENTITY] Recognition unavailable: {e}")
                if "cuda" in str(e).lower() and "CUDAExecutionProvider" in self.active_providers:
                    self.active_providers = ["CPUExecutionProvider"]
                    self.face_app = FaceAnalysis(name=config.FACE_MODEL_NAME, providers=self.active_providers)
                    self.face_app.prepare(ctx_id=-1, det_thresh=0.50, det_size=(640, 640))
                return None

    def identify_tracks(self, frame, camera_id, tracks):
        site = self._site_for_camera(camera_id)
        employee_ids, embeddings = self.site_databases[site]
        height, width = frame.shape[:2]
        results = {}

        with self.lock:
            try:
                self.identity_calls += 1
                self.last_error = None
                for track_id, box in tracks:
                    x1, y1, x2, y2 = (int(value) for value in box)
                    box_width = max(1, x2 - x1)
                    box_height = max(1, y2 - y1)
                    crop = frame[
                        max(0, y1 - box_height // 20):min(
                            height, y1 + int(box_height * 0.40)
                        ),
                        max(0, x1 - box_width // 8):min(
                            width, x2 + box_width // 8
                        ),
                    ]
                    if crop.size == 0:
                        results[str(track_id)] = {"state": "NO_FACE"}
                        continue

                    if crop.shape[0] < 200:
                        scale = 200 / crop.shape[0]
                        crop = cv2.resize(
                            crop, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC,
                        )

                    faces = self.face_app.get(crop)
                    if not faces:
                        results[str(track_id)] = {"state": "NO_FACE"}
                        continue

                    face = max(
                        faces,
                        key=lambda candidate: (
                            (candidate.bbox[2] - candidate.bbox[0])
                            * (candidate.bbox[3] - candidate.bbox[1])
                        ),
                    )
                    embedding = getattr(face, "embedding", None)
                    if embedding is None:
                        results[str(track_id)] = {"state": "NO_FACE"}
                        continue

                    employee_id, score = self._match_db(
                        self._normalize_embedding(embedding),
                        employee_ids,
                        embeddings,
                    )
                    state = (
                        "KNOWN" if employee_id else
                        "AMBIGUOUS"
                        if score >= config.UNKNOWN_SIMILARITY_THRESHOLD
                        else "UNKNOWN"
                    )
                    results[str(track_id)] = {
                        "state": state,
                        "name": employee_id,
                        "conf": score,
                    }
                return results
            except Exception as e:
                self.last_error = str(e)
                print(f"[IDENTITY] Track recognition unavailable: {e}")
                if "cuda" in str(e).lower() and "CUDAExecutionProvider" in self.active_providers:
                    self.active_providers = ["CPUExecutionProvider"]
                    self.face_app = FaceAnalysis(
                        name=config.FACE_MODEL_NAME,
                        providers=self.active_providers,
                    )
                    self.face_app.prepare(
                        ctx_id=-1, det_thresh=0.50, det_size=(640, 640)
                    )
                return None