import cv2
import time
import math
import numpy as np
import os
import csv
import threading
import queue
from datetime import datetime
from types import SimpleNamespace
from scipy.sparse.csgraph import connected_components
import scipy.spatial.distance as dist

import config


def _is_outside_office_hours(timestamp):
    current_time = datetime.fromtimestamp(timestamp).time()
    office_start = datetime.strptime(
        config.OFFICE_START_TIME, "%H:%M"
    ).time()
    office_end = datetime.strptime(
        config.OFFICE_END_TIME, "%H:%M"
    ).time()
    if office_start <= office_end:
        return not office_start <= current_time < office_end
    return not (current_time >= office_start or current_time < office_end)


def compute_dhash(image):
    """Return a 64-bit difference hash for an image."""
    resized = cv2.resize(image, (9, 8))
    if resized.ndim == 2:
        gray = resized
    elif resized.shape[2] == 4:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return (gray[:, 1:] > gray[:, :-1]).reshape(-1)


def hamming_distance(hash1, hash2):
    """Count differing bits between two dHash values."""
    return int(np.count_nonzero(hash1 != hash2))


class Track:
    """Represents a single tracked person over time."""
    def __init__(self, tid, uid, t, p):
        self.tid = str(tid)
        self.uid = uid
        self.first_seen = t
        self.last_seen = t
        self.box = None
        self.kp = None
        self.history = []       # [(cx, cy, t), ...]
        self.kp_history = []    # [(kp, t), ...]
        
        self.identity = SimpleNamespace(state="PENDING", name=None, confidence=0.0,
                                       obs=0, ambiguous_obs=0, unknown_obs=0,
                                       last_check=0, uid=uid)
        self.label = ""
        
        self.pose_state = "Unknown"
        self.state_since = t
        self.fall_candidate = False
        self.last_y = None
        self.reversals = 0
        self.last_dir = None
        self.activity_state = "STANDING"
        self.activity_since = t
        self.run_hits = 0
        self.crossing_id = None
        self.jump_frames = 0

class CameraState:
    """Isolated state for a single camera to ensure thread safety."""
    def __init__(self, cam_id, p):
        self.cam_id = cam_id
        self.p = p
        self.tracker = None
        self.tracks = {}
        self.active_track_ids = set()
        self.event_history = []
        self.occupancy = 0
        
        # UC7: Tampering State
        self.tamper = {"status": "OK", "baseline_gray": [], "baseline_des": [], "fail_start": 0}
        self.last_frame = None
        self.last_frame_mono = time.monotonic()
        self.person_boxes = []
        self.cover = 0.0
        
        # UC2/10: Identity
        self.id_results = []
        self.identity_calls = 0
        self.identity_results = 0
        self.expired_track_count = 0
        self.last_identity_debug = 0.0
        
        # UC5: Tailgating
        self.crossings = []
        self.next_crossing_id = 0
        
        # UC8: Fire
        self.fire_history = []
        self.fire_detections = []
        self.last_fire_check = 0.0
        self.crowd_since = 0.0
        self.crowd_active = False
        self.crowd_lost_since = 0.0
        
        # Threading & Perf
        self.lock = threading.Lock()
        self.q = queue.Queue(maxsize=config.INFERENCE_QUEUE_SIZE)
        self.running = True
        self.last_annotated_ts = 0.0
        self.live_overlay = {
            "tracks": [],
            "detections": [],
            "fire_detections": [],
            "alerts": [],
            "occupancy": 0,
        }
        
        self.frames_offered = 0
        self.frames_displayed = 0
        self.frames_inferred = 0
        self.frames_dropped_before_inference = 0
        self.frames_overwritten_for_latest = 0
        self.inference_fps = 0.0
        self.e2e_ms_avg = 0.0
        self.e2e_ms_p95 = 0.0
        self.capture_to_result_ms_avg = 0.0
        self.last_annotated_age_ms = 0.0
        self.latencies = []
        self.cap_latencies = []
        self.inference_ms = []
        
        self.cooldowns = {}
        self.last_saved_hashes = {}

class SurveillanceEngine:
    def __init__(self, identity_service, preload_models=False):
        self.identity = identity_service
        self.p = SimpleNamespace(
            PERF_LOG_INTERVAL_S=60,
            RESTRICTED_ZONES={},
            CAMERA_STALL_SECONDS=3.0,
            STREAM_GAP_RESET_S=2.0,
            TAILGATING_SECONDS=3.0,
            CROWD_MIN_PEOPLE=config.CROWD_MIN_PEOPLE,
            CROWD_PROXIMITY_THRESHOLD=config.CROWD_PROXIMITY_THRESHOLD,
            CROWD_CONFIRMATION_SECONDS=config.CROWD_CONFIRMATION_SECONDS,
            CROWD_RELEASE_SECONDS=config.CROWD_RELEASE_SECONDS,
        )
        self._cams = {}
        self._lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._work_event = threading.Event()
        self._worker_thread = None
        self._worker_running = False
        self._worker_error = None
        self.inference_errors = 0
        self._fire_lock = threading.Lock()
        self._after_hours = {}
        self._after_hours_lock = threading.Lock()

        from ultralytics import YOLO
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        import torch

        pose_model_path = os.path.join(config.MODELS_DIR, 'yolov8s-pose.pt')
        self._torch = torch
        self._boxes_class = Boxes
        self._tracker_class = BYTETracker
        self.detector_factory = lambda: YOLO(config.PERSON_DETECTION_MODEL)
        self.pose_factory = lambda cam: YOLO(pose_model_path)
        self.detector_model = None
        self.pose_model = None
        self._models_loaded = False
        self._fire_half = torch.cuda.is_available()
        
        try:
            fire_path = os.path.join(config.MODELS_DIR, 'fire_smoke.pt')
            self.fire_model = YOLO(fire_path) if os.path.exists(fire_path) else None
            self.fire_error = None if self.fire_model else "Model missing"
            if self.fire_model:
                self.fire_classes, self.fire_class_error = self._resolve_fire_classes(
                    self.fire_model.names, None)
            else:
                self.fire_classes, self.fire_class_error = None, self.fire_error
        except Exception as e:
            self.fire_model = None
            self.fire_error = str(e)
            self.fire_classes, self.fire_class_error = None, self.fire_error
            
        self.events_csv = os.path.join(config.LOGS_DIR, "events.csv")
        for site_dir in ("site 1 - kolkata", "site 2 - pune"):
            for sub in ["unknown", "restricted", "loitering", "tailgating", "crowding", "tampering", "fire", "abnormal", "prolonged_stay", "camera_status", "visibility"]:
                os.makedirs(os.path.join(config.EVIDENCE_DIR, site_dir, sub), exist_ok=True)
        if preload_models:
            self._load_inference_models()

    def _load_inference_models(self):
        with self._model_lock:
            if not self._models_loaded:
                detector = self.detector_factory()
                pose = self.pose_factory("shared")
                self.detector_model = detector
                self.pose_model = pose
                self._models_loaded = True

    def _start_worker(self):
        self._load_inference_models()
        with self._worker_lock:
            if self._worker_thread is None:
                self._worker_running = True
                self._worker_thread = threading.Thread(
                    target=self._worker,
                    name="SurveillanceInferenceWorker",
                    daemon=True,
                )
                self._worker_thread.start()

    @staticmethod
    def _resolve_fire_classes(classes_dict, override):
        if override:
            return override, None
        out = [k for k, v in classes_dict.items()
               if str(v).strip().lower() in ("fire", "smoke")]
        if not out:
            return None, f"Classes {classes_dict} do not contain fire/smoke"
        return sorted(out), None

    def _get_cam(self, cam):
        with self._lock:
            if cam not in self._cams:
                cs = CameraState(cam, self.p)
                tracker_args = SimpleNamespace(
                    track_high_thresh=config.PERSON_DETECTION_CONFIDENCE,
                    track_low_thresh=0.1,
                    new_track_thresh=config.PERSON_DETECTION_CONFIDENCE,
                    track_buffer=30,
                    match_thresh=config.PERSON_TRACK_MATCH_THRESHOLD,
                    fuse_score=True,
                )
                cs.tracker = self._tracker_class(tracker_args)
                self._cams[cam] = cs
            return self._cams[cam]

    def _add_event(self, cs, typ, desc, severity="WARNING", target=None):
        """Adds event to history, respects cooldowns, and writes to CSV."""
        event_description = desc
        if " (" in desc and desc.endswith("s)"):
            event_description, duration = desc.rsplit(" (", 1)
            if not duration[:-2].isdigit():
                event_description = desc
        event_key = (typ, str(target) if target is not None else "", event_description)
        now = time.time()
        if now - cs.cooldowns.get(event_key, 0) < 10.0: # 10s event cooldown
            return
        cs.cooldowns[event_key] = now
        
        ev = {"type": typ, "desc": desc, "severity": severity, "target": str(target) if target else None, "time": now}
        cs.event_history.append(ev)
        
        evidence_saved = False
        evidence_error = None
        path = ""
        site_dir = (
            "site 1 - kolkata" if cs.cam_id.startswith("S1_CAM_")
            else "site 2 - pune" if cs.cam_id.startswith("PUNE_CAM_")
            else None
        )
        hash_key = (typ, str(target) if target is not None else "GLOBAL")
        if cs.last_frame is None:
            evidence_error = "no event frame available"
        elif site_dir is None:
            evidence_error = f"unsupported camera ID for evidence routing: {cs.cam_id}"
        else:
            target_directory = os.path.join(
                config.EVIDENCE_DIR, site_dir, typ
            )
            path = os.path.join(
                target_directory, f"{cs.cam_id}_{typ}_{int(now * 1000)}.jpg"
            )
            try:
                os.makedirs(target_directory, exist_ok=True)
                evidence_frame = cs.last_frame.copy()
                banner_height = min(42, evidence_frame.shape[0])
                cv2.rectangle(
                    evidence_frame, (0, 0),
                    (evidence_frame.shape[1], banner_height),
                    (0, 0, 160), -1
                )
                alert_text = f"[{severity}] {typ.upper()} - {cs.cam_id} | {desc}"
                cv2.putText(
                    evidence_frame, alert_text,
                    (8, min(29, banner_height - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
                )

                if target is not None:
                    track = next(
                        (candidate for candidate in cs.tracks.values()
                         if candidate.tid == str(target) and candidate.box is not None),
                        None,
                    )
                    if track is not None:
                        x1, y1, x2, y2 = [int(value) for value in track.box]
                        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2),
                                      (0, 0, 255), 2)
                        activity = getattr(track, "activity_state", "")
                        posture = getattr(track, "pose_state", "")
                        state = " ".join(
                            value for value in (activity, posture) if value
                        )
                        tag = f"ID: {target} | {typ.upper()}"
                        if state:
                            tag = f"{tag} | {state}"
                        cv2.putText(
                            evidence_frame, tag,
                            (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                        )

                if typ == "fire":
                    for name, confidence, box in cs.fire_detections:
                        x1, y1, x2, y2 = [int(value) for value in box]
                        cv2.rectangle(
                            evidence_frame, (x1, y1), (x2, y2),
                            (0, 0, 255), 2
                        )
                        cv2.putText(
                            evidence_frame, f"{name.upper()} {confidence:.2f}",
                            (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 255), 2
                        )

                curr_hash = compute_dhash(evidence_frame)
                previous = cs.last_saved_hashes.get(hash_key)
                visual_duplicate = (
                    previous is not None
                    and 0 <= now - previous[0] <= 60.0
                    and hamming_distance(curr_hash, previous[1]) <= 4
                )
                if visual_duplicate:
                    path = ""
                    evidence_error = "visual duplicate suppressed"
                else:
                    evidence_saved = bool(cv2.imwrite(path, evidence_frame))
                    if evidence_saved:
                        cs.last_saved_hashes[hash_key] = (now, curr_hash)
                    else:
                        evidence_error = "cv2.imwrite returned False"
            except (OSError, cv2.error) as exc:
                evidence_error = str(exc)
            
        try:
            with open(self.events_csv, "a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([datetime.fromtimestamp(now).isoformat(), "SITE", cs.cam_id, typ, severity, target, desc, path])
        except OSError as exc:
            evidence_error = evidence_error or str(exc)
        ev["evidence_path"] = path
        ev["evidence_saved"] = evidence_saved
        if evidence_error:
            ev["evidence_error"] = evidence_error
            print(f"[EVIDENCE] Failed for {cs.cam_id}/{typ}: {evidence_error}")
        else:
            print(f"[EVENT] {datetime.fromtimestamp(now).isoformat()} "
                  f"{cs.cam_id} {typ} {ev['severity']} evidence={path}")

    @staticmethod
    def _tile_bounds(frame):
        height, width = frame.shape[:2]
        overlap_x = int(width * config.PERSON_DETECTION_TILE_OVERLAP / 2)
        overlap_y = int(height * config.PERSON_DETECTION_TILE_OVERLAP / 2)
        middle_x, middle_y = width // 2, height // 2
        return [
            (0, 0, min(width, middle_x + overlap_x),
             min(height, middle_y + overlap_y)),
            (max(0, middle_x - overlap_x), 0, width,
             min(height, middle_y + overlap_y)),
            (0, max(0, middle_y - overlap_y),
             min(width, middle_x + overlap_x), height),
            (max(0, middle_x - overlap_x),
             max(0, middle_y - overlap_y), width, height),
        ]

    @staticmethod
    def _merge_tile_detections(detections, iou_threshold=0.6):
        remaining = sorted(detections, key=lambda item: item[1], reverse=True)
        kept = []
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            retained = []
            for candidate in remaining:
                box_a, box_b = best[0], candidate[0]
                ix1 = max(box_a[0], box_b[0])
                iy1 = max(box_a[1], box_b[1])
                ix2 = min(box_a[2], box_b[2])
                iy2 = min(box_a[3], box_b[3])
                intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_a = max(0.0, box_a[2] - box_a[0]) * max(
                    0.0, box_a[3] - box_a[1]
                )
                area_b = max(0.0, box_b[2] - box_b[0]) * max(
                    0.0, box_b[3] - box_b[1]
                )
                union = area_a + area_b - intersection
                if union == 0.0 or intersection / union <= iou_threshold:
                    retained.append(candidate)
            remaining = retained
        return kept

    def _worker(self):
        """Run one batched detector worker while preserving per-camera trackers."""
        while self._worker_running:
            self._work_event.wait(timeout=0.1)
            if self._work_event.is_set():
                time.sleep(config.INFERENCE_BATCH_WINDOW_SECONDS)
            self._work_event.clear()
            with self._lock:
                camera_states = list(self._cams.values())

            pending = []
            now = time.monotonic()
            for cs in camera_states:
                try:
                    batch = cs.q.get_nowait()
                    pending.extend(
                        (cs, frame, timestamp, capture_ts)
                        for frame, timestamp, capture_ts in batch
                    )
                except queue.Empty:
                    if (
                        cs.running
                        and now - cs.last_frame_mono
                        > self.p.CAMERA_STALL_SECONDS
                    ):
                        self._add_event(
                            cs, "camera_status",
                            "Camera unavailable / stalled", "INFO",
                        )
                        cs.last_frame_mono = now

            if any(not cs.q.empty() for cs in camera_states):
                self._work_event.set()
            if not pending:
                continue

            try:
                self._process_inference_batch(pending)
                self._worker_error = None
            except Exception as error:
                self.inference_errors += 1
                self._worker_error = str(error)
                print(f"[INFERENCE ERROR] Shared worker failed: {error}")

    def _process_inference_batch(self, pending):
        frame_started = [time.perf_counter() for _ in pending]
        tile_sources = []
        tiles = []
        for job_index, (cs, frame, timestamp, _) in enumerate(pending):
            cs.last_frame_mono = time.monotonic()
            cs.last_frame = frame
            self._tamper_step(cs, frame, timestamp)
            for x1, y1, x2, y2 in self._tile_bounds(frame):
                tiles.append(frame[y1:y2, x1:x2])
                tile_sources.append((job_index, x1, y1))

        predictions = self.detector_model.predict(
            tiles,
            conf=0.25,
            iou=0.30,
            imgsz=config.PERSON_DETECTION_IMAGE_SIZE,
            classes=[0],
            device=config.INFERENCE_DEVICE,
            verbose=False,
        )
        if len(predictions) != len(tile_sources):
            raise RuntimeError(
                "Detector result count did not match the submitted tile count"
            )

        frame_detections = [[] for _ in pending]
        for (job_index, offset_x, offset_y), prediction in zip(
            tile_sources, predictions
        ):
            boxes = prediction.boxes
            if boxes is None or len(boxes) == 0:
                continue
            coordinates = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            frame_height, frame_width = pending[job_index][1].shape[:2]
            for box, confidence in zip(coordinates, confidences):
                full_box = np.asarray(box, dtype=np.float32).copy()
                full_box[[0, 2]] += offset_x
                full_box[[1, 3]] += offset_y
                full_box[[0, 2]] = np.clip(
                    full_box[[0, 2]], 0, frame_width
                )
                full_box[[1, 3]] = np.clip(
                    full_box[[1, 3]], 0, frame_height
                )
                if (
                    full_box[2] <= full_box[0]
                    or full_box[3] <= full_box[1]
                ):
                    continue
                frame_detections[job_index].append(
                    (full_box, float(confidence))
                )

        frame_tracks = []
        pose_inputs = []
        for job_index, (cs, frame, _, _) in enumerate(pending):
            detections = self._merge_tile_detections(
                frame_detections[job_index]
            )
            detection_boxes = np.asarray(
                [box for box, _ in detections], dtype=np.float32
            ).reshape(-1, 4)
            detection_data = np.column_stack((
                detection_boxes,
                np.asarray([score for _, score in detections], dtype=np.float32),
                np.zeros(len(detections), dtype=np.float32),
            ))
            tracker_input = self._boxes_class(
                self._torch.as_tensor(
                    detection_data, dtype=self._torch.float32
                ),
                frame.shape[:2],
            )
            tracked_rows = cs.tracker.update(tracker_input, frame)
            tracked_by_index = {}
            for row in tracked_rows:
                detection_index = int(row[7])
                if not 0 <= detection_index < len(detections):
                    continue
                track_id = str(int(row[4]))
                tracked_by_index[detection_index] = track_id
                x1, y1, x2, y2 = (int(value) for value in row[:4])
                box_height = y2 - y1
                if box_height < config.POSE_MIN_BOX_HEIGHT:
                    continue
                frame_height, frame_width = frame.shape[:2]
                pad_x = max(2, int((x2 - x1) * 0.05))
                pad_y = max(2, int(box_height * 0.05))
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(frame_width, x2 + pad_x)
                crop_y2 = min(frame_height, y2 + pad_y)
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size:
                    pose_inputs.append((
                        job_index, detection_index, crop, crop_x1, crop_y1
                    ))
            frame_tracks.append((detections, tracked_rows, tracked_by_index))

        keypoints_by_detection = [{} for _ in pending]
        if pose_inputs:
            pose_results = self.pose_model.predict(
                [item[2] for item in pose_inputs],
                conf=config.POSE_DETECTION_CONFIDENCE,
                imgsz=640,
                classes=[0],
                device=config.INFERENCE_DEVICE,
                verbose=False,
            )
            if len(pose_results) != len(pose_inputs):
                raise RuntimeError(
                    "Pose result count did not match the submitted person crop count"
                )
            for pose_input, prediction in zip(pose_inputs, pose_results):
                job_index, detection_index, _, offset_x, offset_y = pose_input
                if (
                    prediction.boxes is None
                    or prediction.keypoints is None
                    or len(prediction.boxes) == 0
                ):
                    continue
                best_index = int(
                    prediction.boxes.conf.argmax().item()
                )
                keypoints = (
                    prediction.keypoints.data[best_index]
                    .cpu().numpy().copy()
                )
                keypoints[:, 0] += offset_x
                keypoints[:, 1] += offset_y
                keypoints_by_detection[job_index][detection_index] = keypoints

        for job_index, (cs, frame, timestamp, capture_ts) in enumerate(pending):
            detections, tracked_rows, tracked_by_index = frame_tracks[job_index]
            display_detections = [
                {
                    "box": box.copy(),
                    "track_id": tracked_by_index.get(
                        index, f"temp_{index}"
                    ),
                }
                for index, (box, _) in enumerate(detections)
            ]
            tracked_detections = []
            for row in tracked_rows:
                detection_index = int(row[7])
                if not 0 <= detection_index < len(detections):
                    continue
                box = np.asarray(row[:4], dtype=np.float32)
                keypoints = keypoints_by_detection[job_index].get(
                    detection_index
                )
                if (
                    keypoints is not None
                    and int(np.count_nonzero(
                        keypoints[:, 2]
                        >= config.POSE_KEYPOINT_MIN_CONFIDENCE
                    )) < 4
                ):
                    keypoints = None
                tracked_detections.append({
                    "id": int(row[4]),
                    "box": box,
                    "kp": keypoints,
                })

            inference_time = max(
                0.001, time.perf_counter() - frame_started[job_index]
            )
            cs.inference_fps = 1.0 / inference_time
            cs.inference_ms.append(inference_time * 1000.0)
            if len(cs.inference_ms) > 30:
                cs.inference_ms.pop(0)

            fire_now = time.monotonic()
            if (
                self.fire_model
                and fire_now - cs.last_fire_check
                >= config.FIRE_INFERENCE_INTERVAL_SECONDS
            ):
                cs.last_fire_check = fire_now
                with self._fire_lock:
                    fire_options = {
                        "conf": config.FIRE_SMOKE_CONFIDENCE,
                        "imgsz": 640,
                        "verbose": False,
                    }
                    if self._fire_half:
                        fire_options["half"] = True
                    fire_result = self.fire_model.predict(
                        frame, **fire_options
                    )[0]
                fire_detections = []
                if fire_result.boxes is not None and len(fire_result.boxes):
                    for fire_box, class_id, confidence in zip(
                        fire_result.boxes.xyxy.cpu().numpy(),
                        fire_result.boxes.cls.cpu().numpy(),
                        fire_result.boxes.conf.cpu().numpy(),
                    ):
                        class_id = int(class_id)
                        if (
                            self.fire_classes is not None
                            and class_id in self.fire_classes
                        ):
                            label = str(
                                self.fire_model.names[class_id]
                            ).strip().lower()
                            fire_detections.append((
                                label, float(confidence), fire_box
                            ))
                cs.fire_detections = fire_detections
                self._ingest_fire(
                    cs, fire_detections, timestamp, frame
                )

            self._update_cam(
                cs,
                tracked_detections,
                timestamp,
                frame,
                occupancy_count=len(display_detections),
            )
            with cs.lock:
                cs.live_overlay = {
                    "tracks": [
                        {
                            "box": track.box.copy(),
                            "track_id": track.tid,
                            "identity_state": track.identity.state,
                            "identity": track.identity.name,
                            "posture": track.pose_state,
                            "activity": track.activity_state,
                        }
                        for track in cs.tracks.values()
                        if track.tid in {
                            str(track_id) for track_id in cs.active_track_ids
                        } and track.box is not None
                    ],
                    "detections": [
                        {
                            "box": detection["box"].copy(),
                            "track_id": detection["track_id"],
                        }
                        for detection in display_detections
                    ],
                    "fire_detections": [
                        (name, confidence, box.copy())
                        for name, confidence, box in cs.fire_detections
                    ],
                    "alerts": [
                        {
                            "severity": event["severity"],
                            "type": event["type"],
                            "desc": event["desc"],
                            "time": event["time"],
                        }
                        for event in cs.event_history[-5:]
                    ],
                    "occupancy": cs.occupancy,
                }
                cs.last_annotated_ts = timestamp

            cs.frames_inferred += 1
            cs.latencies.append((time.time() - timestamp) * 1000)
            if len(cs.latencies) > 30:
                cs.latencies.pop(0)
            cs.e2e_ms_avg = np.mean(cs.latencies)
            cs.e2e_ms_p95 = np.percentile(cs.latencies, 95)
            if capture_ts:
                cs.cap_latencies.append(
                    (time.time() - capture_ts) * 1000
                )
                if len(cs.cap_latencies) > 30:
                    cs.cap_latencies.pop(0)
                cs.capture_to_result_ms_avg = np.mean(cs.cap_latencies)

    @staticmethod
    def _draw_live_overlay(frame, overlay):
        annotated = frame.copy()
        active_track_ids = set()
        for track in overlay["tracks"]:
            active_track_ids.add(track["track_id"])
            x1, y1, x2, y2 = [int(value) for value in track["box"]]
            state = track["identity_state"]
            if state == "KNOWN":
                label, color = (
                    f"{track['identity']} (ID-{track['track_id']})",
                    (0, 200, 0),
                )
            elif state == "UNKNOWN":
                label, color = f"Unknown ID-{track['track_id']}", (0, 0, 255)
            elif state == "UNAVAILABLE":
                label, color = f"Unknown ID-{track['track_id']}", (0, 165, 255)
            elif state == "NO_FACE":
                label, color = f"Unknown ID-{track['track_id']}", (255, 180, 0)
            else:
                label, color = f"Unknown ID-{track['track_id']}", (220, 220, 220)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2
            )

        for detection in overlay["detections"]:
            if detection["track_id"] in active_track_ids:
                continue
            x1, y1, x2, y2 = [int(value) for value in detection["box"]]
            color = (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, f"Unknown ID-{detection['track_id']}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2
            )

        for name, confidence, box in overlay["fire_detections"]:
            x1, y1, x2, y2 = [int(value) for value in box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated, f"{name.upper()} {confidence:.2f}",
                (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 2
            )
            cv2.putText(
                annotated, f"{name.upper()} DETECTED",
                (10, 90 if name == "fire" else 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3
            )

        cv2.putText(
            annotated, f"OCCUPANCY: {overlay['occupancy']}",
            (10, annotated.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        y_offset = 60
        now = time.time()
        for event in overlay["alerts"]:
            if now - event["time"] < 5.0:
                cv2.putText(
                    annotated,
                    f"{event['severity']}: {event['type']} {event['desc'][:55]}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 0, 255), 2
                )
                y_offset += 30
        return annotated

    def render_live_frame(self, cam, frame):
        """Draw the latest inference state over the newest captured frame."""
        cs = self._get_cam(cam)
        with cs.lock:
            overlay_age = time.time() - cs.last_annotated_ts
            cs.last_annotated_age_ms = max(0.0, overlay_age * 1000.0)
            if overlay_age > config.MAX_DISPLAY_FRAME_AGE_SECONDS:
                return frame.copy()

            overlay = cs.live_overlay
            overlay = {
                "tracks": [
                    {**track, "box": track["box"].copy()}
                    for track in overlay["tracks"]
                ],
                "detections": [
                    {**detection, "box": detection["box"].copy()}
                    for detection in overlay["detections"]
                ],
                "fire_detections": [
                    (name, confidence, box.copy())
                    for name, confidence, box in overlay["fire_detections"]
                ],
                "alerts": [dict(event) for event in overlay["alerts"]],
                "occupancy": overlay["occupancy"],
            }
        return self._draw_live_overlay(frame, overlay)

    def process_camera(self, cam, frame, capture_ts=None):
        """Offer one frame or a sampled frame batch without blocking inference."""
        cs = self._get_cam(cam)
        if isinstance(frame, np.ndarray):
            samples = [(frame, capture_ts)]
        else:
            samples = []
            for sample in frame:
                if isinstance(sample, tuple) and len(sample) == 2:
                    sample_frame, sample_ts = sample
                    samples.append((sample_frame, sample_ts))
                else:
                    samples.append((sample, capture_ts))
        if not samples:
            raise ValueError("process_camera requires at least one frame")

        items = []
        for sample_frame, sample_ts in samples:
            item_ts = sample_ts if sample_ts is not None else time.time()
            sample_capture_ts = (
                sample_ts if sample_ts is not None else capture_ts
            )
            items.append(
                (sample_frame.copy(), item_ts, sample_capture_ts)
            )
        self._start_worker()
        cs.frames_offered += len(items)

        try:
            cs.q.put_nowait(items)
        except queue.Full:
            try:
                dropped = cs.q.get_nowait()
                cs.frames_overwritten_for_latest += len(dropped)
                cs.frames_dropped_before_inference += len(dropped)
            except queue.Empty:
                pass
            try:
                cs.q.put_nowait(items)
            except queue.Full:
                cs.frames_dropped_before_inference += len(items)

        self._work_event.set()
        cs.frames_displayed += 1
        
    def get_stats(self, cam):
        cs = self._get_cam(cam)
        active_tracks = [cs.tracks[tid] for tid in cs.active_track_ids
                         if tid in cs.tracks]
        return {
            "frames_offered": cs.frames_offered,
            "frames_displayed": cs.frames_displayed,
            "frames_inferred": cs.frames_inferred,
            "frames_dropped_before_inference": cs.frames_dropped_before_inference,
            "frames_overwritten_for_latest": cs.frames_overwritten_for_latest,
            "inference_fps": cs.inference_fps,
            "e2e_ms_avg": cs.e2e_ms_avg,
            "e2e_ms_p95": cs.e2e_ms_p95,
            "capture_to_result_ms_avg": cs.capture_to_result_ms_avg,
            "last_annotated_age_ms": cs.last_annotated_age_ms,
            "inference_ms_avg": float(np.mean(cs.inference_ms)) if cs.inference_ms else 0.0,
            "inference_ms_p95": float(np.percentile(cs.inference_ms, 95)) if cs.inference_ms else 0.0,
            "queue_size": cs.q.qsize(),
            "queue_maxsize": cs.q.maxsize,
            "identity_calls": cs.identity_calls,
            "identity_results": cs.identity_results,
            "active_person_count": cs.occupancy,
            "current_recognized_count": sum(
                tr.identity.state == "KNOWN" for tr in active_tracks
            ),
            "current_unknown_count": sum(
                tr.identity.state == "UNKNOWN" for tr in active_tracks
            ),
            "expired_track_count": cs.expired_track_count,
            "inference_errors": self.inference_errors,
            "last_inference_error": self._worker_error,
        }

    def get_identity_counts(self, cam):
        """Return current counts from unique active tracks, never observations."""
        cs = self._get_cam(cam)
        active_tracks = [cs.tracks[tid] for tid in cs.active_track_ids
                         if tid in cs.tracks]
        return {
            "current_person_count": len(active_tracks),
            "current_recognized_count": sum(
                tr.identity.state == "KNOWN" for tr in active_tracks
            ),
            "current_unknown_count": sum(
                tr.identity.state == "UNKNOWN" for tr in active_tracks
            ),
            "face_not_visible_count": sum(
                tr.identity.state == "NO_FACE" for tr in active_tracks
            ),
            "recognition_unavailable_count": sum(
                tr.identity.state == "UNAVAILABLE" for tr in active_tracks
            ),
        }

    def get_identity_results(self, cam):
        cs = self._get_cam(cam)
        with cs.lock:
            return [
                {"identity": tr.identity.name, "status": tr.identity.state,
                 "confidence": tr.identity.confidence, "track_id": tr.tid,
                 "crossing_id": tr.crossing_id,
                 "crossing_x": (tr.box[0] + tr.box[2]) / 2 if tr.box is not None else None,
                 "direction": next((crossing["dir"] for crossing in cs.crossings
                                    if crossing.get("crossing_id") == tr.crossing_id), None),
                 "crossing_time": next((crossing["time"] for crossing in cs.crossings
                                       if crossing.get("crossing_id") == tr.crossing_id), None)}
                for tr in cs.tracks.values()
                if tr.tid in {str(track_id) for track_id in cs.active_track_ids}
            ]

    # =========================================================================
    # USE CASE ALGORITHMS
    # =========================================================================

    def _tamper_step(self, cs, frame, t):
        """UC7: Camera Tampering. Evaluates blur, darkness, and ORB scene shifts."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.resize(frame, (320, 240))
            
        # Check if a person is blocking the camera (Cover > 35%)
        cs.cover = sum((b[2]-b[0])*(b[3]-b[1]) for b in cs.person_boxes) / (frame.shape[0]*frame.shape[1] + 1)
        if cs.cover > 0.35:
            return # Suspend analysis
            
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_val < 50 and std_val > 15:
            self._add_event(cs, "visibility", "Camera visibility reduced (tampering not established)", "INFO")
            return
        if mean_val < 30 and std_val > 5:
            self._add_event(cs, "visibility", "Camera view suddenly dark", "INFO")
            return
            
        orb = cv2.ORB_create(nfeatures=200)
        kp, des = orb.detectAndCompute(gray, None)
        
        is_tamper = False
        desc_msg = ""
        
        if mean_val < 20 and std_val < 5:
            is_tamper = True
            desc_msg = "Camera fully covered/black"
        elif std_val < 5:
            is_tamper = True
            desc_msg = "Camera texture lost (uniform cover)"
        elif des is not None and len(cs.tamper["baseline_des"]) > 0:
            base_des = cs.tamper["baseline_des"][0]
            if base_des is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(base_des, des)
                if len(matches) < 15:
                    is_tamper = True
                    desc_msg = "Scene shift or major obstruction"
                    
        if is_tamper:
            if cs.tamper["fail_start"] == 0:
                cs.tamper["fail_start"] = t
            elif t - cs.tamper["fail_start"] >= 8.0: # Persistence check
                self._add_event(cs, "tampering", desc_msg, "CRITICAL")
                cs.tamper["fail_start"] = 0
        else:
            cs.tamper["fail_start"] = 0
            
        if len(cs.tamper["baseline_gray"]) < 50:
            cs.tamper["baseline_gray"].append(gray)
            cs.tamper["baseline_des"].append(des)
        else:
            cs.tamper["baseline_gray"].pop(0)
            cs.tamper["baseline_des"].pop(0)
            cs.tamper["baseline_gray"].append(gray)
            cs.tamper["baseline_des"].append(des)

    def _ingest_fire(self, cs, detections, t, frame):
        """UC8: Fire/Smoke."""
        if not detections:
            cs.fire_history = [fh for fh in cs.fire_history if t - fh < 4.0]
            return
            
        cs.fire_history.append(t)
        cs.fire_history = [fh for fh in cs.fire_history if t - fh < 4.0]
        
        if len(cs.fire_history) >= 2: # 2 detections in 4 seconds
            self._add_event(cs, "fire", "Fire or smoke confirmed", "CRITICAL")

    def _associate_faces(self, snap, faces, t):
        """Identity matcher mapping faces to YOLO tracking boxes."""
        res = {}
        used_faces = set()
        for tid, uid, box in snap:
            x1, y1, x2, y2 = box
            match = None
            if faces:
                candidates = []
                for index, f in enumerate(faces):
                    if index in used_faces:
                        continue
                    fx1, fy1, fx2, fy2 = f["bbox"]
                    fcx, fcy = (fx1+fx2)/2, (fy1+fy2)/2
                    if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                        candidates.append((f.get("confidence", 0.0), index, f))
                if candidates:
                    _, face_index, match = max(candidates, key=lambda item: item[0])
                    used_faces.add(face_index)
                        
            if match:
                if match["confidence"] < 0.40 and (match["bbox"][2]-match["bbox"][0]) < 20:
                    state = "NO_FACE"
                elif 0.40 <= match["confidence"] < 0.60:
                    state = "AMBIGUOUS"
                else:
                    state = match["status"]
                res[tid] = {"state": state, "name": match.get("identity"), "conf": match["confidence"]}
            elif faces is None:
                res[tid] = {"state": "UNAVAILABLE"}
            else:
                res[tid] = {"state": "NO_FACE"}
        return res

    def _apply_identity_results(self, cs, t):
        if not cs.id_results: return
        latest = cs.id_results[-1]
        for tid_str, data in latest.items():
            tid = str(tid_str)
            tr = next((candidate for candidate in cs.tracks.values()
                       if candidate.tid == tid), None)
            if tr is not None:
                
                # Check for ID re-use
                if data.get("uid") is not None and data["uid"] != tr.uid:
                    continue
                    
                if tr.identity.state == "KNOWN": 
                    continue # Keep known
                    
                result_state = data["state"]
                if result_state == "KNOWN":
                    tr.identity.obs = 0
                    tr.identity.ambiguous_obs = 0
                    tr.identity.unknown_obs = 0
                    tr.identity.state = "KNOWN"
                    tr.identity.name = data.get("name")
                    tr.identity.confidence = data.get("conf", 0.0)
                elif result_state == "AMBIGUOUS":
                    tr.identity.unknown_obs = 0
                    if tr.identity.state == "UNKNOWN":
                        continue
                    tr.identity.ambiguous_obs += 1
                    tr.identity.obs = tr.identity.ambiguous_obs
                    tr.identity.state = (
                        "UNKNOWN"
                        if tr.identity.ambiguous_obs
                        >= config.AMBIGUOUS_IDENTITY_CONFIRMATIONS
                        else "AMBIGUOUS"
                    )
                    tr.identity.name = data.get("name")
                    tr.identity.confidence = data.get("conf", 0.0)
                elif result_state == "UNKNOWN":
                    tr.identity.ambiguous_obs = 0
                    if tr.identity.state == "UNKNOWN":
                        continue
                    tr.identity.unknown_obs += 1
                    tr.identity.obs = tr.identity.unknown_obs
                    tr.identity.state = (
                        "UNKNOWN" if tr.identity.unknown_obs >= 2 else "PENDING"
                    )
                    tr.identity.name = data.get("name")
                    tr.identity.confidence = data.get("conf", 0.0)
                else:
                    tr.identity.obs = 0
                    tr.identity.ambiguous_obs = 0
                    tr.identity.unknown_obs = 0
                    tr.identity.state = result_state

    def _after_hours_presence(self, cs, track, t):
        if not _is_outside_office_hours(t):
            with self._after_hours_lock:
                self._after_hours.clear()
            return

        known = track.identity.state == "KNOWN" and track.identity.name
        key = f"EMP:{track.identity.name}" if known else f"UNKNOWN:{cs.cam_id}"
        label = str(track.identity.name) if known else "Unknown person"
        with self._after_hours_lock:
            record = self._after_hours.setdefault(
                key, {"first": t, "last": t, "alerted": 0.0}
            )
            if t - record["last"] > config.AFTER_HOURS_GAP_SECONDS:
                record.update(first=t, alerted=0.0)
            record["last"] = t
            stay = t - record["first"]
            if (
                stay >= config.AFTER_HOURS_MIN_STAY_SECONDS
                and t - record["alerted"]
                >= config.AFTER_HOURS_LOG_INTERVAL_SECONDS
            ):
                record["alerted"] = t
                self._add_event(
                    cs,
                    "prolonged_stay",
                    f"After-hours stay: {label}, {int(stay // 60)} min",
                    "WARNING",
                    track.tid,
                )

    @staticmethod
    def _speed_bh_per_s(track, body_height, t, window=2.0, min_body_height=70):
        if body_height < min_body_height:
            return None
        points = [point for point in track.history if t - point[2] <= window]
        if len(points) < 3 or points[-1][2] - points[0][2] < 1.0:
            return None

        steps = [
            math.hypot(later[0] - earlier[0], later[1] - earlier[1])
            for earlier, later in zip(points, points[1:])
        ]
        if max(steps) > 2.5 * body_height:
            return None

        elapsed = points[-1][2] - points[0][2]
        distance = math.hypot(
            points[-1][0] - points[0][0],
            points[-1][1] - points[0][1],
        )
        return distance / elapsed / body_height

    def _identity_events(self, cs, track, t, frame):
        """UC2: Unknown Person / Restricted Area."""
        if track.identity.state == "UNKNOWN":
            # Check restricted zones if configured
            zones = self.p.RESTRICTED_ZONES.get(cs.cam_id, [])
            in_zone = False
            if not zones: 
                in_zone = True # Default all area
            else:
                bx, by = (track.box[0]+track.box[2])/2, track.box[3] # Foot center
                h, w = frame.shape[:2]
                nx, ny = bx/w, by/h
                for z in zones:
                    poly = np.array(z)
                    if cv2.pointPolygonTest(poly.astype(np.float32), (nx, ny), False) >= 0:
                        in_zone = True
                        break
                        
            if in_zone:
                self._add_event(cs, "restricted", "Unknown person (not a confirmed breach)", "HIGH", track.tid)
            self._add_event(cs, "unknown", "Unknown intruder detected", "HIGH", track.tid)

    def _update_cam(self, cs, dets, t, frame, occupancy_count=None):
        """Master update loop for identity, tailgating, crowding, and motion analytics."""
        current_tids = set()
        cs.person_boxes = []
        centers = []
        current_track_ids = set()
        
        # 1. Update tracks
        for d in dets:
            track_key = d["id"]
            tid = str(track_key)
            current_tids.add(tid)
            current_track_ids.add(tid)
            cs.person_boxes.append(d["box"])
            
            if track_key not in cs.tracks:
                cs.tracks[track_key] = Track(tid, tid, t, self.p)
            tr = cs.tracks[track_key]
            tr.last_seen = t
            tr.box = d["box"]
            tr.kp = d.get("kp")
            
            cx = (tr.box[0] + tr.box[2]) / 2
            cy = tr.box[3]
            centers.append([cx, cy])
            tr.history.append((cx, cy, t))
            if tr.kp is not None:
                tr.kp_history.append((tr.kp, t))
                
            # History bounds
            if len(tr.history) > 300: tr.history.pop(0) # Keep ~30s at 10fps
            if len(tr.kp_history) > 30: tr.kp_history.pop(0)
            
        # Stream gap cleanup
        for tid in list(cs.tracks.keys()):
            if t - cs.tracks[tid].last_seen > self.p.STREAM_GAP_RESET_S:
                del cs.tracks[tid]
                cs.expired_track_count += 1
        
        cs.occupancy = (
            len(current_tids) if occupancy_count is None else occupancy_count
        )
        cs.active_track_ids = {
            track_key for track_key, track in cs.tracks.items()
            if track.tid in current_tids
        }

        # 2. Identity (UC2/UC10)
        if self.identity is not None:
            pending_identity = [
                (track.tid, track.box)
                for track in cs.tracks.values()
                if (
                    track.tid in current_tids
                    and track.identity.state != "KNOWN"
                    and track.box is not None
                    and track.box[3] - track.box[1]
                    >= config.IDENTITY_MIN_BOX_HEIGHT
                    and t - track.identity.last_check
                    >= config.IDENTITY_RETRY_SECONDS
                )
            ]
        else:
            pending_identity = []

        if pending_identity:
            res = self.identity.identify_tracks(
                frame, cs.cam_id, pending_identity
            )
            cs.identity_calls += 1
            if res is None:
                res = {
                    track_id: {"state": "UNAVAILABLE"}
                    for track_id, _ in pending_identity
                }
            cs.identity_results += len(res)
            for track_id, result in res.items():
                track = next(
                    (candidate for candidate in cs.tracks.values()
                     if candidate.tid == str(track_id)),
                    None,
                )
                if track is not None:
                    result["uid"] = track.uid
                    track.identity.last_check = t
            cs.id_results = [res]
            self._apply_identity_results(cs, t)
            for tr in cs.tracks.values():
                if tr.tid in {str(track_id) for track_id in res}:
                    self._identity_events(cs, tr, t, frame)
            if getattr(config, "IDENTITY_DEBUG", False):
                active = sorted(current_tids)
                unknown = sorted(tr.tid for tr in cs.tracks.values()
                                 if tr.identity.state == "UNKNOWN")
                print("[IDENTITY] "
                      f"camera_id={cs.cam_id} frame_id={int(t * 1000)} "
                      f"active_track_ids={active} active_person_count={len(active)} "
                      f"identity_calls={cs.identity_calls} identity_results={cs.identity_results} "
                      f"unique_unknown_track_ids={unknown} current_unknown_count={len(unknown)} "
                      f"expired_track_count={cs.expired_track_count}")

        # 3. UC5: Tailgating
        entrance_cameras = {
            camera_id
            for site in config.SITES.values()
            for camera_id in site.get("entrance_cameras", [])
        }
        if "ENTRANCE" in cs.cam_id or cs.cam_id in entrance_cameras:
            entry_line_y = int(frame.shape[0] * 0.6)
            for tid in current_tids:
                tr = next(tr for tr in cs.tracks.values() if tr.tid == tid)
                cy = (tr.box[1] + tr.box[3]) / 2
                if tr.last_y is not None:
                    # Crossing Inward (Y increases)
                    crossing = None
                    if tr.last_y < entry_line_y and cy >= entry_line_y:
                        crossing = {"time": t, "id": tid, "dir": "in",
                                    "x": (tr.box[0]+tr.box[2])/2}
                        cs.crossings.append(crossing)
                    # Crossing Outward
                    elif tr.last_y > entry_line_y and cy <= entry_line_y:
                        crossing = {"time": t, "id": tid, "dir": "out",
                                    "x": (tr.box[0]+tr.box[2])/2}
                        cs.crossings.append(crossing)
                    if crossing is not None:
                        cs.next_crossing_id += 1
                        crossing["crossing_id"] = f"{cs.cam_id}:{cs.next_crossing_id}"
                        tr.crossing_id = crossing["crossing_id"]
                tr.last_y = cy
                
            cs.crossings = [c for c in cs.crossings if t - c["time"] < self.p.TAILGATING_SECONDS]
            
            # Analyze Inward crossings
            in_cross = [c for c in cs.crossings if c["dir"] == "in"]
            if len(in_cross) > 1:
                # Group by proximity (< 1000px apart)
                groups = []
                for c in in_cross:
                    placed = False
                    for g in groups:
                        if abs(c["x"] - g[0]["x"]) < 1000:
                            g.append(c)
                            placed = True
                            break
                    if not placed: groups.append([c])
                
                for g in groups:
                    unique_ids = set([c["id"] for c in g])
                    if len(unique_ids) > 1:
                        # Check if known employees
                        all_known = all(next((tr.identity.state == "KNOWN" for tr in cs.tracks.values()
                                              if tr.tid == i), False) for i in unique_ids)
                        if not all_known:
                            self._add_event(cs, "tailgating", "SUSPECTED tailgating event", "HIGH", "ENTRANCE")

        # 4. UC6: Crowding
        if len(centers) >= self.p.CROWD_MIN_PEOPLE:
            avg_h = np.mean([b[3]-b[1] for b in cs.person_boxes])
            dist_px = self.p.CROWD_PROXIMITY_THRESHOLD * avg_h
            dists = dist.squareform(dist.pdist(centers))
            adj = (dists < dist_px).astype(int)
            n_comps, labels = connected_components(adj, directed=False)
            for i in range(n_comps):
                if (labels == i).sum() >= self.p.CROWD_MIN_PEOPLE:
                    cs.crowd_lost_since = 0.0
                    if cs.crowd_since == 0.0:
                        cs.crowd_since = t
                    if not cs.crowd_active and t - cs.crowd_since >= self.p.CROWD_CONFIRMATION_SECONDS:
                        self._add_event(cs, f"crowding",
                                        f"CROWD DETECTED: {(labels == i).sum()} people",
                                        "WARNING", "GROUP")
                        cs.crowd_active = True
                    break
            else:
                if cs.crowd_active:
                    if cs.crowd_lost_since == 0.0:
                        cs.crowd_lost_since = t
                    elif t - cs.crowd_lost_since >= self.p.CROWD_RELEASE_SECONDS:
                        cs.crowd_since = 0.0
                        cs.crowd_lost_since = 0.0
                        cs.crowd_active = False
                else:
                    cs.crowd_since = 0.0
                    cs.crowd_lost_since = 0.0
        else:
            if cs.crowd_active:
                if cs.crowd_lost_since == 0.0:
                    cs.crowd_lost_since = t
                elif t - cs.crowd_lost_since >= self.p.CROWD_RELEASE_SECONDS:
                    cs.crowd_since = 0.0
                    cs.crowd_lost_since = 0.0
                    cs.crowd_active = False
            else:
                cs.crowd_since = 0.0
                cs.crowd_lost_since = 0.0

        # 5. UC3/4/9: Analytics per Track
        for tid in current_tids:
            tr = next(tr for tr in cs.tracks.values() if tr.tid == tid)
            bw, bh = max(1, tr.box[2]-tr.box[0]), max(1, tr.box[3]-tr.box[1])
            ar = bw / bh
            confidence_ok = False

            self._after_hours_presence(cs, tr, t)
            
            # --- Pose Geometry (UC9) ---
            if tr.kp is not None and len(tr.kp) >= 17:
                k = tr.kp
                nose_y = k[0][1]
                sh_y = (k[5][1] + k[6][1])/2
                hip_y = (k[11][1] + k[12][1])/2
                ank_y = max(k[15][1], k[16][1])
                
                torso_len = hip_y - sh_y
                
                confidence_ok = float(np.mean(k[:, 2])) >= 0.5
                new_state = "Standing" if confidence_ok else "Unknown"
                moving = len(tr.history) >= 5 and np.linalg.norm(
                    np.asarray(tr.history[-1][:2]) - np.asarray(tr.history[-5][:2])
                ) / max(bh, 1) > 0.35

                def joint_angle(a, b, c):
                    ba = np.asarray(a[:2]) - np.asarray(b[:2])
                    bc = np.asarray(c[:2]) - np.asarray(b[:2])
                    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
                    if denominator < 1e-6:
                        return 180.0
                    return float(np.degrees(np.arccos(np.clip(
                        np.dot(ba, bc) / denominator, -1.0, 1.0))))

                left_knee = joint_angle(k[11], k[13], k[15])
                right_knee = joint_angle(k[12], k[14], k[16])
                hip_y = (k[11][1] + k[12][1]) / 2
                knee_y = (k[13][1] + k[14][1]) / 2
                seated = (confidence_ok and not moving
                          and min(left_knee, right_knee) < 135.0
                          and abs(hip_y - knee_y) < bh * 0.35)
                # Fall / Lying
                if confidence_ok and ar > 1.2 and abs(ank_y - nose_y) < bw * 0.7:
                    new_state = "Lying Down"
                elif seated:
                    new_state = "Sitting"
                # Bending
                elif confidence_ok and not moving and torso_len > 0 and torso_len < bh * 0.25 and ar < 1.2:
                    new_state = "Bending"
                
                if new_state != tr.pose_state:
                    tr.fall_candidate = (
                        new_state == "Lying Down"
                        and tr.pose_state in ("Standing", "Sitting", "Bending")
                    )
                    tr.pose_state = new_state
                    tr.state_since = t
                    
                state_dur = t - tr.state_since
                
                if tr.pose_state == "Lying Down":
                    if state_dur >= 3.0:
                        self._add_event(cs, "abnormal", "lying down", "WARNING", tid)
                    if tr.fall_candidate and state_dur > 2.0:
                        self._add_event(cs, "abnormal", "Suspected fall", "CRITICAL", tid)
            elif tr.pose_state != "Unknown":
                tr.pose_state = "Unknown"
                tr.state_since = t

            # --- Kinematics ---
            speed = self._speed_bh_per_s(tr, bh, t)
            tr.run_hits = (
                tr.run_hits + 1
                if speed is not None and speed >= config.RUN_SPEED_BH_PER_S
                else 0
            )
            activity = (
                "RUNNING"
                if tr.run_hits >= config.RUN_CONFIRM_SAMPLES
                else "WALKING"
            )
            if activity != tr.activity_state:
                tr.activity_state = activity
                tr.activity_since = t
            if activity == "RUNNING":
                self._add_event(cs, "abnormal", "running", "WARNING", tid)

            if config.ENABLE_JUMP_DETECTION:
                # UC9: Jumping
                if len(tr.kp_history) >= 10 and confidence_ok:
                    recent = [entry[0] for entry in tr.kp_history[-10:]]
                    if all(np.mean(points[:, 2]) >= 0.5 for points in recent):
                        hips = [float(np.mean(points[11:13, 1])) for points in recent]
                        ankles = [float(np.mean(points[15:17, 1])) for points in recent]
                        hip_excursion = max(hips) - min(hips)
                        ankle_excursion = max(ankles) - min(ankles)
                        airborne = (hip_excursion > 0.15 * bh
                                    and ankle_excursion > 0.10 * bh)
                        tr.jump_frames = tr.jump_frames + 1 if airborne else 0
                        if tr.jump_frames >= config.JUMP_CONFIRMATION_FRAMES:
                            self._add_event(cs, "abnormal", "jumping", "WARNING", tid)
                            tr.jump_frames = 0

            if config.ENABLE_ERRATIC_DETECTION and len(tr.history) > 5:
                xs = [point[0] for point in tr.history[-15:]]
                # UC9: Erratic (ZigZag)
                dx = xs[-1] - xs[-2]
                cur_dir = 1 if dx > 0 else -1
                if tr.last_dir is not None and cur_dir != tr.last_dir and abs(dx) > bw * 0.1:
                    tr.reversals += 1
                tr.last_dir = cur_dir
                if tr.reversals >= 3:
                    self._add_event(cs, "abnormal", "erratic", "WARNING", tid)
                    tr.reversals = 0

    def shutdown(self):
        """Gracefully stop all camera workers."""
        for cs in self._cams.values():
            cs.running = False
        self._worker_running = False
        self._work_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)