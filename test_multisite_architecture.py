import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import attendance
import camera
import config
import identity
import other


class MultiSiteArchitectureTests(unittest.TestCase):
    def test_live_track_labels_only_show_identity_and_track_id(self):
        overlay = {
            "tracks": [
                {
                    "box": np.array([1, 1, 20, 20]),
                    "track_id": "7",
                    "identity_state": "KNOWN",
                    "identity": "Employee A",
                    "posture": "SITTING",
                    "activity": "WALKING",
                },
                {
                    "box": np.array([25, 1, 45, 20]),
                    "track_id": "8",
                    "identity_state": "NO_FACE",
                    "identity": None,
                    "posture": "STANDING",
                    "activity": "RUNNING",
                },
            ],
            "detections": [],
            "fire_detections": [],
            "occupancy": 2,
            "alerts": [],
        }
        with patch.object(other.cv2, "putText", wraps=cv2.putText) as put_text:
            other.SurveillanceEngine._draw_live_overlay(
                np.zeros((60, 60, 3), dtype=np.uint8), overlay
            )

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("Employee A (ID-7)", labels)
        self.assertIn("Unknown ID-8", labels)
        self.assertFalse(any(
            word in label for label in labels
            for word in ("SITTING", "STANDING", "WALKING", "RUNNING")
        ))

    def test_site_configuration_is_separated(self):
        kolkata = config.SITES["site_1_kolkata"]
        pune = config.SITES["site_2_pune"]

        self.assertTrue(all(cam.startswith("S1_CAM_") for cam in kolkata["camera_ids"]))
        self.assertTrue(all(cam.startswith("PUNE_CAM_") for cam in pune["camera_ids"]))
        self.assertEqual(
            kolkata["attendance_streams"],
            {"S1_CAM_01": "ENTRY", "S1_CAM_02": "EXIT"},
        )
        self.assertEqual(
            pune["attendance_streams"],
            {"PUNE_CAM_01": "ENTRY", "PUNE_CAM_02": "EXIT"},
        )
        self.assertEqual(config.INFERENCE_FPS, 1)
        self.assertEqual(config.INFERENCE_BATCH_SIZE, 1)
        self.assertEqual(config.INFERENCE_QUEUE_SIZE, 1)

    def test_pune_camera_stream_argument_selects_subtype(self):
        variables = {
            "PUNE_IP": "127.0.0.1",
            "PUNE_PORT": "554",
            "PUNE_USERNAME": "viewer",
            "PUNE_PASSWORD": "example-password",
        }
        with patch.dict(os.environ, variables):
            self.assertIn(
                "channel=1&subtype=1",
                config.get_camera_url("PUNE_CAM_01", "SUB"),
            )
            self.assertIn(
                "channel=1&subtype=0",
                config.get_camera_url("PUNE_CAM_01", "MAIN"),
            )

    def test_identity_databases_load_and_route_by_camera(self):
        with tempfile.TemporaryDirectory() as identity_dir:
            kolkata_path = os.path.join(identity_dir, "kolkata.npz")
            pune_path = os.path.join(identity_dir, "pune.npz")
            embeddings = np.eye(2, 512, dtype=np.float32)
            np.savez(
                kolkata_path,
                employee_ids=np.array(["KOLKATA_EMP"], dtype=object),
                embeddings=embeddings[:1],
            )
            np.savez(
                pune_path,
                employee_ids=np.array(["PUNE_EMP"], dtype=object),
                embeddings=embeddings[:1],
            )

            service = identity.IdentityService.__new__(identity.IdentityService)
            with patch.object(config, "KOLKATA_IDENTITY_DB", kolkata_path), patch.object(
                config, "PUNE_IDENTITY_DB", pune_path
            ):
                service.site_databases = service._load_db()

            self.assertEqual(
                service.site_databases["site_1_kolkata"][0].tolist(),
                ["KOLKATA_EMP"],
            )
            self.assertEqual(
                service.site_databases["site_2_pune"][0].tolist(),
                ["PUNE_EMP"],
            )

        class FakeFaceApp:
            def get(self, frame):
                return [SimpleNamespace(
                    embedding=np.eye(1, 512, dtype=np.float32)[0],
                    bbox=np.array([0, 0, 10, 10]),
                )]

        service.lock = threading.Lock()
        service.face_app = FakeFaceApp()
        service.identity_calls = 0
        service.last_error = None
        service.active_providers = ["CPUExecutionProvider"]
        service.site_databases = {
            "site_1_kolkata": (
                np.array(["KOLKATA_EMP"], dtype=object),
                np.eye(1, 512, dtype=np.float32),
            ),
            "site_2_pune": (
                np.array(["PUNE_EMP"], dtype=object),
                np.eye(1, 512, dtype=np.float32),
            ),
        }

        self.assertEqual(
            service.identify(np.zeros((10, 10, 3), dtype=np.uint8), "S1_CAM_01")[0]["identity"],
            "KOLKATA_EMP",
        )
        self.assertEqual(
            service.identify(np.zeros((10, 10, 3), dtype=np.uint8), "PUNE_CAM_01")[0]["identity"],
            "PUNE_EMP",
        )
        crop_shapes = []
        service.face_app.get = lambda crop: (
            crop_shapes.append(crop.shape[:2])
            or [SimpleNamespace(
                embedding=np.eye(1, 512, dtype=np.float32)[0],
                bbox=np.array([0, 0, 10, 10]),
            )]
        )
        track_results = service.identify_tracks(
            np.zeros((400, 300, 3), dtype=np.uint8),
            "PUNE_CAM_01",
            [(7, np.array([100, 20, 200, 300], dtype=np.float32))],
        )
        self.assertEqual(track_results["7"]["state"], "KNOWN")
        self.assertGreaterEqual(crop_shapes[0][0], 200)

    def test_attendance_passes_camera_context_to_identity_service(self):
        class IdentityRecorder:
            def __init__(self):
                self.camera_id = None

            def identify(self, frame, camera_id):
                self.camera_id = camera_id
                return []

        monitor = attendance.AttendanceMonitor.__new__(attendance.AttendanceMonitor)
        monitor.identity_service = IdentityRecorder()
        monitor.process_results = lambda camera_id, results: None
        monitor.process_frame("PUNE_CAM_01", np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertEqual(monitor.identity_service.camera_id, "PUNE_CAM_01")

    def test_camera_stream_samples_fps_and_emits_full_batches(self):
        class FakeCapture:
            def __init__(self, source, backend):
                self.opened = True

            def set(self, property_id, value):
                return True

            def isOpened(self):
                return self.opened

            def read(self):
                time.sleep(0.002)
                return True, np.zeros((4, 4, 3), dtype=np.uint8)

            def release(self):
                self.opened = False

        with patch.object(config, "INFERENCE_FPS", 20), patch.object(
            config, "INFERENCE_BATCH_SIZE", 3
        ), patch.object(camera.cv2, "VideoCapture", FakeCapture):
            stream = camera.CameraStream("S1_CAM_01", "fake-stream")
            stream.start()
            batch = None
            deadline = time.monotonic() + 2.0
            try:
                while time.monotonic() < deadline and batch is None:
                    batch = stream.get_inference_batch()
                    time.sleep(0.005)
            finally:
                stream.stop()

        self.assertIsNotNone(batch)
        self.assertEqual(len(batch), 3)
        timestamps = [timestamp for _, timestamp in batch]
        self.assertTrue(all(
            later - earlier >= 0.035
            for earlier, later in zip(timestamps, timestamps[1:])
        ))

    def test_inference_worker_processes_every_frame_in_a_batch(self):
        class FakeTensor:
            def __init__(self, value):
                self.value = value

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class FakeBoxes:
            def __init__(self, boxes, confidence):
                self.xyxy = FakeTensor(boxes)
                self.conf = FakeTensor(confidence)

            def __len__(self):
                return len(self.xyxy.value)

        class EmptyDetector:
            def predict(self, images, **kwargs):
                return [
                    SimpleNamespace(boxes=FakeBoxes(
                        np.empty((0, 4), dtype=np.float32),
                        np.empty((0,), dtype=np.float32),
                    ))
                    for _ in images
                ]

        engine = other.SurveillanceEngine(None)
        engine.detector_factory = EmptyDetector
        engine.pose_factory = lambda camera_id: None
        frames = [
            (np.zeros((16, 16, 3), dtype=np.uint8), time.time() + index * 0.01)
            for index in range(3)
        ]
        result = engine.process_camera("S1_CAM_01", frames)
        deadline = time.monotonic() + 3.0
        try:
            while (engine.get_stats("S1_CAM_01")["frames_inferred"] < len(frames)
                   and time.monotonic() < deadline):
                time.sleep(0.01)
            stats = engine.get_stats("S1_CAM_01")
        finally:
            engine.shutdown()

        self.assertIsNone(result)
        self.assertEqual(stats["frames_offered"], len(frames))
        self.assertEqual(stats["frames_inferred"], len(frames))

    def test_worker_keeps_low_keypoint_and_untracked_person_detections(self):
        class FakeTensor:
            def __init__(self, value):
                self.value = value

            def cpu(self):
                return self

            def numpy(self):
                return self.value

            def int(self):
                return self

            def tolist(self):
                return self.value

        class FakeBoxes:
            def __init__(self, boxes, confidence):
                self.xyxy = FakeTensor(boxes)
                self.conf = FakeTensor(confidence)

            def __len__(self):
                return len(self.xyxy.value)

        class FakeDetector:
            def __init__(self):
                self.predict_kwargs = None

            def predict(self, images, **kwargs):
                self.predict_kwargs = kwargs
                results = []
                for index, _ in enumerate(images):
                    if index % 4 == 0:
                        boxes = np.array([
                            [5, 5, 12, 30],
                            [20, 5, 30, 30],
                        ], dtype=np.float32)
                        confidence = np.array([0.9, 0.9], dtype=np.float32)
                    else:
                        boxes = np.empty((0, 4), dtype=np.float32)
                        confidence = np.empty((0,), dtype=np.float32)
                    results.append(SimpleNamespace(
                        boxes=FakeBoxes(boxes, confidence)
                    ))
                return results

        engine = other.SurveillanceEngine(None)
        fake_detector = FakeDetector()
        engine.detector_factory = lambda: fake_detector
        engine.pose_factory = lambda camera_id: None
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        engine.process_camera("S1_CAM_01", frame)
        deadline = time.monotonic() + 3.0
        try:
            camera_state = engine._cams["S1_CAM_01"]
            while camera_state.frames_inferred < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            overlay = camera_state.live_overlay
            occupancy = camera_state.occupancy
        finally:
            engine.shutdown()

        self.assertEqual(fake_detector.predict_kwargs["conf"], 0.25)
        self.assertEqual(fake_detector.predict_kwargs["iou"], 0.30)
        self.assertEqual(fake_detector.predict_kwargs["imgsz"], 1280)
        self.assertEqual(fake_detector.predict_kwargs["classes"], [0])
        self.assertEqual(fake_detector.predict_kwargs["device"], "cpu")
        self.assertEqual(occupancy, 2)
        self.assertEqual(len(overlay["detections"]), 2)
        self.assertTrue(all(
            isinstance(detection["track_id"], str)
            and detection["track_id"]
            for detection in overlay["detections"]
        ))

    def test_latest_inference_overlay_is_drawn_on_each_live_frame(self):
        engine = other.SurveillanceEngine.__new__(other.SurveillanceEngine)
        engine._lock = threading.Lock()
        engine._cams = {"S1_CAM_01": other.CameraState("S1_CAM_01", None)}
        camera_state = other.CameraState("S1_CAM_01", None)
        camera_state.live_overlay = {
            "tracks": [{
                "box": np.array([10, 10, 40, 50]),
                "track_id": "17",
                "identity_state": "KNOWN",
                "identity": "EMP017",
                "posture": "Standing",
                "activity": "STANDING",
            }],
            "detections": [{
                "box": np.array([10, 10, 40, 50]),
                "track_id": "17",
            }],
            "fire_detections": [],
            "alerts": [{
                "severity": "WARNING",
                "type": "prolonged_stay",
                "desc": "standing too long",
                "time": time.time(),
            }],
            "occupancy": 1,
        }
        camera_state.last_annotated_ts = time.time()
        engine._cams = {"S1_CAM_01": camera_state}
        frame_one = np.zeros((80, 80, 3), dtype=np.uint8)
        frame_two = np.zeros_like(frame_one)

        rendered_one = engine.render_live_frame("S1_CAM_01", frame_one)
        rendered_two = engine.render_live_frame("S1_CAM_01", frame_two)

        self.assertFalse(np.any(frame_one))
        self.assertFalse(np.any(frame_two))
        self.assertGreater(int(rendered_one[10, 10][1]), 150)
        self.assertGreater(int(rendered_two[10, 10][1]), 150)
        self.assertTrue(np.any(rendered_one[55:75, :, 2] > 0))
        self.assertTrue(np.any(rendered_two[55:75, :, 2] > 0))

    def test_stale_inference_overlay_is_not_drawn_on_live_frame(self):
        engine = other.SurveillanceEngine.__new__(other.SurveillanceEngine)
        engine._lock = threading.Lock()
        camera_state = other.CameraState("S1_CAM_01", None)
        camera_state.live_overlay["tracks"] = [{
            "box": np.array([10, 10, 40, 50]),
            "track_id": "17",
            "identity_state": "KNOWN",
            "identity": "EMP017",
            "posture": "Standing",
            "activity": "STANDING",
        }]
        camera_state.last_annotated_ts = (
            time.time() - config.MAX_DISPLAY_FRAME_AGE_SECONDS - 0.1
        )
        engine._cams = {"S1_CAM_01": camera_state}
        frame = np.zeros((80, 80, 3), dtype=np.uint8)

        rendered = engine.render_live_frame("S1_CAM_01", frame)

        self.assertIsNot(rendered, frame)
        np.testing.assert_array_equal(rendered, frame)

    def test_selected_camera_draws_runtime_sized_detection_tiles(self):
        bounds = other.SurveillanceEngine._tile_bounds(
            np.zeros((100, 200, 3), dtype=np.uint8)
        )
        self.assertEqual(bounds, [
            (0, 0, 110, 55),
            (90, 0, 200, 55),
            (0, 45, 110, 100),
            (90, 45, 200, 100),
        ])

    def test_prolonged_stay_is_after_hours_only_and_posture_independent(self):
        def make_detection(sitting):
            keypoints = np.zeros((17, 3), dtype=np.float32)
            positions = {
                0: (50, 10), 1: (48, 9), 2: (52, 9), 3: (46, 10), 4: (54, 10),
                5: (40, 25), 6: (60, 25), 7: (35, 40), 8: (65, 40),
                9: (32, 48), 10: (68, 48), 11: (45, 55), 12: (55, 55),
                13: (45, 80), 14: (55, 80), 15: (45, 100), 16: (55, 100),
            }
            if sitting:
                positions.update({
                    13: (50, 60), 14: (50, 60),
                    15: (45, 70), 16: (55, 70),
                })
            for index, (x, y) in positions.items():
                keypoints[index] = (x, y, 0.9)
            return {
                "id": 1,
                "box": np.array([30, 5, 70, 105], dtype=np.float32),
                "kp": keypoints,
            }

        for sitting in (False, True):
            events = []
            params = SimpleNamespace(
                CROWD_MIN_PEOPLE=3,
                CROWD_PROXIMITY_THRESHOLD=1.5,
                CROWD_CONFIRMATION_SECONDS=1.0,
                CROWD_RELEASE_SECONDS=1.0,
                LOITERING_DURATION=10.0,
                TAILGATING_SECONDS=3.0,
                STREAM_GAP_RESET_S=2.0,
            )
            engine = SimpleNamespace(
                p=params,
                identity=None,
                _after_hours={},
                _after_hours_lock=threading.Lock(),
                _add_event=lambda cs, typ, desc, severity="WARNING", target=None:
                    events.append(typ),
            )
            engine._after_hours_presence = lambda cs, track, timestamp: (
                other.SurveillanceEngine._after_hours_presence(
                    engine, cs, track, timestamp
                )
            )
            engine._speed_bh_per_s = other.SurveillanceEngine._speed_bh_per_s
            camera_state = other.CameraState("S1_CAM_01", params)
            detection = make_detection(sitting)
            frame = np.zeros((120, 100, 3), dtype=np.uint8)
            with patch.object(other, "_is_outside_office_hours", return_value=False):
                for timestamp in (1000.0, 1300.0, 1600.0):
                    other.SurveillanceEngine._update_cam(
                        engine, camera_state, [detection], timestamp, frame,
                    )
            self.assertEqual(events, [])

            with patch.object(other, "_is_outside_office_hours", return_value=True):
                for timestamp in range(1900, 2801, 100):
                    other.SurveillanceEngine._update_cam(
                        engine, camera_state, [detection], timestamp, frame,
                    )
            self.assertEqual(events, ["prolonged_stay", "prolonged_stay"])

    def test_occupancy_counts_all_visible_tracks(self):
        params = SimpleNamespace(
            CROWD_MIN_PEOPLE=3,
            CROWD_PROXIMITY_THRESHOLD=1.5,
            CROWD_CONFIRMATION_SECONDS=1.0,
            CROWD_RELEASE_SECONDS=1.0,
            TAILGATING_SECONDS=3.0,
            STREAM_GAP_RESET_S=2.0,
        )
        engine = SimpleNamespace(
            p=params,
            identity=None,
            _add_event=lambda *args: None,
            _after_hours_presence=lambda *args: None,
            _speed_bh_per_s=other.SurveillanceEngine._speed_bh_per_s,
        )
        camera_state = other.CameraState("S1_CAM_01", params)
        detections = [
            {
                "id": track_id,
                "box": np.array(
                    [track_id * 250, 10, track_id * 250 + 30, 100],
                    dtype=np.float32,
                ),
                "kp": None,
            }
            for track_id in range(1, 7)
        ]
        with patch.object(other, "_is_outside_office_hours", return_value=False):
            other.SurveillanceEngine._update_cam(
                engine, camera_state, detections, 1000.0,
                np.zeros((120, 1700, 3), dtype=np.uint8),
            )

        self.assertEqual(camera_state.occupancy, 6)

    def test_event_evidence_isolated_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            evidence_dir = os.path.join(temporary_dir, "evidence")
            logs_dir = os.path.join(temporary_dir, "logs")
            os.makedirs(logs_dir)
            engine = other.SurveillanceEngine.__new__(other.SurveillanceEngine)
            engine.events_csv = os.path.join(logs_dir, "events.csv")
            camera_state = other.CameraState("S1_CAM_01", None)
            camera_state.last_frame = np.zeros((80, 80, 3), dtype=np.uint8)
            camera_state.tracks = {
                1: SimpleNamespace(tid="1", box=np.array([5, 45, 25, 65])),
                2: SimpleNamespace(tid="2", box=np.array([45, 45, 65, 65])),
            }

            with patch.object(config, "EVIDENCE_DIR", evidence_dir), patch.object(
                config, "EVIDENCE_DEDUP_SECONDS", 300.0
            ), patch.object(other.time, "time", side_effect=[1000.0, 1011.0, 1012.0]):
                engine._add_event(camera_state, "prolonged_stay", "standing too long", target="1")
                engine._add_event(camera_state, "prolonged_stay", "standing too long", target="1")
                camera_state.last_frame = np.tile(
                    np.arange(80, dtype=np.uint8)[None, :, None], (80, 1, 3)
                )
                engine._add_event(camera_state, "prolonged_stay", "different anomaly", target="1")

            first_event = camera_state.event_history[0]
            self.assertIn(os.path.join("site 1 - kolkata", "prolonged_stay"), first_event["evidence_path"])
            image = cv2.imread(first_event["evidence_path"])
            self.assertGreater(int(image[45, 5][2]), 150)
            self.assertLess(int(image[45, 5][0]), 50)
            self.assertLess(int(image[45, 45].max()), 20)
            self.assertFalse(camera_state.event_history[1]["evidence_saved"])
            self.assertTrue(camera_state.event_history[2]["evidence_saved"])
            saved_files = [
                name for _, _, files in os.walk(evidence_dir) for name in files
            ]
            self.assertEqual(len(saved_files), 2)


if __name__ == "__main__":
    unittest.main()
