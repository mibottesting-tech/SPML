import cv2
import time
import threading
import os
import queue
import config

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
    "analyzeduration;1000000|probesize;1000000"
)


class CameraStream:
    def __init__(self, camera_id, source):
        self.camera_id = camera_id
        self.source = source
        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.frames_read = 0
        self.read_failures = 0
        self.reconnects = 0
        self.capture_fps = 0.0
        self.read_ms_avg = 0.0
        self._fps_started = 0.0
        self._fps_frames = 0
        self._read_ms_total = 0.0
        self._inference_interval = 1.0 / config.INFERENCE_FPS
        self._inference_frames = []
        self._inference_batches = queue.Queue(maxsize=1)
        self.inference_batches_dropped = 0
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.connected = False
                self._inference_frames.clear()
                self.read_failures += 1
                self.reconnects += 1
                time.sleep(2.0)
                continue
            
            self.connected = True
            last_inference_sample = 0.0
            while self.running:
                read_started = time.perf_counter()
                ret, frame = cap.read()
                read_ms = (time.perf_counter() - read_started) * 1000.0
                self._read_ms_total += read_ms
                if not ret or frame is None:
                    self.read_failures += 1
                    break
                self.frames_read += 1
                self._fps_frames += 1
                now = time.monotonic()
                if self._fps_started == 0.0:
                    self._fps_started = now
                elapsed = now - self._fps_started
                if elapsed >= 1.0:
                    self.capture_fps = self._fps_frames / elapsed
                    self._fps_frames = 0
                    self._fps_started = now
                with self.lock:
                    self.latest_frame = frame
                    self.latest_timestamp = time.time()
                    self.read_ms_avg = self._read_ms_total / self.frames_read
                if (last_inference_sample == 0.0
                        or now - last_inference_sample >= self._inference_interval):
                    capture_ts = time.time()
                    self._inference_frames.append((frame.copy(), capture_ts))
                    last_inference_sample = now
                    if len(self._inference_frames) >= config.INFERENCE_BATCH_SIZE:
                        batch = self._inference_frames
                        self._inference_frames = []
                        try:
                            self._inference_batches.put_nowait(batch)
                        except queue.Full:
                            try:
                                self._inference_batches.get_nowait()
                                self.inference_batches_dropped += 1
                            except queue.Empty:
                                pass
                            try:
                                self._inference_batches.put_nowait(batch)
                            except queue.Full:
                                self.inference_batches_dropped += 1
            
            self.connected = False
            cap.release()
            self._inference_frames.clear()
            if self.running:
                time.sleep(2.0)

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), self.latest_timestamp

    def get_inference_batch(self):
        try:
            batch = self._inference_batches.get_nowait()
        except queue.Empty:
            return None
        return [(frame.copy(), timestamp) for frame, timestamp in batch]

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

class CameraManager:
    def __init__(self):
        self.streams = {}

    def add_camera(self, camera_id, source):
        if camera_id not in self.streams:
            stream = CameraStream(camera_id, source)
            self.streams[camera_id] = stream
            stream.start()

    def get_frame(self, camera_id):
        if camera_id in self.streams:
            return self.streams[camera_id].get_latest_frame()
        return None, 0.0

    def get_inference_batch(self, camera_id):
        stream = self.streams.get(camera_id)
        return stream.get_inference_batch() if stream is not None else None

    def get_stats(self, camera_id):
        stream = self.streams.get(camera_id)
        if stream is None:
            return {}
        return {
            "frames_read": stream.frames_read,
            "capture_fps": stream.capture_fps,
            "read_failures": stream.read_failures,
            "reconnects": stream.reconnects,
            "read_ms_avg": stream.read_ms_avg,
            "connected": stream.connected,
            "inference_batches_dropped": stream.inference_batches_dropped,
        }

    def shutdown(self):
        for stream in self.streams.values():
            stream.stop()
