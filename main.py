import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import sys
import cv2
import numpy as np
import config
from camera import CameraManager
from identity import IdentityService
from attendance import AttendanceMonitor
from other import SurveillanceEngine


SITE_WINDOWS = {
    "site_1_kolkata": ("Site 1 - Kolkata", "S1_CAM_"),
    "site_2_pune": ("Site 2 - Pune", "PUNE_CAM_"),
}


def main():
    print("Initializing SPML Multi-Site AI CCTV Surveillance System...")

    ident_service = IdentityService()
    att_monitor = AttendanceMonitor(ident_service)
    surv_engine = SurveillanceEngine(ident_service, preload_models=True)
    cam_manager = CameraManager()

    site = config.get_site("all_sites")
    active_cameras = list(site["camera_ids"])
    attendance_cameras = set(site.get("attendance_streams", {}))
    site_cameras = {
        site_key: [
            cam for cam in active_cameras
            if cam.startswith(prefix)
        ]
        for site_key, (_, prefix) in SITE_WINDOWS.items()
    }

    print(f"Configured cameras: {len(active_cameras)}")
    for site_key, (window_name, _) in SITE_WINDOWS.items():
        print(f"{window_name} cameras: {len(site_cameras[site_key])}")

    print(f"System active. Attempting to connect to {len(active_cameras)} cameras...")
    for cam in active_cameras:
        stream_type = "MAIN" if cam.startswith("PUNE_CAM_") else "SUB"

        try:
            cam_url = config.get_camera_url(cam, stream_type)
            cam_manager.add_camera(cam, cam_url)
        except ValueError as e:
            print(f"[WARNING] Skipping {cam}: {e}")

    selected_camera = {"camera": None, "site": None}
    panel_bounds = {site_key: {} for site_key in SITE_WINDOWS}
    displayed_frames = {}
    displayed_timestamps = {}

    for window_name, _ in SITE_WINDOWS.values():
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def make_select_camera(site_key):
        def select_camera(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            if selected_camera["camera"] is None:
                for cam, (left, top, right, bottom) in panel_bounds[site_key].items():
                    if left <= x < right and top <= y < bottom:
                        selected_camera["camera"] = cam
                        selected_camera["site"] = site_key
                        break
            else:
                selected_camera["camera"] = None
                selected_camera["site"] = None
        return select_camera

    for site_key, (window_name, _) in SITE_WINDOWS.items():
        cv2.setMouseCallback(window_name, make_select_camera(site_key))

    prev_display_time = time.time()
    try:
        print("Press 'q' in either video window to quit.")
        while True:
            current_time = time.time()
            display_fps = 1.0 / (current_time - prev_display_time + 0.0001)
            prev_display_time = current_time

            for cam in active_cameras:
                frame, frame_ts = cam_manager.get_frame(cam)
                if frame is not None and frame_ts != displayed_timestamps.get(cam):
                    displayed_timestamps[cam] = frame_ts
                    live_frame = surv_engine.render_live_frame(
                        cam, frame,
                    )
                    cv2.putText(
                        live_frame, f"{cam} - LIVE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    role = site.get("attendance_streams", {}).get(cam)
                    if role:
                        cv2.putText(
                            live_frame, role, (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                        )
                    displayed_frames[cam] = live_frame

                batch = cam_manager.get_inference_batch(cam)
                if batch:
                    inference_start = time.time()
                    surv_engine.process_camera(cam, batch)
                    inference_latency = (time.time() - inference_start) * 1000

                    print(
                        f"[INFERENCE] {cam} submitted {len(batch)} frames "
                        f"in {inference_latency:.1f}ms (display FPS {display_fps:.1f})"
                    )

                if cam in attendance_cameras:
                    att_monitor.process_results(
                        cam, surv_engine.get_identity_results(cam)
                    )

                if frame is None and cam not in displayed_frames:
                    stream = cam_manager.streams.get(cam)
                    status = (
                        "CONNECTING"
                        if stream is not None and not stream.connected
                        else "NO FRAME"
                    )
                    placeholder = np.zeros((270, 480, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder, f"{cam}: {status}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2
                    )
                    displayed_frames[cam] = placeholder

            for site_key, (window_name, _) in SITE_WINDOWS.items():
                panel_bounds[site_key].clear()
                cameras = site_cameras[site_key]
                selected = selected_camera["camera"]
                if (selected_camera["site"] == site_key
                        and selected in displayed_frames):
                    frame = displayed_frames[selected].copy()
                    cv2.putText(
                        frame, "Click to return to dashboard", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                    )
                    cv2.imshow(window_name, frame)
                    continue

                columns = max(1, int(np.ceil(np.sqrt(len(cameras)))))
                cell_width, cell_height = 320, 180
                rows = max(1, int(np.ceil(len(cameras) / columns)))
                dashboard = np.zeros(
                    (rows * cell_height + 30, columns * cell_width, 3),
                    dtype=np.uint8
                )
                cv2.putText(
                    dashboard, window_name, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
                for index, cam in enumerate(cameras):
                    row, column = divmod(index, columns)
                    left, top = column * cell_width, 30 + row * cell_height
                    frame = displayed_frames.get(cam)
                    if frame is None:
                        frame = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                        cv2.putText(
                            frame, "WAITING FOR CAMERA", (20, cell_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2
                        )
                    else:
                        frame = cv2.resize(frame, (cell_width, cell_height))
                    dashboard[top:top + cell_height, left:left + cell_width] = frame
                    panel_bounds[site_key][cam] = (
                        left, top, left + cell_width, top + cell_height
                    )
                    cv2.rectangle(
                        dashboard, (left, top),
                        (left + cell_width - 1, top + cell_height - 1),
                        (255, 255, 255), 1
                    )
                cv2.imshow(window_name, dashboard)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                print("\nQuit key pressed...")
                break

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt received...")
    finally:
        print("Shutting down gracefully...")
        cam_manager.shutdown()
        surv_engine.shutdown()
        att_monitor.close()
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == "__main__":
    main()
