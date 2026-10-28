# SPML Multi-Site AI CCTV Surveillance System

## Setup Instructions

1. **Install Dependencies**
   Make sure you have Python 3.9+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Model & Identity Files**
   - Place `fire_smoke.pt` and `yolov8s-pose.pt` inside the `models/` folder. `yolo11m.pt` is loaded once during engine startup and downloaded by Ultralytics if needed.
   - Place Kolkata identities in `identity_data/employee_embeddings.npz` and Pune identities in `identity_data/pune_embeddings.npz`. The Pune database is a separate file and must be generated for Pune employees.

3. **Configure Environment Variables**
   Configure each Kolkata main stream as `CAM_<CAMERA_ID>_MAIN` and set the Pune connection variables (`PUNE_IP`, `PUNE_PORT`, `PUNE_USERNAME`, and `PUNE_PASSWORD`) in `.env`. All cameras use their main stream for person detection; the application reports missing stream URLs instead of silently falling back to substreams. Keep camera credentials out of source files.

4. **Run the Application**
   ```bash
   python main.py
   ```

## Live identity-count semantics

Identity results are associated with the persistent person tracks created by the
pose tracker. `current_unknown_count` is the number of active tracks whose
identity state has been confirmed as unknown; it is not a count of face
detections, recognition calls, frames, or historical events. Unknown
observations require two independent recognition observations before an alert
is emitted.

The engine keeps separate states for `KNOWN`, `UNKNOWN`, `AMBIGUOUS`, `NO_FACE`,
and `UNAVAILABLE`. A face that is not visible, or a recognition failure, is not
automatically treated as an intruder. Repeated ambiguous matches are confirmed
as unknown only after three track-level observations. Tracks are expired after
the configured stream-gap grace period. Use `SurveillanceEngine.get_identity_counts(camera_id)`
for current counts and `get_stats(camera_id)` for performance telemetry.

The live display consumes the newest captured frame and uses a bounded
inference queue. Slow inference can overwrite queued work, but it cannot
accumulate an unbounded backlog and delay the displayed camera feed.

Inference runs at one sampled frame per second with one-frame batches and a
single-frame queue. This is a conservative CPU-only setting for a multi-camera
deployment; increase it only after measuring the target host's throughput.

## Camera deployment and monitoring

Edit [config.py](./config.py) to configure the camera and attendance mappings
under `site_1_kolkata` and `site_2_pune`. Kolkata cameras use the `S1_CAM_`
prefix; Pune cameras use `PUNE_CAM_`. Every configured camera runs the general
surveillance pipeline in `other.py`. Put Kolkata RTSP URLs in `.env` under
`CAM_<CAMERA_ID>_SUB`; Pune URLs are assembled from the Pune connection values in `.env`; the main
stream is selected for surveillance inference.
Camera credentials remain only in `.env`.

The application opens separate `Site 1 - Kolkata` and `Site 2 - Pune` windows.
Click a camera tile to expand it and click again to return to its site's grid.
Inference samples one frame per second from each capture thread and submits
single-frame work to bounded per-camera queues. A shared `yolo11m.pt` detector
processes overlapping 2×2 tiles from all cameras in one inference worker at
1280 resolution and 0.45 confidence; ByteTrack state remains isolated per
camera. Pose estimation runs only on detected people at least 120 pixels tall.
Tile boundaries are shown when a camera is selected. Capture and display remain
independent from model processing. The FFmpeg capture options request
low-latency TCP RTSP playback.

Evidence is written beneath site-specific directories:
`evidence/site 1 - kolkata/` and `evidence/site 2 - pune/`. The Pune attendance
mapping is a placeholder for `PUNE_CAM_01` entry and `PUNE_CAM_02` exit.
Kolkata's attendance mappings are intentionally empty until configured.

Build each face database from its own employee-image directory:

```powershell
python build_db.py site_1_kolkata <kolkata-image-directory>
python build_db.py site_2_pune <pune-image-directory>
```

Attendance is restricted to the IDs in that site's `attendance_streams`
mapping. Assign `"ENTRY"` or `"EXIT"` there; cameras in `camera_ids` but not
in this mapping still receive general surveillance and never create attendance
records. The temporary test mapping is:

- `TEST_CAM_01_ENTRY` -> `CAM_TEST_CAM_01_MAIN` -> `ENTRY`
- `TEST_CAM_01_EXIT` -> `CAM_TEST_CAM_01_SUB` -> `EXIT`

Attendance persistence is active through
`attendance/db_manager.py` and the existing `attendance/attendance.db`.
The legacy root [attendance.py](./attendance.py) remains the application
adapter and is not deleted; it routes valid crossing events to the database
manager while retaining the existing CSV/event behavior. Database camera
numbers are configured beside each site's `attendance_streams` mapping in
`config.py`: channel 1 is `ENTRY`, channel 2 is `EXIT`. The database schema
stores that direction in its existing `status` column because it has no
separate direction column.

The OpenCV monitoring window renders one panel per configured camera. Click a
panel to expand it; click the expanded view to return to the grid. A camera
that has no frame shows its own connection state without blocking other panels.

## MySQL attendance

The attendance integration uses [attendance_db.py](./attendance_db.py)
and [sql/attendance_schema.sql](./sql/attendance_schema.sql). It reuses the
recognized employee ID and queues only confirmed crossing events from cameras
listed in `attendance_streams`. A background writer commits each transaction,
so a slow or unavailable MySQL server does not block capture, inference, or
display. Failed commits are reported as database errors and are not reported
as successful writes.

Each confirmed attendance crossing is written to `logs/attendance.csv` as
well as the existing `logs/events.csv`, then queued for the MySQL writer.
Unknown identities and cameras without an attendance stream are not saved as
attendance records.

Install the dependency, initialize the schema, and configure `.env`:

```powershell
pip install -r requirements.txt
mysql -u root -p < sql\attendance_schema.sql
```

Create the least-privilege application user using the commented statements in
the schema, then set `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and
`DB_PASSWORD` in `.env`. Timestamps are stored using the `Asia/Kolkata`
attendance timezone. An entry opens a session; an exit closes the latest open
session, including when the exit occurs after midnight. An unmatched exit is
reported and does not create a fabricated entry.

Example queries:

```sql
SELECT employee_id, attendance_date, in_time, out_time, status
FROM employee_attendance
WHERE attendance_date = CURDATE()
ORDER BY in_time;

SELECT employee_id, in_time,
       TIMESTAMPDIFF(MINUTE, in_time, out_time) AS minutes_worked
FROM employee_attendance
WHERE employee_id = 'EMP001'
ORDER BY in_time DESC;
```

Crowding, after-hours presence, and motion thresholds are configured in
[config.py](./config.py). After-hours presence alerts require ten minutes of
continuous observations; a gap longer than two minutes starts a new session.

The display draws the latest completed inference overlay on the newest
captured frame. This keeps the camera feed current without retaining a
redundant full-frame copy in each inference worker.

For temporary attendance testing, `test_office` uses two logical streams from
the same physical camera: `TEST_CAM_01_ENTRY` reads
`CAM_TEST_CAM_01_MAIN` (subtype 0), and `TEST_CAM_01_EXIT` reads
`CAM_TEST_CAM_01_SUB` (subtype 1). This validates software entry/exit routing
only; it does not represent two independent physical entrances.
"# SPML" 
"# SPML" 
