import time
import threading
import csv
import os
from datetime import datetime
import config
from attendance_db import get_database, save_attendance

class AttendanceMonitor:
    def __init__(self, identity_service):
        self.identity_service = identity_service
        self.person_states = {}
        self.logged_crossings = set()
        self.seen_crossings = set()
        self.recent_crossings = []
        self.pending_unknown = {}
        self.employee_names = self._load_names()
        self.lock = threading.Lock()
        self.database_write_ms = []
        self.database_write_failures = 0
        self.database = get_database()
        self.attendance_file = os.path.join(config.LOGS_DIR, "events.csv")
        self.attendance_records_file = os.path.join(config.LOGS_DIR, "attendance.csv")
        self._ensure_file()

        self.monitor_thread = threading.Thread(target=self._exit_monitor, daemon=True)
        self.monitor_thread.start()

    def _load_names(self):
        names = {}
        try:
            with open(config.EMPLOYEE_NAMES_FILE, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    names[row["employee_id"].strip()] = row["name"].strip()
            print(f"[ATTENDANCE] Loaded {len(names)} employee names")
        except FileNotFoundError:
            print(f"[ATTENDANCE] Names file not found: {config.EMPLOYEE_NAMES_FILE}")
        return names

    def employee_name(self, identity):
        if identity == config.UNKNOWN_IDENTITY:
            return config.UNKNOWN_NAME
        return getattr(self, "employee_names", {}).get(str(identity), str(identity))

    def _ensure_file(self):
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, "w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(["Timestamp", "Site", "Camera", "UseCase", "Event", "Identity", "Score"])
        if not os.path.exists(self.attendance_records_file):
            with open(self.attendance_records_file, "w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(
                    ["Timestamp", "Site", "Camera", "UseCase", "Event", "Identity", "Score"]
                )

    def log_event(self, camera_id, use_case, event_type, identity, score):
        now = datetime.now().isoformat()
        site_name = config.get_site_info(camera_id)["site"]
        with open(self.attendance_file, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([now, site_name, camera_id, use_case, event_type, identity, f"{score:.4f}"])

    def _log_attendance(self, camera_id, event_type, identity, score,
                        event_timestamp):
        timestamp = datetime.fromtimestamp(event_timestamp).isoformat()
        site_name = config.get_site_info(camera_id)["site"]
        with open(self.attendance_records_file, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [timestamp, site_name, camera_id, "ATTENDANCE", event_type,
                 identity, f"{score:.4f}"]
            )
        self.log_event(camera_id, "ATTENDANCE", event_type, identity, score)

    def stream_role(self, camera_id):
        for site in config.SITES.values():
            role = site.get("attendance_streams", {}).get(camera_id)
            if role:
                return role
        return None

    def database_camera_number(self, camera_id):
        if camera_id == "N/A":
            return 0
        for site in config.SITES.values():
            number = site.get("attendance_camera_numbers", {}).get(camera_id)
            if number is not None:
                return number
        return None

    def _save_database_event(self, camera_id, event_type, identity, score,
                             event_timestamp):
        camera_number = self.database_camera_number(camera_id)
        if camera_number is None:
            print(f"[DATABASE ERROR] No camera number configured for {camera_id}")
            self.database_write_failures += 1
            return
        site_info = config.get_site_info(camera_id)
        started = time.perf_counter()
        saved = save_attendance(
            employee_id=identity,
            name=self.employee_name(identity),
            db_name=site_info["db_name"],
            site=site_info["site"],
            event_timestamp=event_timestamp,
            camera=camera_number,
            gender="unknown",
            status=event_type,
            similarity=score,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        self.database_write_ms.append(elapsed)   # enqueue time only
        if len(self.database_write_ms) > 100:
            self.database_write_ms.pop(0)
        if saved:
            print(f"[ATTENDANCE] {event_type} for {identity} sent to DB queue "
                  f"({site_info['site']}, camera {camera_number})")
        else:
            self.database_write_failures += 1
            print(f"[ATTENDANCE] {event_type} for {identity} could NOT be queued for DB (CSV only)")

    def _record(self, camera_id, event_type, identity, score, event_timestamp):
        self._log_attendance(camera_id, event_type, identity, score, event_timestamp)
        self._save_database_event(camera_id, event_type, identity, score, event_timestamp)

    def _flush_unknowns(self, now):
        """Store crossings that stayed unrecognised after the wait time."""
        for crossing_id, item in list(self.pending_unknown.items()):
            if crossing_id in self.logged_crossings:
                del self.pending_unknown[crossing_id]
            elif now - item["first_seen"] >= config.UNKNOWN_CONFIRM_SECONDS:
                self.logged_crossings.add(crossing_id)
                del self.pending_unknown[crossing_id]
                self._record(item["camera_id"], item["event_type"],
                             config.UNKNOWN_IDENTITY, item["score"],
                             item["crossing_time"])

    def process_frame(self, camera_id, frame):
        identities = self.identity_service.identify(frame, camera_id)
        self.process_results(camera_id, identities)

    def process_results(self, camera_id, identities):
        now = time.time()
        role = self.stream_role(camera_id)
        if identities is None:
            return

        with self.lock:
            self._flush_unknowns(now)
            for res in identities:
                crossing_id = res.get("crossing_id")
                crossing_time = res.get("crossing_time") or now
                if crossing_id is not None and crossing_id not in self.seen_crossings:
                    direction = res.get("direction")
                    position = res.get("crossing_x")
                    for previous in self.recent_crossings:
                        if (previous["direction"] == direction
                                and crossing_time - previous["time"] <= config.TAILGATING_SECONDS
                                and position is not None
                                and abs(position - previous["x"]) <= config.TAILGATING_DISTANCE
                                and previous["track_id"] != res.get("track_id")):
                            self.log_event(camera_id, "TAILGATING", "SUSPECTED",
                                           res.get("identity") or "UNKNOWN", 0.0)
                            break
                    self.recent_crossings.append({
                        "id": crossing_id,
                        "track_id": res.get("track_id"),
                        "direction": direction,
                        "x": position,
                        "time": crossing_time,
                    })
                    self.recent_crossings = [
                        item for item in self.recent_crossings
                        if crossing_time - item["time"] <= config.TAILGATING_SECONDS
                    ]
                    self.seen_crossings.add(crossing_id)
                if res["status"] != "KNOWN":
                    if (crossing_id is not None
                            and crossing_id not in self.logged_crossings
                            and crossing_id not in self.pending_unknown):
                        self.pending_unknown[crossing_id] = {
                            "camera_id": camera_id,
                            "first_seen": now,
                            "event_type": role or ("ENTRY" if res.get("direction") == "in" else "EXIT"),
                            "score": res.get("confidence", 0.0),
                            "crossing_time": crossing_time,
                        }
                    continue
                emp_id = res["identity"]
                if crossing_id is not None:
                    if crossing_id in self.logged_crossings:
                        continue
                    self.logged_crossings.add(crossing_id)
                    self.pending_unknown.pop(crossing_id, None)
                    event_type = role or ("ENTRY" if res.get("direction") == "in" else "EXIT")
                    self._record(camera_id, event_type, emp_id,
                                 res.get("confidence", 0.0), crossing_time)
                    self.person_states[emp_id] = {
                        "last_seen": now,
                        "last_check": now,
                        "present": res.get("direction") == "in",
                    }
                    continue
                if emp_id not in self.person_states:
                    self.person_states[emp_id] = {"last_seen": now, "last_check": now, "present": False}
                else:
                    state = self.person_states[emp_id]
                    state["last_seen"] = now

    def _exit_monitor(self):
        while True:
            time.sleep(5.0)
            now = time.time()
            with self.lock:
                self._flush_unknowns(now)
                for emp_id, state in list(self.person_states.items()):
                    if state["present"] and (now - state["last_seen"] >= config.EXIT_AFTER_SECONDS):
                        state["present"] = False
                        self.log_event("N/A", "ATTENDANCE", "EXIT_NOT_CONFIRMED",
                                       emp_id, 0.0)

    def get_database_stats(self):
        durations = self.database_write_ms
        stats = self.database.stats()
        return {
            "database_writes": len(durations),
            "database_write_failures": self.database_write_failures,
            "database_write_ms_avg": (
                sum(durations) / len(durations) if durations else 0.0
            ),
            "database_write_ms_p95": (
                sorted(durations)[int((len(durations) - 1) * 0.95)]
                if durations else 0.0
            ),
            **stats,
        }

    def close(self):
        self.database.close()