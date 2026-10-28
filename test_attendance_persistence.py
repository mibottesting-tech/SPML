import csv
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import attendance


class AttendancePersistenceTests(unittest.TestCase):
    def test_delayed_known_crossing_is_written_to_csv_and_database(self):
        with tempfile.TemporaryDirectory() as logs_dir:
            monitor = attendance.AttendanceMonitor.__new__(
                attendance.AttendanceMonitor
            )
            monitor.person_states = {}
            monitor.logged_crossings = set()
            monitor.seen_crossings = set()
            monitor.recent_crossings = []
            monitor.lock = threading.Lock()
            monitor.database_write_ms = []
            monitor.database_write_failures = 0
            monitor.pending_unknown = {}
            monitor.employee_names = {}
            monitor.attendance_file = os.path.join(logs_dir, "events.csv")
            monitor.attendance_records_file = os.path.join(
                logs_dir, "attendance.csv"
            )
            monitor._ensure_file()

            camera_id = "TEST_CAM_01"
            crossing_id = f"{camera_id}:1"
            crossing_time = time.time()
            base_result = {
                "crossing_id": crossing_id,
                "crossing_time": crossing_time,
                "direction": "in",
                "crossing_x": 100,
                "track_id": "track-1",
            }
            sites = {
                "test": {
                    "attendance_streams": {camera_id: "ENTRY"},
                    "attendance_camera_numbers": {camera_id: 1},
                }
            }

            with patch.object(attendance.config, "SITES", sites), patch.object(
                attendance, "save_attendance", return_value=True
            ) as save_database:
                monitor.process_results(
                    camera_id,
                    [{**base_result, "status": "UNKNOWN", "identity": None}],
                )
                self.assertFalse(save_database.called)

                monitor.process_results(
                    camera_id,
                    [{
                        **base_result,
                        "crossing_time": None,
                        "status": "KNOWN",
                        "identity": "EMP001",
                        "confidence": 0.91,
                    }],
                )

            save_database.assert_called_once()
            args = save_database.call_args.kwargs
            self.assertEqual(args["employee_id"], "EMP001")
            self.assertEqual(args["camera"], 1)
            self.assertEqual(args["status"], "ENTRY")

            with open(monitor.attendance_records_file, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Identity"], "EMP001")
            self.assertEqual(rows[0]["Event"], "ENTRY")
            self.assertEqual(rows[0]["UseCase"], "ATTENDANCE")

            with open(monitor.attendance_file, newline="", encoding="utf-8") as file:
                event_rows = list(csv.DictReader(file))
            attendance_rows = [
                row for row in event_rows if row["UseCase"] == "ATTENDANCE"
            ]
            self.assertEqual(len(attendance_rows), 1)
            self.assertEqual(attendance_rows[0]["Identity"], "EMP001")


if __name__ == "__main__":
    unittest.main()
