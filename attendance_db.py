import os
import queue
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import config

LOCAL_TIMEZONE = ZoneInfo("Asia/Kolkata")


class AttendanceDatabase:
    """Asynchronous MySQL writer that never blocks camera processing."""

    def __init__(self):
        self._queue = queue.Queue(maxsize=256)
        self._running = True
        self.write_count = 0
        self.write_failures = 0
        self.last_error = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def enqueue(self, event):
        try:
            self._queue.put_nowait(event)
            print(f"[DATABASE] queued {event.get('status')} for {event.get('employee_id')} -> {event.get('db_name')}")
            return True
        except queue.Full:
            self.write_failures += 1
            self.last_error = "attendance database queue is full"
            print(f"[DATABASE ERROR] {self.last_error}")
            return False

    def _connect(self, db_name=None):
        try:
            import mysql.connector
        except ImportError as exc:
            raise RuntimeError(
                "mysql-connector-python is not installed; attendance was not saved"
            ) from exc
        return mysql.connector.connect(
            host=os.environ.get("DB_HOST", "127.0.0.1"),
            port=int(os.environ.get("DB_PORT", "3306")),
            database=db_name or os.environ.get("DB_NAME", "spml_attendance"),
            user=os.environ.get("DB_USER", "spml_attendance_app"),
            password=os.environ.get("DB_PASSWORD", ""),
            connection_timeout=int(os.environ.get("DB_CONNECT_TIMEOUT", "5")),
        )

    def _check_connection(self):
        db_names = {info["db_name"] for info in config.SITE_DATABASES.values()}
        for db_name in sorted(db_names):
            try:
                conn = self._connect(db_name)
                cur = conn.cursor()
                cur.execute("SELECT name FROM employee_attendance LIMIT 1")
                cur.fetchall()
                cur.close()
                conn.close()
                print(f"[DATABASE] CONNECTED - {db_name}.employee_attendance is reachable")
            except Exception as exc:
                self.last_error = str(exc)
                print(f"[DATABASE ERROR] {db_name} check FAILED: {exc}")

    def _save(self, event):
        connection = self._connect(event.get("db_name"))
        cursor = connection.cursor()
        stored = False
        try:
            employee_id = str(event["employee_id"])
            name = str(event.get("name") or config.UNKNOWN_NAME)
            timestamp = datetime.fromtimestamp(
                float(event["event_timestamp"]), tz=LOCAL_TIMEZONE
            ).replace(tzinfo=None)
            role = str(event["status"]).upper()
            camera = int(event["camera"])
            open_row_sql = """
                SELECT id FROM employee_attendance
                WHERE employee_id = %s AND out_time IS NULL
                ORDER BY in_time DESC LIMIT 1 FOR UPDATE
            """
            if role not in ("ENTRY", "EXIT"):
                raise ValueError(f"unsupported attendance status: {role}")

            if employee_id == config.UNKNOWN_IDENTITY:
                is_exit = role == "EXIT"
                cursor.execute(
                    """
                    INSERT INTO employee_attendance
                    (employee_id, name, attendance_date, in_time, out_time, status,
                     entry_camera_id, exit_camera_id, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    """,
                    (employee_id, name, timestamp.date(), timestamp,
                     timestamp if is_exit else None,
                     "OUT" if is_exit else "IN",
                     camera, camera if is_exit else None),
                )
                stored = True
                detail = f"UNKNOWN {role} row inserted (id={cursor.lastrowid})"
            elif role == "ENTRY":
                cursor.execute(open_row_sql, (employee_id,))
                if cursor.fetchone() is None:
                    cursor.execute(
                        """
                        INSERT INTO employee_attendance
                        (employee_id, name, attendance_date, in_time, status,
                         entry_camera_id, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, 'IN', %s, NOW(), NOW())
                        """,
                        (employee_id, name, timestamp.date(), timestamp, camera),
                    )
                    stored = True
                    detail = f"IN row inserted (id={cursor.lastrowid}, in_time={timestamp})"
                else:
                    detail = "SKIPPED - employee already has an open IN record"
            else:  # EXIT
                cursor.execute(open_row_sql, (employee_id,))
                row = cursor.fetchone()
                if row is None:
                    event["_attempts"] = 99  # retrying cannot fix this
                    raise RuntimeError(f"unmatched exit for employee {employee_id}")
                cursor.execute(
                    """
                    UPDATE employee_attendance
                    SET out_time = %s, status = 'OUT',
                        exit_camera_id = %s, updated_at = NOW()
                    WHERE id = %s AND out_time IS NULL
                    """,
                    (timestamp, camera, row[0]),
                )
                stored = cursor.rowcount == 1
                detail = f"OUT row updated (id={row[0]}, out_time={timestamp})"
            connection.commit()
            where = f"{event.get('site')}/{event.get('db_name')}"
            if stored:
                self.write_count += 1
                print(f"[DATABASE] STORED {role} for {employee_id} in {where} -> {detail}")
            else:
                print(f"[DATABASE] NOT STORED {role} for {employee_id} in {where} -> {detail}")
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def _run(self):
        self._check_connection()
        while self._running or not self._queue.empty():
            try:
                event = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._save(event)
            except Exception as exc:
                attempts = int(event.get("_attempts", 0))
                if attempts < 2 and self._running:
                    event["_attempts"] = attempts + 1
                    print(f"[DATABASE] write failed ({exc}); retry {attempts + 1}/2")
                    time.sleep(1.0)
                    try:
                        self._queue.put_nowait(event)
                    except queue.Full:
                        self.write_failures += 1
                        self.last_error = "attendance database retry queue is full"
                        print(f"[DATABASE ERROR] NOT STORED - {self.last_error}")
                else:
                    self.write_failures += 1
                    self.last_error = str(exc)
                    print(f"[DATABASE ERROR] NOT STORED {event.get('status')} "
                          f"for {event.get('employee_id')}: {exc}")
            finally:
                self._queue.task_done()

    def stats(self):
        return {
            "database_writes": self.write_count,
            "database_write_failures": self.write_failures,
            "database_queue_size": self._queue.qsize(),
            "database_last_error": self.last_error,
        }

    def close(self):
        self._running = False
        self._worker.join(timeout=5.0)


_default_database = None
_default_lock = threading.Lock()


def get_database():
    global _default_database
    with _default_lock:
        if _default_database is None:
            _default_database = AttendanceDatabase()
        return _default_database


def save_attendance(employee_id, event_timestamp, camera, gender, status,
                    similarity, name=None, db_name=None, site=None):
    """Compatibility API: enqueue a validated attendance transaction."""
    return get_database().enqueue({
        "employee_id": employee_id,
        "name": name,
        "db_name": db_name,
        "site": site,
        "event_timestamp": event_timestamp,
        "camera": camera,
        "gender": gender,
        "status": status,
        "similarity": similarity,
    })