import os
from urllib.parse import quote
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_loaded = load_dotenv(_ENV_PATH, encoding="utf-8-sig", override=True)
print(f"[CONFIG] .env path={_ENV_PATH} loaded={_loaded} "
      f"DB_PASSWORD set={bool(os.environ.get('DB_PASSWORD'))}")

# Deployment configuration:
# - Camera URLs remain in .env as CAM_<CAMERA_ID>_SUB, except Pune URLs which
#   are assembled by get_camera_url().
# - Only cameras in attendance_streams create attendance records.
SITE_1_KOLKATA = {
    "camera_ids": [f"S1_CAM_{i:02d}" for i in range(1, 17)],
    "entrance_cameras": ["S1_CAM_01", "S1_CAM_02"],
    "attendance_streams": {
        "S1_CAM_01": "ENTRY",
        "S1_CAM_02": "EXIT",
    },
    "attendance_camera_numbers": {
        "S1_CAM_01": 1,
        "S1_CAM_02": 2,
    },
}

SITE_2_PUNE = {
    "camera_ids": [f"PUNE_CAM_{i:02d}" for i in range(1, 7)],
    "entrance_cameras": ["PUNE_CAM_01", "PUNE_CAM_02"],
    "attendance_streams": {
        "PUNE_CAM_01": "ENTRY",
        "PUNE_CAM_02": "EXIT",
    },
    "attendance_camera_numbers": {
        "PUNE_CAM_01": 1,
        "PUNE_CAM_02": 2,
    },
}

SITES = {
    "site_1_kolkata": SITE_1_KOLKATA,
    "site_2_pune": SITE_2_PUNE,
    "all_sites": {
        "camera_ids": (
            SITE_1_KOLKATA["camera_ids"] + SITE_2_PUNE["camera_ids"]
        ),
        "entrance_cameras": (
            SITE_1_KOLKATA["entrance_cameras"] + SITE_2_PUNE["entrance_cameras"]
        ),
        "attendance_streams": {
            **SITE_1_KOLKATA["attendance_streams"],
            **SITE_2_PUNE["attendance_streams"],
        },
        "attendance_camera_numbers": {
            **SITE_1_KOLKATA["attendance_camera_numbers"],
            **SITE_2_PUNE["attendance_camera_numbers"],
        },
    },
    
}


def get_site(site_name=None):
    """Return the configured deployment site without duplicating routing logic."""
    selected = site_name or os.environ.get("SPML_SITE", "all_sites")
    if selected not in SITES:
        raise ValueError(f"Unknown SPML_SITE: {selected}")
    return SITES[selected]

def get_camera_url(cam_id, stream="SUB"):
    # Pune cameras use their own RTSP URL format.
    if cam_id.startswith("PUNE_CAM_"):
        ip = os.environ.get("PUNE_IP", "").strip()
        port = os.environ.get("PUNE_PORT", "554").strip()
        username = os.environ.get("PUNE_USERNAME", "").strip()
        password = os.environ.get("PUNE_PASSWORD", "")

        if not all([ip, port, username, password]):
            raise ValueError(
                "Missing PUNE_IP, PUNE_PORT, PUNE_USERNAME "
                "or PUNE_PASSWORD in .env"
            )

        channel = int(cam_id.rsplit("_", 1)[1])
        user = quote(username, safe="")
        pwd = quote(password, safe="")

        subtype = 0 if stream.upper() == "MAIN" else 1
        return (
            f"rtsp://{user}:{pwd}@{ip}:{port}"
            f"/cam/realmonitor?channel={channel}&subtype={subtype}"
        )

    # Existing site's camera URLs remain unchanged.
    env_name = f"CAM_{cam_id}_{stream}"
    url = os.environ.get(env_name, "").strip()

    if not url or "dummy" in url.lower():
        raise ValueError(
            f"Missing or invalid camera URL: {env_name}"
        )

    return url


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IDENTITY_DIR = os.path.join(BASE_DIR, "identity_data")
KOLKATA_IDENTITY_DB = os.path.join(IDENTITY_DIR, "employee_embeddings.npz")
PUNE_IDENTITY_DB = os.path.join(IDENTITY_DIR, "employee_embeddings.npz")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Identity & Attendance Settings
FACE_MODEL_NAME = "buffalo_l"
SIMILARITY_THRESHOLD = 0.60
UNKNOWN_SIMILARITY_THRESHOLD = 0.40
CHECK_INTERVAL_SECONDS = 60.0
EXIT_AFTER_SECONDS = 60.0

# Use Case Thresholds
CROWD_MIN_PEOPLE = 3
CROWD_PROXIMITY_THRESHOLD = 1.5  # Average-person-height units
CROWD_CONFIRMATION_SECONDS = 1.0
CROWD_RELEASE_SECONDS = 1.0
ABNORMAL_ACTIVITY_CONFIRMATION_SECONDS = 0.6
PERSON_DETECTION_CONFIDENCE = 0.45
POSE_KEYPOINT_MIN_CONFIDENCE = 0.35
POSE_DETECTION_CONFIDENCE = 0.18
PERSON_DETECTION_MODEL = "yolo11m.pt"
PERSON_DETECTION_IMAGE_SIZE = 1280
PERSON_DETECTION_TILE_OVERLAP = 0.10
POSE_MIN_BOX_HEIGHT = 120
INFERENCE_DEVICE = "cpu"
PERSON_TRACK_MATCH_THRESHOLD = 0.9
JUMP_CONFIRMATION_FRAMES = 3
ENABLE_JUMP_DETECTION = False
ENABLE_ERRATIC_DETECTION = False
FIRE_SMOKE_CONFIDENCE = 0.25
FIRE_INFERENCE_INTERVAL_SECONDS = 1.5
TAILGATING_SECONDS = 3.0   # UC6
TAILGATING_DISTANCE = 200  # Pixels
TAMPER_FRAMES_BASELINE = 50
MAX_DISPLAY_FRAME_AGE_SECONDS = 0.25
IDENTITY_TRACK_GRACE_SECONDS = 2.0
IDENTITY_DEBUG = False
EVIDENCE_DEDUP_SECONDS = 300.0

# Office Hours & After-Hours Settings
OFFICE_START_TIME = "09:30"
OFFICE_END_TIME = "19:30"
AFTER_HOURS_LOG_INTERVAL_SECONDS = 300.0  # 5 minutes
AFTER_HOURS_MIN_STAY_SECONDS = 600.0
AFTER_HOURS_GAP_SECONDS = 120.0
IDENTITY_RETRY_SECONDS = 2.0
IDENTITY_MIN_BOX_HEIGHT = 110
AMBIGUOUS_IDENTITY_CONFIRMATIONS = 3
RUN_SPEED_BH_PER_S = 2.2
RUN_CONFIRM_SAMPLES = 3

# MySQL attendance settings are read from .env; no password belongs in source.
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_NAME = os.environ.get("DB_NAME", "spml_attendance")
DB_USER = os.environ.get("DB_USER", "spml_attendance_app")
DB_CONNECT_TIMEOUT = int(os.environ.get("DB_CONNECT_TIMEOUT", "5"))

# Worker limits
MAX_WORKERS = 4
INFERENCE_QUEUE_SIZE = 1
INFERENCE_FPS = 1
INFERENCE_BATCH_SIZE = 1
INFERENCE_BATCH_WINDOW_SECONDS = 0.02

# ---- Per-site databases ----
SITE_DATABASES = {
    "S1_CAM_":   {"site": "KOLKATA", "db_name": os.environ.get("DB_NAME_KOLKATA", "spml_kolkata")},
    "PUNE_CAM_": {"site": "PUNE",    "db_name": os.environ.get("DB_NAME_PUNE", "spml_pune")},
    "TEST_CAM_": {"site": "OFFICE_TEST", "db_name": DB_NAME},
}
DEFAULT_SITE_INFO = {"site": "SITE_X", "db_name": DB_NAME}

def get_site_info(camera_id):
    for prefix, info in SITE_DATABASES.items():
        if str(camera_id).startswith(prefix):
            return info
    return DEFAULT_SITE_INFO

# Unknown people
UNKNOWN_IDENTITY = "UNKNOWN"
UNKNOWN_NAME = "Unknown"
UNKNOWN_CONFIRM_SECONDS = 3.0

# employee_id -> name file
EMPLOYEE_NAMES_FILE = os.path.join(IDENTITY_DIR, "employee_names.csv")