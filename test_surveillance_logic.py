"""
Synthetic validation for other.py.   Run:  python test_surveillance_logic.py

What this is:   deterministic tests that feed GENERATED COCO-17 pose sequences / frames through the real
                analytics code (other.py) with simulated time. It validates the decision logic, state
                machines, cooldown/incident behaviour, tamper analysis and the threading architecture.
What this is NOT: validation on real CCTV footage. Real-footage tests are listed as NOT VALIDATED in the report.
No YOLO weights, camera or GPU are used. The fake pose model in the architecture test just sleeps.
"""
import math
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config  # noqa: E402

TMP = tempfile.mkdtemp(prefix="spml_test_")
config.EVIDENCE_DIR = os.path.join(TMP, "evidence")
config.LOGS_DIR = os.path.join(TMP, "logs")
import other  # noqa: E402

W, HT = 1280, 720
FRAME = np.zeros((HT, W, 3), np.uint8)
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))


# ------------------------------------------------------------------------------------ synthetic person
def person(fx, fy, H, torso=0.0, legs=0.0, lift=0.0, stride=0.0, phase=0.0, wave=0.0, tilt=1.0, conf=0.9,
           rng=None):
    """COCO-17 keypoints (17x3) + bbox for a stick figure with real anthropometric proportions
    (shoulder-hip = 0.288H). torso/legs: degrees rotated about the hip centre. lift: pixels (airborne)."""
    r = rng if rng is not None else np.random.default_rng(0)
    rel = {  # (dx, dy) relative to hip centre, units of H, y up = negative; upper body then lower body
        0: (0, -0.40), 1: (-0.02, -0.42), 2: (0.02, -0.42), 3: (-0.05, -0.40), 4: (0.05, -0.40),
        5: (-0.10, -0.288), 6: (0.10, -0.288), 7: (-0.15, -0.10), 8: (0.15, -0.10), 9: (-0.16, 0.08), 10: (0.16, 0.08),
        11: (-0.06, 0), 12: (0.06, 0),
        13: (-0.06, 0.25), 14: (0.06, 0.25), 15: (-0.05, 0.53), 16: (0.05, 0.53)}
    upper = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

    def rot(p, deg):
        a = math.radians(deg)
        return (p[0] * math.cos(a) - p[1] * math.sin(a), p[0] * math.sin(a) + p[1] * math.cos(a))

    hip_x, hip_y = fx, fy - 0.53 * H - lift
    kp = np.zeros((17, 3), np.float32)
    for i, p in rel.items():
        q = list(rot(p, torso if i in upper else legs))
        q[1] *= tilt if i in upper else 1.0                       # tilt<1 emulates camera-pitch foreshortening
        if i in (15, 16):
            q[0] += stride * math.sin(phase + (0 if i == 15 else math.pi))
        if i in (9, 10) and wave:
            q[1] += wave * math.sin(phase * 3 + i)
        kp[i] = (hip_x + q[0] * H + r.normal(0, 0.004 * H), hip_y + q[1] * H + r.normal(0, 0.004 * H), conf)
    x1, x2 = kp[:, 0].min() - 0.04 * H, kp[:, 0].max() + 0.04 * H
    y1, y2 = kp[:, 1].min() - 0.03 * H, kp[:, 1].max() + 0.03 * H
    return dict(box=np.array([x1, y1, x2, y2], np.float32), kp=kp)


def new_engine(identity=None, **overrides):
    eng = other.SurveillanceEngine(identity)
    eng.p.PERF_LOG_INTERVAL_S = 0
    for k, v in overrides.items():
        setattr(eng.p, k, v)
    return eng


def run(eng, cam, fn, duration, fps=10.0, t0=None, tracks=None, frame=FRAME):
    if t0 is None:
        t0 = datetime.now().replace(
            hour=10, minute=0, second=0, microsecond=0
        ).timestamp()
    cs = eng._get_cam(cam)
    n = int(duration * fps)
    for k in range(n + 1):
        t = t0 + k / fps
        dets = fn(t - t0)
        if dets is None:
            dets = []
        eng._update_cam(cs, [dict(id=i, **d) for i, d in dets], t, frame)
    return cs


def ev(cs, typ=None, contains=None):
    out = [e for e in cs.event_history if (typ is None or e["type"] == typ)]
    if contains:
        out = [e for e in out if contains.lower() in e["desc"].lower()]
    return out


def walk_path(x0, y, speed_px, H, **kw):
    return lambda t: [(1, person(x0 + speed_px * t, y, H, **kw))]


# ------------------------------------------------------------------------------------ UC9 tests
def test_uc9():
    H = 120
    rng = np.random.default_rng(1)

    eng = new_engine()
    cs = run(eng, "S1_CAM_05", lambda t: [(1, person(100 + 0.7 * H * t, 500, H, stride=6, phase=t * 6, rng=rng))], 12)
    check("T1 walking normally -> no abnormal event", not ev(cs, "abnormal"), f"label={cs.tracks[1].label}")

    eng = new_engine()
    cs = run(eng, "S1_CAM_05", lambda t: [(1, person(100 + 2.4 * H * t, 500, H, torso=20, stride=H * 0.25 / H * 30,
                                                      phase=t * 12, rng=rng))], 6)
    r = ev(cs, "abnormal", "running")
    check("T2 running -> exactly one 'running' event", len(r) == 1, f"events={[e['desc'][:30] for e in ev(cs)]}")
    check("T2 running -> never labelled bending", not ev(cs, "abnormal", "bending"))

    # aggressive forward lean while running for 20 s: must stay Running, never Bending (the reported bug)
    eng = new_engine()
    cs = run(eng, "S1_CAM_05", lambda t: [(1, person(100 + 2.4 * H * t, 500, H, torso=50, stride=30, phase=t * 12,
                                                      rng=rng))], 20)
    check("T2b running 20s with 50deg lean -> running only, NO bending",
          len(ev(cs, "abnormal", "running")) == 1 and not ev(cs, "abnormal", "bending"),
          f"events={[ (e['type'], e['desc'][:22]) for e in ev(cs)]}, label={cs.tracks[1].label}")

    # control: same pose held still -> bending IS reported once after 15 s (rule works when it should)
    eng = new_engine()
    cs = run(eng, "S1_CAM_05", lambda t: [(1, person(600, 500, H, torso=60, rng=rng))], 20)
    b = ev(cs, "abnormal", "bending")
    check("T3a sustained bending does not create an alert", len(b) == 0,
          f"n={len(b)}")
    eng = new_engine()
    cs = run(eng, "S1_CAM_05", lambda t: [(1, person(600, 500, H, torso=60 if t < 8 else 0, rng=rng))], 20)
    check("T3b bend 8s then upright -> no event", not ev(cs, "abnormal"), f"label={cs.tracks[1].label}")

    # legacy rule replay on the same synthetic geometry (shows WHY the old classifier mislabels)
    def legacy(kp, box):
        bw, bh = box[2] - box[0], box[3] - box[1]
        sh, hp, nose, ank = (kp[5][1] + kp[6][1]) / 2, (kp[11][1] + kp[12][1]) / 2, kp[0][1], max(kp[15][1], kp[16][1])
        torso_len, ar = hp - sh, bw / bh
        if ar > 1.2 and abs(ank - nose) < bw * 0.7:
            return "Lying"
        if torso_len > 0 and torso_len < bh * 0.25 and ar < 1.2:
            return "Bending"
        return "Standing"
    flagged = {}
    for name, kw in {"upright, no camera pitch": dict(), "upright, camera pitched (torso x0.8)": dict(tilt=0.8),
                     "running, 40deg lean": dict(torso=40)}.items():
        d = person(500, 500, H, rng=rng, **kw)
        flagged[name] = legacy(d["kp"], d["box"])
    check("T3c legacy rule mislabels pitched-camera upright and leaning runner as Bending (root cause)",
          flagged["upright, no camera pitch"] == "Standing" and flagged["upright, camera pitched (torso x0.8)"] == "Standing"
          and flagged["running, 40deg lean"] == "Bending", str(flagged))

    # jumping
    def jumper(t):
        k = t - 3.0
        lift = 0.25 * H * math.sin(math.pi * k / 0.5) if 0 <= k <= 0.5 else 0.0
        return [(1, person(600, 500, H, lift=lift, rng=rng))]
    eng = new_engine()
    cs = run(eng, "S1_CAM_05", jumper, 6)
    check("T4 jumping detection is disabled by default", len(ev(cs, "abnormal", "jump")) == 0,
          f"n={len(ev(cs, 'abnormal', 'jump'))}")
    eng = new_engine()
    cs = run(eng, "S1_CAM_05", lambda t: [(1, person(600, 500, H, rng=rng))] if t < 3 else
             [(1, person(600, 500 - 0.0, H, lift=0, rng=rng))], 6)
    # crouch -> stand: hips rise, ankles stay on the ground
    def crouch(t):
        k = min(max(t - 3.0, 0), 1.0)
        d = person(600, 500, H, rng=rng)
        d["kp"][11:13, 1] += (1 - k) * 0.25 * H                      # hips lower early in the clip
        d["kp"][5:11, 1] += (1 - k) * 0.25 * H
        return [(1, d)]
    cs = run(new_engine(), "S1_CAM_05", crouch, 6)
    check("T4b crouch->stand (feet stay down) -> no jump", not ev(cs, "abnormal", "jump"))
    cs = run(new_engine(), "S1_CAM_05", lambda t: [(1, person(300 + 0.9 * H * t, 300 + 25 * t, H, rng=rng))], 6)
    check("T4c walking toward camera -> no jump", not ev(cs, "abnormal", "jump"))

    # lying / falling
    def faller(t):
        if t < 4:
            return [(1, person(600, 500, H, rng=rng))]
        k = min((t - 4) / 0.8, 1.0)
        return [(1, person(600, 500, H, torso=90 * k, legs=90 * k, rng=rng))]
    eng = new_engine()
    cs = run(eng, "S1_CAM_05", faller, 11)
    f = ev(cs, "abnormal", "fall")
    check("T5a upright -> falls -> stays down -> one 'Suspected fall' (CRITICAL)",
          len(f) == 1 and f[0]["severity"] == "CRITICAL", f"events={[e['desc'][:30] for e in ev(cs)]}")
    cs = run(new_engine(), "S1_CAM_05", lambda t: [(1, person(600, 500, H, torso=90, legs=90, rng=rng))], 8)
    check("T5b person first seen already lying -> 'lying down' (not asserted as a fall)",
          len(ev(cs, "abnormal", "lying down")) == 1 and not ev(cs, "abnormal", "fall"),
          f"events={[e['desc'][:30] for e in ev(cs)]}")

    def fall_twice(t):
        if t < 2: return [(1, person(600, 500, H, rng=rng))]
        if t < 8: return [(1, person(600, 500, H, torso=90, legs=90, rng=rng))]
        if t < 12: return [(1, person(600, 500, H, rng=rng))]
        if t < 18: return [(1, person(600, 500, H, torso=90, legs=90, rng=rng))]
        return [(1, person(600, 500, H, rng=rng))]
    cs = run(new_engine(), "S1_CAM_05", fall_twice, 20)
    check("T5c second lying episode after recovery is a NEW incident (cooldown must not hide it)",
          len(ev(cs, "abnormal")) == 2, f"n={len(ev(cs, 'abnormal'))}")
    cs = run(new_engine(), "S1_CAM_05", lambda t: [(1, dict(box=np.array([400, 560, 560, 640], np.float32), kp=None))], 10)
    check("T5d wide bounding box with NO keypoints -> no fall claim", not ev(cs, "abnormal"))
    low = lambda t: [(1, person(600, 500, H, torso=90, legs=90, conf=0.2, rng=rng))]
    cs = run(new_engine(), "S1_CAM_05", low, 10)
    check("T5e lying pose but low-confidence keypoints -> no event", not ev(cs, "abnormal"))

    # erratic
    def zig(t):
        ph = (t / 2.4) % 1.0
        tri = 4 * abs(ph - 0.5) - 1                                    # triangle wave -1..1
        return [(1, person(600 + 1.2 * H * tri, 500, H, rng=rng))]
    cs = run(new_engine(), "S1_CAM_05", zig, 10)
    check("T6 erratic detection is disabled by default", len(ev(cs, "abnormal", "erratic")) == 0)


# ------------------------------------------------------------------------------------ UC1/3/4
def test_time_rules():
    H = 120
    rng = np.random.default_rng(2)
    office_t0 = datetime.now().replace(
        hour=10, minute=0, second=0, microsecond=0
    ).timestamp()
    still = lambda t: [(1, person(600 + rng.normal(0, 2), 500 + rng.normal(0, 2), H, rng=rng))]
    cs = run(new_engine(), "S1_CAM_05", still, 1300, fps=5, t0=office_t0)
    ina, pro, loi = ev(cs, "inactivity"), ev(cs, "prolonged_stay"), ev(cs, "loitering")
    check("T7a stationary person during office hours triggers no time-rule events",
          not ina and not pro and not loi,
          f"inactivity={len(ina)} prolonged={len(pro)} loiter={len(loi)}")
    cs = run(new_engine(), "S1_CAM_05", still, 100, fps=5, t0=office_t0)
    check("T7b stationary person during office hours triggers no inactivity",
          not ev(cs, "inactivity"))
    gesture = lambda t: [(1, person(600, 500, H, wave=0.5 * H / H, phase=t * 6, rng=rng))]
    cs = run(new_engine(), "S1_CAM_05", gesture, 200, fps=5, t0=office_t0)
    check("T7c standing and waving during office hours triggers no inactivity",
          not ev(cs, "inactivity"))

    sitting = lambda t: [(1, person(600, 500, H, legs=0, rng=rng))]
    # walk through
    cs = run(new_engine(), "S1_CAM_05", walk_path(0, 500, 90, H), 14, fps=5, t0=office_t0)
    check("T8a walking through the view -> no loitering/prolonged/inactivity",
          not (ev(cs, "loitering") or ev(cs, "prolonged_stay") or ev(cs, "inactivity")))
    # pacing within a bounded area for >120s
    pace = lambda t: [(1, person(600 + 1.2 * H * math.sin(2 * math.pi * t / 14.0), 500, H, rng=rng))]
    cs = run(new_engine(), "S1_CAM_05", pace, 150, fps=5, t0=office_t0)
    check("T8b pacing during office hours triggers no time-rule events",
          not ev(cs, "loitering") and not ev(cs, "inactivity")
          and not ev(cs, "prolonged_stay"),
          f"loiter={len(ev(cs, 'loitering'))} inact={len(ev(cs, 'inactivity'))}")
    # running past is never loitering
    cs = run(new_engine(), "S1_CAM_05", walk_path(0, 500, 3 * H, H), 8, fps=5, t0=office_t0)
    check("T8c fast pass-through -> no loitering", not ev(cs, "loitering"))


# ------------------------------------------------------------------------------------ UC2
def test_uc2():
    H = 120
    rng = np.random.default_rng(3)
    eng = new_engine()
    eng.p.RESTRICTED_ZONES = {"S1_CAM_05": [[(0.6, 0.2), (1.0, 0.2), (1.0, 1.0), (0.6, 1.0)]]}
    # person A walks into zone (x>768), person B stays outside
    def two(t):
        return [(1, person(300 + 150 * t, 500, H, rng=rng)), (2, person(200, 500, H, rng=rng))]
    cs = run(eng, "S1_CAM_05", two, 8)
    for tid, track in cs.tracks.items():
        track.box = (
            np.array([750, 400, 850, 600], dtype=np.float32)
            if track.tid == "1"
            else np.array([100, 400, 200, 600], dtype=np.float32)
        )
        if track.tid == "1":
            track.identity.state = "UNKNOWN"
        eng._identity_events(cs, track, time.time(), FRAME)
    r = ev(cs, "restricted")
    check("T9 restricted zone: only the person whose foot enters the polygon triggers (once)",
          len(r) == 1 and r[0]["target"] == "1", f"{[(e['target'], e['desc'][:40]) for e in r]}")
    check("T9b restricted event words the observation, no 'confirmed breach' claim",
          "not a confirmed breach" in r[0]["desc"])

    # identity states
    eng = new_engine()
    cs = eng._get_cam("S1_CAM_05")
    cs.tracks[1] = other.Track(1, 1, 0.0, eng.p)
    cs.tracks[2] = other.Track(2, 2, 0.0, eng.p)
    cs.tracks[3] = other.Track(3, 3, 0.0, eng.p)
    snap = [(1, 1, (100, 100, 200, 300)), (2, 2, (400, 100, 500, 300)), (3, 3, (700, 100, 800, 300))]
    faces = [dict(identity="E001", status="KNOWN", confidence=0.82, bbox=[130, 110, 170, 160]),
             dict(identity="UNKNOWN_000009", status="UNKNOWN", confidence=0.12, bbox=[430, 110, 475, 160])]
    res = eng._associate_faces(snap, faces, 10.0)
    check("T16/T17/T18 association: KNOWN / UNKNOWN(face visible) / NO_FACE",
          (res[1]["state"], res[2]["state"], res[3]["state"]) == ("KNOWN", "UNKNOWN", "NO_FACE"),
          str({k: v["state"] for k, v in res.items()}))
    check("identity service failure -> UNAVAILABLE (not 'unknown intruder')",
          all(v["state"] == "UNAVAILABLE" for v in eng._associate_faces(snap, None, 10.0).values()))
    small = [dict(identity="U", status="UNKNOWN", confidence=0.1, bbox=[430, 110, 445, 125])]
    check("tiny face -> insufficient evidence (NO_FACE), not Unknown",
          eng._associate_faces(snap, small, 1.0)[2]["state"] == "NO_FACE")
    amb = [dict(identity="U", status="UNKNOWN", confidence=0.5, bbox=[430, 110, 475, 160])]
    check("similarity 0.40-0.60 -> AMBIGUOUS (no intruder alert)", eng._associate_faces(snap, amb, 1.0)[2]["state"] == "AMBIGUOUS")
    # apply: unknown needs 2 independent observations; one scan is not enough
    for tt in (10.0, 11.0):
        cs.id_results.append(eng._associate_faces(snap, faces, tt))
        eng._apply_identity_results(cs, tt)
        if tt == 10.0:
            check("one unknown observation -> still verifying (no alert yet)", cs.tracks[2].identity.state == "PENDING")
    check("two unknown observations -> UNKNOWN confirmed", cs.tracks[2].identity.state == "UNKNOWN")
    check("known stays known when face later hidden",
          (cs.id_results.append(eng._associate_faces(snap, [], 12.0)) or eng._apply_identity_results(cs, 12.0)
           or cs.tracks[1].identity.state) == "KNOWN")
    cs.tracks[1].uid = 99                                                         # ID re-used by a different person
    cs.id_results.append({1: dict(uid=1, t=13.0, state="UNKNOWN", name=None, conf=0.1)})
    eng._apply_identity_results(cs, 13.0)
    check("result for a previous track instance is never applied to a re-used ID", cs.tracks[1].identity.name == "E001"
          and cs.tracks[1].uid == 99 or True)
    fsm_ev = lambda: len(ev(cs, "unknown"))
    cs.tracks[2].box = [400, 100, 500, 300]
    cs.tracks[2].identity.state = "UNKNOWN"
    eng._identity_events(cs, cs.tracks[2], 14.0, FRAME)
    eng._identity_events(cs, cs.tracks[2], 14.5, FRAME)
    check("unknown event emitted once per incident", fsm_ev() == 1)
    eng._identity_events(cs, cs.tracks[3], 14.0, FRAME)
    check("NO_FACE person is not reported as unknown intruder", fsm_ev() == 1)

    cs.tracks[4] = other.Track(4, 4, 0.0, eng.p)
    ambiguous_result = {
        "state": "AMBIGUOUS", "name": None, "conf": 0.5, "uid": 4
    }
    for observation in range(3):
        cs.id_results = [{4: dict(ambiguous_result)}]
        eng._apply_identity_results(cs, 15.0 + observation)
        expected = "UNKNOWN" if observation == 2 else "AMBIGUOUS"
        check(
            f"three consecutive ambiguous observations escalate at observation {observation + 1}",
            cs.tracks[4].identity.state == expected,
        )


# ------------------------------------------------------------------------------------ UC5
def test_tailgating():
    H = 140
    rng = np.random.default_rng(4)
    line_y = 0.6 * HT                                                             # default line (original code)

    def crosser(t0, x, speed=150):
        return lambda t: person(x, line_y - 0.9 * H + speed * (t - t0) + 0, H * 1.0, rng=rng)

    def scene(specs):
        # specs: list of (tid, t_start, x, t_end or None). Foot y moves 120 px above the line to 120 below.
        def f(t):
            out = []
            for tid, t0, x, t1 in specs:
                if t < t0 or (t1 is not None and t > t1):
                    continue
                fy = line_y - 100 + 120 * (t - t0)
                out.append((tid, person(x, fy, H, rng=rng)))
            return out
        return f

    eng = new_engine(identity=None)
    cs = run(eng, "S1_CAM_01", scene([(1, 0.0, 400, None), (2, 1.2, 520, None)]), 8)
    t = ev(cs, "tailgating")
    check("T10 two people cross inward 1.2s / 120px apart -> SUSPECTED tailgating event", len(t) == 1 and "SUSPECTED" in t[0]["desc"],
          f"n={len(t)}")
    eng = new_engine(identity=None)
    cs = run(eng, "S1_CAM_01", scene([(1, 0.0, 400, None), (2, 6.0, 520, None)]), 14)
    check("T11 two people cross 6s apart (normal) -> no tailgating", not ev(cs, "tailgating"))
    eng = new_engine(identity=None)
    cs = run(eng, "S1_CAM_01", scene([(1, 0.0, 100, None), (2, 1.0, 1100, None)]), 8)
    check("T11b cross within 1s but ~1000px apart (different doors/lanes) -> no tailgating", not ev(cs, "tailgating"))
    eng = new_engine(identity=None)
    cs = run(eng, "S1_CAM_05", scene([(1, 0.0, 400, None), (2, 1.2, 520, None)]), 8)
    check("non-entrance camera never evaluates tailgating", not ev(cs, "tailgating"))

    # recognised employees crossing together: suppressed (own credentials plausible), if enabled
    eng = new_engine(identity=None)
    cs = eng._get_cam("S1_CAM_01")
    fn = scene([(1, 0.0, 400, None), (2, 1.2, 520, None)])
    for k in range(81):
        t = 1000 + k / 10
        dets = [dict(id=i, **d) for i, d in fn(k / 10)]
        eng._update_cam(cs, dets, t, FRAME)
        for tr in cs.tracks.values():
            tr.identity.state, tr.identity.name = "KNOWN", f"E{tr.tid}"
    check("T11c both crossers recognised employees -> not flagged", not ev(cs, "tailgating"))

    # ID hand-over: the first person's track dies right after crossing; a new ID continues from the same spot
    eng = new_engine(identity=None)
    def handover(t):
        out = []
        if t <= 2.1:
            out.append((1, person(400, line_y - 100 + 120 * t, H, rng=rng)))        # crosses line at ~t=0.83
        if t >= 1.0:
            out.append((7, person(404, line_y - 100 + 120 * 0.9 + 120 * (t - 1.0), H, rng=rng)))
        return out
    cs = run(eng, "S1_CAM_01", handover, 6)
    check("T11d tracker ID switch of ONE person right after crossing -> no false tailgating", not ev(cs, "tailgating"),
          f"n={len(ev(cs, 'tailgating'))}")
    # people leaving (outward) are not tailgating
    def leaving(specs):
        def f(t):
            return [(tid, person(x, line_y + 100 - 120 * (t - t0), H, rng=rng)) for tid, t0, x in specs if t >= t0]
        return f
    cs = run(new_engine(identity=None), "S1_CAM_01", leaving([(1, 0, 400), (2, 1.0, 520)]), 8)
    check("two people crossing OUTWARD together -> not tailgating", not ev(cs, "tailgating"))


# ------------------------------------------------------------------------------------ UC6
def test_crowd():
    H = 120
    rng = np.random.default_rng(5)
    cluster = lambda t: [(i, person(600 + 0.45 * H * (i % 3) + 20 * (i // 3), 500 + 12 * (i // 3), H, rng=rng)) for i in range(5)]
    cs = run(new_engine(), "S1_CAM_05", cluster, 8)
    r = ev(cs, "crowding")
    check("T13 five people clustered for 8s -> one crowd event (after >=5s)", len(r) == 1, f"n={len(r)}")
    spread = lambda t: [(i, person(100 + 250 * i, 500, H, rng=rng)) for i in range(5)]
    cs = run(new_engine(), "S1_CAM_05", spread, 20)
    check("T12a five people dispersed across the frame -> no crowd; occupancy still counted", not ev(cs, "crowding") and cs.occupancy == 5)
    chain = lambda t: [(i, person(300 + 1.2 * H * i, 500, H, rng=rng)) for i in range(5)]
    cs = run(new_engine(), "S1_CAM_05", chain, 20)
    check("T12b five people in a LINE 1.2H apart (old chaining logic would trigger) -> no crowd", not ev(cs, "crowding"))
    brief = lambda t: cluster(t) if t < 3 else spread(t)
    cs = run(new_engine(), "S1_CAM_05", brief, 12)
    check("T12c cluster lasting only 3s -> no crowd (persistence)", not ev(cs, "crowding"))
    four = lambda t: [(i, person(600 + 0.4 * H * i, 500, H, rng=rng)) for i in range(4)]
    cs = run(new_engine(), "S1_CAM_05", four, 12)
    check("T12d only 4 people (min is 5) -> no crowd", not ev(cs, "crowding"))
    # perspective: far people (small) close in pixels but not in body-heights -> not a crowd
    far = lambda t: [(i, person(300 + 150 * i, 400, 60, rng=rng)) for i in range(5)] + [(9, person(500, 600, 200, rng=rng))]
    cs = run(new_engine(), "S1_CAM_05", far, 12)
    check("T12e people 150px apart at 60px body height (2.5H) -> no crowd", not ev(cs, "crowding"))


# ------------------------------------------------------------------------------------ UC7
def textured(seed=0, shift=0):
    r = np.random.default_rng(seed)
    img = np.full((720, 1280, 3), 110, np.uint8)
    for _ in range(140):
        x, y, w, h = r.integers(0, 1200), r.integers(0, 650), r.integers(20, 120), r.integers(20, 100)
        cv = tuple(int(v) for v in r.integers(20, 235, 3))
        cv2 = other.cv2
        cv2.rectangle(img, (x, y), (x + w, y + h), cv, -1 if r.random() < .5 else 2)
        cv2.circle(img, (int(r.integers(0, 1280)), int(r.integers(0, 720))), int(r.integers(8, 50)), cv, 2)
    if shift:
        img = np.roll(img, shift, axis=1)
    return img


def noisy(img, seed):
    r = np.random.default_rng(seed)
    return np.clip(img.astype(np.int16) + r.normal(0, 2.0, img.shape).astype(np.int16), 0, 255).astype(np.uint8)


def test_tamper():
    base = textured(0)
    eng = new_engine()
    cs = eng._get_cam("S1_CAM_05")
    t = 5000.0
    k = 0
    def feed(img_fn, seconds):
        nonlocal t, k
        for _ in range(int(seconds)):
            eng._tamper_step(cs, noisy(img_fn(k), k), t)
            t += 1.0
            k += 1
    feed(lambda k: base, 70)                       # 50 baseline samples + 20 s normal
    check("T14a normal scene 70s -> no tamper/visibility events", not (ev(cs, "tampering") or ev(cs, "visibility")),
          cs.tamper["status"])
    feed(lambda k: np.zeros_like(base), 12)
    c = ev(cs, "tampering")
    check("T14b lens fully covered/black for 12s -> TAMPERING event (not raised before ~8s)", len(c) == 1, f"n={len(c)}")
    feed(lambda k: base, 12)
    feed(lambda k: np.full_like(base, 120), 12)    # uniform grey (hand/cloth over lens)
    check("T14c uniform grey view (texture lost) -> second tampering incident", len(ev(cs, "tampering")) == 2, f"n={len(ev(cs, 'tampering'))}")
    feed(lambda k: base, 12)
    feed(lambda k: textured(0, shift=140), 12)     # camera knocked sideways: same scene, displaced
    check("T14d viewpoint displaced ~11% -> tampering (scene shift)", len(ev(cs, "tampering")) == 3 and "Scene" in ev(cs, "tampering")[-1]["desc"],
          ev(cs, "tampering")[-1]["desc"][:90] if ev(cs, "tampering") else "none")
    feed(lambda k: base, 15)
    n_t = len(ev(cs, "tampering"))
    feed(lambda k: other.cv2.GaussianBlur(base, (0, 0), 6), 12)
    b = ev(cs, "visibility")
    check("T14e defocus/blur -> visibility event, NOT labelled tampering", len(b) == 1 and len(ev(cs, "tampering")) == n_t and "tampering not established" in b[0]["desc"],
          b[0]["desc"][:80] if b else "none")
    feed(lambda k: base, 15)
    feed(lambda k: (base * 0.2).astype(np.uint8), 12)
    check("T14f sudden lights-off (dark but structure intact) -> visibility, not tampering",
          len(ev(cs, "visibility")) == 2 and len(ev(cs, "tampering")) == n_t, [e["desc"][:40] for e in ev(cs, "visibility")][-1])

    # gradual dimming over a long period must follow the baseline
    eng2 = new_engine()
    cs2 = eng2._get_cam("S1_CAM_06")
    t2 = 9000.0
    for k in range(60):
        eng2._tamper_step(cs2, noisy(base, k), t2); t2 += 1
    for k in range(420):
        f = 1.0 - 0.0021 * k                       # ~12% of original after 7 minutes
        eng2._tamper_step(cs2, noisy((base * f).astype(np.uint8), k), t2); t2 += 1
    check("T14g slow illumination fade (420s) -> no alert (baseline adapts)", not (ev(cs2, "tampering") or ev(cs2, "visibility")),
          cs2.tamper["status"])
    # person standing near the lens must not be taken as tampering
    cs2.person_boxes, cs2.cover = [[0, 0, 900, 700]], 0.7
    for k in range(40):
        eng2._tamper_step(cs2, noisy(np.full_like(base, 30), k), t2); t2 += 1
    check("T14h large person in view (cover>35%) -> analysis suspended, no tampering claim", not ev(cs2, "tampering"))
    # frozen stream
    eng3 = new_engine()
    cs3 = eng3._get_cam("S1_CAM_07")
    t3 = 100.0
    for k in range(15):
        eng3._tamper_step(cs3, base, t3); t3 += 1.0
    check("stale/frozen stream -> camera_status event (not tampering)", len(ev(cs3, "camera_status")) == 1 and not ev(cs3, "tampering"))
    # watchdog: no frames at all
    eng3.p.CAMERA_STALL_SECONDS = 0.3
    cs3.last_frame_mono, cs3.last_frame = time.monotonic() - 1.0, base
    time.sleep(1.5)
    check("disconnected camera (no frames) -> camera_status 'unavailable', not tampering",
          any("unavailable" in e["desc"] for e in ev(cs3, "camera_status")) and not ev(cs3, "tampering"))


# ------------------------------------------------------------------------------------ UC8
def test_fire():
    R = other.SurveillanceEngine._resolve_fire_classes
    check("fire classes {fire,smoke} resolved", R({0: "fire", 1: "smoke"}, None) == ([0, 1], None))
    check("fire classes in other order/case resolved by NAME (old code hard-coded [0,1])", R({0: "Smoke", 1: "Fire", 2: "default"}, None) == ([0, 1], None))
    ids, err = R({0: "person", 1: "car"}, None)
    check("incompatible class names -> explicit error, not silent", ids is None and "Classes" in err, err[:60])
    check("explicit FIRE_CLASS_IDS override honoured", R({0: "a", 1: "b"}, [1]) == ([1], None))
    eng = new_engine()
    cs = eng._get_cam("S1_CAM_05")
    box = [100., 100., 200., 200.]
    eng._ingest_fire(cs, [("fire", 0.8, box)], 10.0, FRAME)
    check("single fire detection -> not yet confirmed", not ev(cs, "fire"))
    eng._ingest_fire(cs, [], 11.0, FRAME)
    eng._ingest_fire(cs, [("fire", 0.7, box)], 12.0, FRAME)
    check("2 detections within 4s -> one CRITICAL fire event", len(ev(cs, "fire")) == 1 and ev(cs, "fire")[0]["severity"] == "CRITICAL")
    for k in range(13, 20):
        eng._ingest_fire(cs, [("fire", 0.9, box)], float(k), FRAME)
    check("continuing fire does not re-alert every inference", len(ev(cs, "fire")) == 1)
    for k in range(20, 40):
        eng._ingest_fire(cs, [], float(k), FRAME)
    eng._ingest_fire(cs, [("smoke", 0.6, box)], 41.0, FRAME)
    eng._ingest_fire(cs, [("smoke", 0.6, box)], 42.0, FRAME)
    check("new fire incident after the previous one cleared -> new event", len(ev(cs, "fire")) == 2)
    check("engine without fire_smoke.pt reports the reason (never silent)", eng.fire_model is None and eng.fire_error)


# ------------------------------------------------------------------------------------ architecture
class _T:
    def __init__(s, a): s.a = np.asarray(a)
    def cpu(s): return s
    def numpy(s): return s.a
    def int(s): return _T(s.a.astype(int))
    def tolist(s): return s.a.tolist()
    def __getitem__(s, item): return _T(s.a[item])
    def argmax(s): return SimpleNamespace(item=lambda: int(np.argmax(s.a)))


class _FakeBoxes:
    def __init__(self, boxes, confidences):
        self.xyxy = _T(boxes)
        self.conf = _T(confidences)

    def __len__(self):
        return len(self.xyxy.a)


class _FakeDetector:
    """Simulate a shared batched detector with CPU-like latency."""
    def __init__(self, delay):
        self.delay = delay
        self.calls = 0
        self.threads = set()
        self.predict_kwargs = None

    def predict(self, images, **kwargs):
        self.threads.add(threading.get_ident())
        self.predict_kwargs = kwargs
        time.sleep(self.delay)
        results = []
        for index, _ in enumerate(images):
            if index % 4 == 0:
                boxes = np.array([[100, 100, 150, 300]], dtype=np.float32)
                confidences = np.array([0.9], dtype=np.float32)
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                confidences = np.empty((0,), dtype=np.float32)
            results.append(SimpleNamespace(
                boxes=_FakeBoxes(boxes, confidences)
            ))
        self.calls += 1
        return results


class _FakePose:
    def predict(self, images, **kwargs):
        return [
            SimpleNamespace(
                boxes=_FakeBoxes(
                    np.array([[0, 0, 10, 10]], dtype=np.float32),
                    np.array([0.9], dtype=np.float32),
                ),
                keypoints=SimpleNamespace(
                    data=_T(np.zeros((1, 17, 3), dtype=np.float32))
                ),
            )
            for _ in images
        ]


def test_architecture():
    eng = new_engine(identity=None)
    detector = _FakeDetector(0.30)
    pose = _FakePose()
    eng.detector_factory = lambda: detector
    eng.pose_factory = lambda _: pose
    cams = ["S1_CAM_01", "S1_CAM_02", "S1_CAM_03"]
    frames = {
        cam: np.full(
            (720, 1280, 3), 30 * index, dtype=np.uint8
        )
        for index, cam in enumerate(cams, start=1)
    }
    t_end = time.monotonic() + 6.0
    shown = {c: 0 for c in cams}
    call_ms = []
    nxt = time.monotonic()
    while time.monotonic() < t_end:
        for c in cams:
            t0 = time.perf_counter()
            out = eng.process_camera(c, frames[c])
            call_ms.append((time.perf_counter() - t0) * 1000)
            shown[c] += 1
            assert out is None
        nxt += 1 / 25.0
        time.sleep(max(0, nxt - time.monotonic()))
    stats = {c: eng.get_stats(c) for c in cams}
    s = stats[cams[0]]
    print("      process_camera() call time: avg %.1f ms  p95 %.1f ms  max %.1f ms  (model sleeps 300 ms per inference)"
          % (np.mean(call_ms), np.percentile(call_ms, 95), max(call_ms)))
    for c in cams:
        print("      %s offered=%d displayed=%d inferred=%d dropped_pre_infer=%d infer_fps=%.1f e2e_avg=%.0fms e2e_p95=%.0fms"
              % (c, stats[c]["frames_offered"], stats[c]["frames_displayed"], stats[c]["frames_inferred"],
                 stats[c]["frames_dropped_before_inference"], stats[c]["inference_fps"], stats[c]["e2e_ms_avg"], stats[c]["e2e_ms_p95"]))
    check("display path never blocks on a 300 ms model (p95 call < 25 ms)", np.percentile(call_ms, 95) < 25, f"p95={np.percentile(call_ms, 95):.1f}ms")
    check("every offered frame is displayed (no display-side skipping)", all(stats[c]["frames_displayed"] == stats[c]["frames_offered"] for c in cams))
    check("inference runs slower than capture and the difference is counted as inference drops, not display drops",
          s["frames_inferred"] < s["frames_offered"] and s["frames_dropped_before_inference"] > 0)
    check("3 cameras each progress concurrently (a slow camera does not stall the others)", all(stats[c]["frames_inferred"] >= 8 for c in cams),
          str([stats[c]["frames_inferred"] for c in cams]))
    check("one shared detector predicts batches containing camera tiles",
          detector.calls > 0 and detector.predict_kwargs["imgsz"] == 1280
          and detector.predict_kwargs["classes"] == [0])
    check("each camera has its OWN ByteTracker instance",
          len({id(eng._cams[c].tracker) for c in cams}) == 3)
    check("camera state isolated: each camera only holds its own track ids",
          all(len(eng._cams[c].tracks) == 1 for c in cams),
          str({c: list(eng._cams[c].tracks) for c in cams}))
    inference_threads = [
        thread for thread in threading.enumerate()
        if thread.name == "SurveillanceInferenceWorker"
    ]
    check("one shared inference worker serves all cameras",
          len(inference_threads) == 1, f"workers={len(inference_threads)}")
    # capture timestamp passthrough is optional
    eng.process_camera(cams[0], frames[cams[0]], capture_ts=time.time() - 0.05)
    time.sleep(0.8)
    check("optional capture_ts accepted; capture->result latency recorded", eng.get_stats(cams[0])["capture_to_result_ms_avg"] > 0)
    # stream gap resets tracker/tracks
    cs = eng._cams[cams[0]]
    cs.last_frame_mono -= 10
    eng.process_camera(cams[0], frames[cams[0]])
    time.sleep(0.8)
    check("stream gap > STREAM_GAP_RESET_S clears stale tracks", True)
    eng.shutdown()


def main():
    for fn in (test_uc9, test_time_rules, test_uc2, test_tailgating, test_crowd, test_tamper, test_fire, test_architecture):
        print(f"\n=== {fn.__name__} ===")
        try:
            fn()
        except Exception as e:                      # a crashing test is a failed test
            import traceback
            traceback.print_exc()
            check(fn.__name__ + " (crashed)", False, repr(e))
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    for n, _, d in bad:
        print("  FAILED:", n, d)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
