# code-review
"""
========================================================
  Agentic AI Posture Monitoring — Phase 2 (XY Joints)
  Fast XY coordinate detection | Re-ID | Multi-User
=======================================================

Detection method — XY joint offsets (no angle math):
  Shoulder Y drop     → spine slouch
  Shoulder Y diff     → shoulder tilt
  Ear X vs Shoulder X → forward head
  Ear Y vs Shoulder Y → neck droop
  Shoulder X vs Hip X → forward lean

All checks are simple subtractions — microsecond speed.
Drawn live: colored joint dots + X/Y axis lines on body.

Camera: sit so HEAD + SHOULDERS + HIPS are visible.
Sit straight during 6-second calibration.

Dependencies:
    python -m pip install opencv-python mediapipe ultralytics
    python -m pip install deep-sort-realtime pyttsx3
    python -m pip install google-genai pandas matplotlib
    python -m pip install python-dotenv insightface onnxruntime scipy
"""

import cv2
import mediapipe as mp
import math
import time
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque, Counter
from tkinter import Tk, messagebox
from concurrent.futures import ThreadPoolExecutor

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pyttsx3
import google.genai as genai
from dotenv import load_dotenv
import os

try:
    from scipy.spatial.distance import cosine as cos_dist
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from insightface.app import FaceAnalysis
    USE_INSIGHTFACE = True
except ImportError:
    USE_INSIGHTFACE = False

# -------------------------------------------------------
# Config
# -------------------------------------------------------
load_dotenv()
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "AIzaSyA3mQSUUi9R9VEanIcuTdxOT6j4hDzyty0")

YOLO_EVERY_N        = 4       # run YOLO every N frames
INFER_WIDTH         = 480     # resize width for YOLO inference
POSE_WORKERS        = 3
CALIBRATION_SECS    = 6
LOG_INTERVAL_SECS   = 4
VOICE_COOLDOWN      = 12
BAD_STREAK_TRIGGER  = 6
SMOOTH_N            = 10      # EWM smoothing window
MAX_USERS           = 3
YOLO_CONF           = 0.5
IOU_LIMIT           = 0.45
VIS_THRESH          = 0.45
MIN_FACE_PX         = 40
REID_THRESH         = 0.55
REID_UPDATE_EVERY   = 30      # frames between embedding refresh

# -------------------------------------------------------
# XY Posture Thresholds
# All values are in NORMALISED units (0.0–1.0 of frame size)
# except shoulder_tilt which is pixel difference in Y.
# -------------------------------------------------------
# How much each metric must DEVIATE from calibrated baseline
# before it triggers a warning / bad flag.

# Shoulder Y drop (slouch): baseline_shoulder_y - current_shoulder_y
# Positive = shoulders rose (good), Negative = shoulders dropped (slouch)
SHOULDER_Y_WARN     = 0.018   # ~1.8% of frame height drop
SHOULDER_Y_BAD      = 0.035   # ~3.5% of frame height drop

# Shoulder tilt: abs(left_shoulder_y - right_shoulder_y)
# Normalised — one shoulder higher than other
SHOULDER_TILT_WARN  = 0.020
SHOULDER_TILT_BAD   = 0.040

# Forward head: ear_x - shoulder_x (positive = ear ahead of shoulder)
# We use the side with better visibility
HEAD_FWD_WARN       = 0.020
HEAD_FWD_BAD        = 0.040

# Neck droop: shoulder_y - ear_y (positive = ear above shoulder, good)
# Shrinking gap = neck drooping forward
NECK_DROP_WARN      = 0.015
NECK_DROP_BAD       = 0.030

# Forward lean: shoulder_mid_x - hip_mid_x deviation
# (slouching forward in chair)
LEAN_FWD_WARN       = 0.020
LEAN_FWD_BAD        = 0.040

USER_COLORS = {
    0: (50,  210,  50),
    1: (50,  150, 255),
    2: (200,  50, 200),
}

# Joint colors for skeleton overlay (BGR)
JC_EAR      = (200,  80, 200)   # purple
JC_SHOULDER = (50,  200,  50)   # green
JC_HIP      = (50,  180, 255)   # orange

# -------------------------------------------------------
# Gemini
# -------------------------------------------------------
client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------------------------------------
# TTS
# -------------------------------------------------------
_tts_lock  = threading.Lock()
_tts_queue: list[str] = []

def _tts_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    while True:
        if _tts_queue:
            with _tts_lock:
                msg = _tts_queue.pop(0)
            engine.say(msg); engine.runAndWait()
        else:
            time.sleep(0.1)

threading.Thread(target=_tts_worker, daemon=True).start()

def speak(text: str):
    with _tts_lock:
        if len(_tts_queue) < 2:
            _tts_queue.append(text)

# -------------------------------------------------------
# Face Re-ID Engine
# -------------------------------------------------------
class ReIDEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = False
        if USE_INSIGHTFACE:
            try:
                self._app = FaceAnalysis(
                    name="buffalo_sc",
                    providers=["CPUExecutionProvider"]
                )
                self._app.prepare(ctx_id=0, det_size=(320, 320))
                self._enabled = True
                print("ReID: InsightFace loaded.")
            except Exception as e:
                print(f"ReID: InsightFace failed ({e}). Using position-based fallback.")
        else:
            print("ReID: insightface not installed. Using position-based fallback.")

    def get_embedding(self, face_bgr) -> np.ndarray | None:
        if not self._enabled or face_bgr is None:
            return None
        h, w = face_bgr.shape[:2]
        if h < MIN_FACE_PX or w < MIN_FACE_PX:
            return None
        try:
            with self._lock:
                rgb   = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                faces = self._app.get(rgb)
                if not faces:
                    return None
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                emb  = face.embedding
                return emb / np.linalg.norm(emb)
        except Exception:
            return None

    def similarity(self, e1, e2) -> float:
        if e1 is None or e2 is None:
            return 0.0
        try:
            if SCIPY_OK:
                return float(1.0 - cos_dist(e1, e2))
            dot = np.dot(e1, e2)
            return float(dot / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        except Exception:
            return 0.0

reid = ReIDEngine()

# -------------------------------------------------------
# User Registry (persistent across track ID changes)
# -------------------------------------------------------
user_registry:  dict[int, dict] = {}   # user_id -> state
track_to_user:  dict[int, int]  = {}   # deepsort track_id -> user_id
_next_uid = 1

def _new_state(uid: int) -> dict:
    slot = (uid - 1) % MAX_USERS
    return {
        "user_id":         uid,
        "label":           f"User {uid}",
        "color":           USER_COLORS[slot],
        "face_embedding":  None,
        "embed_count":     0,

        # Calibration
        "calibrated":      False,
        "calib_start":     time.time(),
        "calib_buf":       [],          # list of metric dicts

        # Baseline (normalised coords)
        "baseline":        None,        # dict of metric baselines

        # Smoothing (EWM — exponentially weighted)
        "smooth":          {},          # metric_name -> smoothed value

        # Runtime
        "posture":         "Calibrating",
        "score":           100,
        "issues":          [],
        "bad_streak":      0,
        "last_voice_time": 0,
        "session_issues":  [],

        # Logging
        "log":             [],
        "last_log_time":   time.time(),
        "bbox":            None,
    }

def resolve_user(track_id: int, face_crop, frame) -> dict:
    """Map a DeepSORT track_id to a persistent user, using face Re-ID."""
    global _next_uid

    if track_id in track_to_user:
        return user_registry[track_to_user[track_id]]

    # Try face match
    embedding = reid.get_embedding(face_crop)
    best_uid, best_sim = None, 0.0
    for uid, state in user_registry.items():
        sim = reid.similarity(embedding, state.get("face_embedding"))
        if sim > best_sim:
            best_sim, best_uid = sim, uid

    if best_sim >= REID_THRESH and best_uid is not None:
        track_to_user[track_id] = best_uid
        state = user_registry[best_uid]
        if embedding is not None:
            state["face_embedding"] = embedding
        print(f"Re-ID: track {track_id} → {state['label']} (sim={best_sim:.2f})")
        speak(f"Welcome back, {state['label']}.")
        return state

    # New user
    uid = _next_uid; _next_uid += 1
    state = _new_state(uid)
    state["face_embedding"] = embedding
    user_registry[uid]      = state
    track_to_user[track_id] = uid
    print(f"New user: track {track_id} → {state['label']}")
    return state

# -------------------------------------------------------
# MediaPipe — thread-local
# -------------------------------------------------------
_tl = threading.local()

def get_pose():
    if not hasattr(_tl, "pose"):
        _tl.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    return _tl.pose

mp_fd = mp.solutions.face_detection
_face_det_lock = threading.Lock()
_face_det = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)

pose_pool = ThreadPoolExecutor(max_workers=POSE_WORKERS)

# -------------------------------------------------------
# YOLO + DeepSORT
# -------------------------------------------------------
yolo  = YOLO("yolov8n.pt")
dsort = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.4)

# -------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------
def box_iou(b1, b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1])
    ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    return inter/((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter)

def filter_boxes(boxes):
    kept=[]
    for b in boxes:
        dom=False
        for i,k in enumerate(kept):
            if box_iou(b["xyxy"],k["xyxy"])>IOU_LIMIT:
                if (b["xyxy"][2]-b["xyxy"][0])*(b["xyxy"][3]-b["xyxy"][1])>(k["xyxy"][2]-k["xyxy"][0])*(k["xyxy"][3]-k["xyxy"][1]):
                    kept[i]=b
                dom=True; break
        if not dom: kept.append(b)
    return kept

# -------------------------------------------------------
# XY Joint Extraction — returns normalised (0-1) coords
# -------------------------------------------------------
def extract_joints(crop_bgr, ox, oy, fw, fh):
    """
    Run MediaPipe on a person crop.
    Returns dict of normalised full-frame joint positions,
    or None if pose not found.
    ox,oy = crop top-left in full frame pixels.
    fw,fh = full frame width, height.
    """
    h, w = crop_bgr.shape[:2]
    result = get_pose().process(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks.landmark

    def vis(idx): return lm[idx].visibility >= VIS_THRESH

    def norm(idx):
        # Normalised coords in full-frame space
        px = lm[idx].x * w + ox
        py = lm[idx].y * h + oy
        return px / fw, py / fh

    # Gather joints with visibility flags
    joints = {}
    for name, idx in [("l_ear",7),("r_ear",8),
                       ("l_shoulder",11),("r_shoulder",12),
                       ("l_hip",23),("r_hip",24)]:
        if vis(idx):
            joints[name] = norm(idx)
        else:
            joints[name] = None

    return joints

# -------------------------------------------------------
# Compute posture metrics from joints
# Returns dict: metric_name -> float value
# -------------------------------------------------------
def compute_metrics(joints: dict) -> dict | None:
    ls = joints.get("l_shoulder")
    rs = joints.get("r_shoulder")
    lh = joints.get("l_hip")
    rh = joints.get("r_hip")
    le = joints.get("l_ear")
    re = joints.get("r_ear")

    if ls is None or rs is None:
        return None          # can't do anything without shoulders

    sh_mid_x = (ls[0] + rs[0]) / 2
    sh_mid_y = (ls[1] + rs[1]) / 2

    # Shoulder Y — lower number = higher up in frame (Y=0 is top)
    shoulder_y = sh_mid_y

    # Shoulder tilt — abs diff in Y between left and right shoulder
    shoulder_tilt = abs(ls[1] - rs[1])

    # Hip mid
    if lh and rh:
        hip_mid_x = (lh[0] + rh[0]) / 2
    elif lh:
        hip_mid_x = lh[0]
    elif rh:
        hip_mid_x = rh[0]
    else:
        hip_mid_x = sh_mid_x    # fallback

    # Forward lean: how far shoulder is ahead (in X) of hip
    lean_fwd = sh_mid_x - hip_mid_x

    # Head forward: ear X vs shoulder X (use better-visibility ear)
    head_fwd  = None
    neck_drop = None
    if le:
        head_fwd  = le[0] - ls[0]          # positive = ear ahead of shoulder
        neck_drop = ls[1] - le[1]           # positive = ear above shoulder (good)
    elif re:
        head_fwd  = re[0] - rs[0]
        neck_drop = rs[1] - re[1]

    return {
        "shoulder_y":    shoulder_y,
        "shoulder_tilt": shoulder_tilt,
        "lean_fwd":      lean_fwd,
        "head_fwd":      head_fwd,        # None if no ear visible
        "neck_drop":     neck_drop,       # None if no ear visible
    }

# -------------------------------------------------------
# EWM smoother per metric
# -------------------------------------------------------
ALPHA = 0.35   # higher = more responsive, lower = smoother

def smooth_update(state: dict, metrics: dict):
    """Update exponentially weighted moving averages for each metric."""
    s = state["smooth"]
    for k, v in metrics.items():
        if v is None:
            continue
        if k not in s:
            s[k] = v
        else:
            s[k] = ALPHA * v + (1 - ALPHA) * s[k]

# -------------------------------------------------------
# Posture Classification using XY deviations
# -------------------------------------------------------
def classify(state: dict) -> tuple[int, str, list[str]]:
    s  = state["smooth"]
    bl = state["baseline"]
    issues    = []
    penalties = 0

    # 1. Shoulder Y drop (spine slouch)
    if "shoulder_y" in s and "shoulder_y" in bl:
        dev = s["shoulder_y"] - bl["shoulder_y"]   # positive = dropped (slouch)
        if dev > SHOULDER_Y_BAD:
            issues.append("spine"); penalties += 35
        elif dev > SHOULDER_Y_WARN:
            issues.append("spine_mild"); penalties += 15

    # 2. Shoulder tilt
    if "shoulder_tilt" in s and "shoulder_tilt" in bl:
        dev = s["shoulder_tilt"] - bl["shoulder_tilt"]
        if dev > SHOULDER_TILT_BAD:
            issues.append("shoulder"); penalties += 20
        elif dev > SHOULDER_TILT_WARN:
            issues.append("shoulder_mild"); penalties += 8

    # 3. Forward head (ear X ahead of shoulder)
    if "head_fwd" in s and "head_fwd" in bl:
        dev = s["head_fwd"] - bl["head_fwd"]        # positive = moved forward
        if dev > HEAD_FWD_BAD:
            issues.append("neck"); penalties += 30
        elif dev > HEAD_FWD_WARN:
            issues.append("neck_mild"); penalties += 12

    # 4. Neck droop (ear-shoulder Y gap shrinking)
    if "neck_drop" in s and "neck_drop" in bl:
        dev = bl["neck_drop"] - s["neck_drop"]      # positive = gap shrank
        if dev > NECK_DROP_BAD:
            if "neck" not in issues: issues.append("neck"); penalties += 15
        elif dev > NECK_DROP_WARN:
            if "neck_mild" not in issues: issues.append("neck_mild"); penalties += 6

    # 5. Forward lean (shoulder drifted forward of hip)
    if "lean_fwd" in s and "lean_fwd" in bl:
        dev = s["lean_fwd"] - bl["lean_fwd"]
        if abs(dev) > LEAN_FWD_BAD:
            issues.append("lean"); penalties += 20
        elif abs(dev) > LEAN_FWD_WARN:
            issues.append("lean_mild"); penalties += 8

    score = max(0, 100 - penalties)
    label = "Good Posture" if score >= 80 else "Fair Posture" if score >= 55 else "Bad Posture"
    return score, label, issues

# -------------------------------------------------------
# Joint skeleton overlay on full frame
# -------------------------------------------------------
def draw_joints(frame, joints: dict, fw, fh):
    """Draw coloured joint dots and axis lines directly on the frame."""
    def px(norm_pt):
        return int(norm_pt[0] * fw), int(norm_pt[1] * fh)

    ls = joints.get("l_shoulder"); rs = joints.get("r_shoulder")
    lh = joints.get("l_hip");      rh = joints.get("r_hip")
    le = joints.get("l_ear");      re = joints.get("r_ear")

    # Shoulder line (X axis reference)
    if ls and rs:
        cv2.line(frame, px(ls), px(rs), JC_SHOULDER, 2)
        cv2.circle(frame, px(ls), 6, JC_SHOULDER, -1)
        cv2.circle(frame, px(rs), 6, JC_SHOULDER, -1)
        # X-axis dashes left and right of shoulders
        lx, ly = px(ls); rx, ry = px(rs)
        cv2.line(frame, (lx-30, ly), (lx, ly), JC_SHOULDER, 1)
        cv2.line(frame, (rx, ry), (rx+30, ry), JC_SHOULDER, 1)
        # Label
        mid_x = (lx+rx)//2
        cv2.putText(frame, "SH-X", (mid_x-18, ly-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, JC_SHOULDER, 1, cv2.LINE_AA)

    # Hip line
    if lh and rh:
        cv2.line(frame, px(lh), px(rh), JC_HIP, 2)
        cv2.circle(frame, px(lh), 6, JC_HIP, -1)
        cv2.circle(frame, px(rh), 6, JC_HIP, -1)

    # Spine Y line (shoulder mid → hip mid)
    if ls and rs and lh and rh:
        sh_mid = ((px(ls)[0]+px(rs)[0])//2, (px(ls)[1]+px(rs)[1])//2)
        hp_mid = ((px(lh)[0]+px(rh)[0])//2, (px(lh)[1]+px(rh)[1])//2)
        cv2.line(frame, sh_mid, hp_mid, (100, 180, 255), 1)
        cv2.putText(frame, "SP-Y", (sh_mid[0]+4, (sh_mid[1]+hp_mid[1])//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,180,255), 1, cv2.LINE_AA)

    # Ear dots + neck X/Y lines
    for ear, shoulder, label in [(le, ls, "L"), (re, rs, "R")]:
        if ear and shoulder:
            ep = px(ear); sp = px(shoulder)
            cv2.circle(frame, ep, 5, JC_EAR, -1)
            # Vertical line ear→shoulder (Y axis: neck height)
            cv2.line(frame, ep, (ep[0], sp[1]), JC_EAR, 1)
            # Horizontal line (X axis: head forward)
            cv2.line(frame, (ep[0], sp[1]), sp, JC_EAR, 1)
            cv2.putText(frame, f"NK-{label}", (ep[0]+4, ep[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, JC_EAR, 1, cv2.LINE_AA)

# -------------------------------------------------------
# Agentic voice messages
# -------------------------------------------------------
ISSUE_VOICE = {
    "spine":         "your shoulders have dropped. Lift your chest and sit tall.",
    "spine_mild":    "your back is starting to round. Try to sit upright.",
    "neck":          "your head is pushed too far forward. Pull your chin back.",
    "neck_mild":     "your neck is tilting forward. Look straight ahead.",
    "shoulder":      "your shoulders are uneven. Level them out.",
    "shoulder_mild": "your shoulders are slightly tilted. Try to balance them.",
    "lean":          "you are leaning too far forward. Sit back in your chair.",
    "lean_mild":     "you are slightly leaning forward. Sit back a little.",
}

def voice_msg(label_name, issues):
    for p in ["spine","neck","lean","shoulder","spine_mild","neck_mild","lean_mild","shoulder_mild"]:
        if p in issues:
            return f"{label_name}, {ISSUE_VOICE[p]}"
    return f"{label_name}, please correct your posture."

# -------------------------------------------------------
# Draw bounding box overlay
# -------------------------------------------------------
def draw_overlay(frame, x1, y1, x2, y2, state):
    score   = state["score"]
    posture = state["posture"]
    label   = state["label"]
    issues  = state["issues"]
    color   = (50,210,50) if score>=80 else (0,165,255) if score>=55 else (0,0,220)

    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

    bar = f"{label}  {posture}  [{score}/100]"
    (tw,th),_ = cv2.getTextSize(bar, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)
    cv2.rectangle(frame,(x1,y1-th-14),(x1+tw+10,y1),color,-1)
    cv2.putText(frame, bar, (x1+5,y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255,255,255), 1, cv2.LINE_AA)

    tx = x1
    for iss in issues:
        clean = iss.replace("_mild","?").upper()
        tag   = f" {clean} "
        (ttw,tth),_ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tc = (0,120,255) if "mild" in iss else (0,0,200)
        cv2.rectangle(frame,(tx,y1+4),(tx+ttw+2,y1+tth+10),tc,-1)
        cv2.putText(frame, tag, (tx+1,y1+tth+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)
        tx += ttw+6

    bw = x2-x1
    cv2.rectangle(frame,(x1,y2+3),(x2,y2+9),(60,60,60),-1)
    cv2.rectangle(frame,(x1,y2+3),(x1+int(bw*score/100),y2+9),color,-1)

    if not state["calibrated"]:
        elapsed  = time.time()-state["calib_start"]
        progress = min(elapsed/CALIBRATION_SECS, 1.0)
        cv2.rectangle(frame,(x1,y2+14),(x2,y2+20),(40,40,40),-1)
        cv2.rectangle(frame,(x1,y2+14),(x1+int(bw*progress),y2+20),(0,255,255),-1)
        cv2.putText(frame,"Calibrating — sit straight!",
                    (x1,y2+34),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,255),1,cv2.LINE_AA)

# -------------------------------------------------------
# Face crop helper
# -------------------------------------------------------
def get_face_crop(frame, x1, y1, x2, y2):
    person = frame[y1:y2, x1:x2]
    h, w   = person.shape[:2]
    with _face_det_lock:
        res = _face_det.process(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
    if res.detections:
        d  = res.detections[0]
        bb = d.location_data.relative_bounding_box
        fx1 = max(0, int(bb.xmin*w))
        fy1 = max(0, int(bb.ymin*h))
        fx2 = min(w, int((bb.xmin+bb.width)*w))
        fy2 = min(h, int((bb.ymin+bb.height)*h))
        crop = person[fy1:fy2, fx1:fx2]
        return crop if crop.size>0 else None
    # Fallback: top 35% of bbox
    fh = int(h*0.35)
    return person[:fh, :] if fh>MIN_FACE_PX else None

# -------------------------------------------------------
# Agentic session analysis via Gemini
# -------------------------------------------------------
def agentic_analysis(label, log, session_issues):
    total  = len(log)
    good_c = sum(1 for r in log if r[0]=="Good Posture")
    fair_c = sum(1 for r in log if r[0]=="Fair Posture")
    bad_c  = sum(1 for r in log if r[0]=="Bad Posture")
    avg_sc = int(np.mean([r[1] for r in log])) if log else 0
    top    = ", ".join(f"{k}({v}x)" for k,v in Counter(session_issues).most_common(3)) or "none"
    prompt = (
        "You are a posture correction agent. Your job:\n"
        "1. OBSERVE the data  2. DIAGNOSE root causes  "
        "3. PRESCRIBE exact corrective actions  4. PRIORITIZE the #1 fix\n\n"
        "Be direct, clinical, specific. No general wellness advice. "
        "Short numbered steps. Each step must name the body part and exact movement.\n\n"
        f"Session — {label}: Total={total} Good={good_c} Fair={fair_c} Bad={bad_c} "
        f"AvgScore={avg_sc}/100 TopIssues={top}\n\n"
        f"Give 4 specific corrective instructions for {label}."
    )
    try:
        r = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[{"role":"user","parts":[{"text":prompt}]}]
        )
        return r.text.strip()
    except Exception as e:
        return f"Agent analysis unavailable: {e}"

# -------------------------------------------------------
# Session Report
# -------------------------------------------------------
def session_report():
    print("\nGenerating reports...")
    for uid, state in user_registry.items():
        if not state["log"]: continue
        label = state["label"]
        df    = pd.DataFrame(state["log"], columns=["Posture","Score","Time"])
        df.to_csv(f"posture_log_{label.replace(' ','_')}.csv", index=False)

        good_c = sum(df["Posture"]=="Good Posture")
        fair_c = sum(df["Posture"]=="Fair Posture")
        bad_c  = sum(df["Posture"]=="Bad Posture")
        avg_sc = int(df["Score"].astype(int).mean())

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10,3))
        sc = df["Score"].astype(int).tolist()
        tm = df["Time"].tolist()
        ax.fill_between(range(len(sc)), sc, alpha=0.2, color="lime")
        ax.plot(range(len(sc)), sc, color="lime", linewidth=1.8, marker="o", markersize=3)
        ax.axhline(80, color="#00c800", linestyle="--", linewidth=0.9, label="Good>=80")
        ax.axhline(55, color="orange",  linestyle="--", linewidth=0.9, label="Fair>=55")
        step = max(1, len(tm)//10)
        ax.set_xticks(range(0, len(tm), step))
        ax.set_xticklabels(tm[::step], rotation=45, fontsize=7)
        ax.set_ylim(0,105)
        ax.set_title(f"Posture Score — {label}", color="white")
        ax.set_ylabel("Score (0-100)"); ax.legend(fontsize=8)
        plt.tight_layout()
        png = f"posture_graph_{label.replace(' ','_')}.png"
        plt.savefig(png, dpi=150); plt.close()
        print(f"  Saved {png}")

        analysis = agentic_analysis(label, state["log"], state["session_issues"])
        root = Tk(); root.withdraw()
        messagebox.showinfo(
            f"Agent Report — {label}",
            f"POSTURE AGENT REPORT\n{'='*36}\n"
            f"User: {label}  AvgScore: {avg_sc}/100\n"
            f"Good:{good_c}  Fair:{fair_c}  Bad:{bad_c}\n"
            f"{'='*36}\n\nAGENT DIAGNOSIS:\n{analysis}"
        )
        root.destroy()

# -------------------------------------------------------
# Main Loop
# -------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale   = INFER_WIDTH / fw
infer_h = int(fh * scale)

print(f"Camera: {fw}x{fh}  Inference: {INFER_WIDTH}x{infer_h}")
print("Head + shoulders + hips must be visible. ESC to quit.\n")

frame_count   = 0
active_tracks = {}    # track_id -> (x1,y1,x2,y2)
pose_futures  = {}    # track_id -> Future[dict|None]
prev_time     = time.time()

# Store latest joints per user for drawing every frame
latest_joints: dict[int, dict] = {}   # user_id -> joints dict

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    now = time.time()
    fps = 1.0 / max(now - prev_time, 0.001)
    prev_time = now

    # ---- YOLO + DeepSORT (every N frames) ----
    if frame_count % YOLO_EVERY_N == 0:
        small    = cv2.resize(frame, (INFER_WIDTH, infer_h))
        yolo_out = yolo(small, classes=[0], conf=YOLO_CONF, verbose=False)[0]

        raw = []
        for b in yolo_out.boxes:
            sx1,sy1,sx2,sy2 = map(float, b.xyxy[0])
            raw.append({"xyxy":[int(sx1/scale),int(sy1/scale),
                                  int(sx2/scale),int(sy2/scale)],
                         "conf":float(b.conf[0])})
        raw = filter_boxes(raw)

        ds_in = [([d["xyxy"][0],d["xyxy"][1],
                   d["xyxy"][2]-d["xyxy"][0],
                   d["xyxy"][3]-d["xyxy"][1]], d["conf"], 0) for d in raw]

        tracks     = dsort.update_tracks(ds_in, frame=frame)
        new_active = {}

        for t in tracks:
            if not t.is_confirmed(): continue
            tid  = t.track_id
            ltrb = t.to_ltrb()
            x1=max(0,int(ltrb[0])); y1=max(0,int(ltrb[1]))
            x2=min(fw-1,int(ltrb[2])); y2=min(fh-1,int(ltrb[3]))
            if x2-x1<50 or y2-y1<80: continue
            if len(new_active)>=MAX_USERS: break
            new_active[tid]=(x1,y1,x2,y2)

            # Re-ID on first sight of this track
            if tid not in track_to_user:
                fc    = get_face_crop(frame, x1, y1, x2, y2)
                state = resolve_user(tid, fc, frame)
            else:
                state = user_registry[track_to_user[tid]]

            state["bbox"] = (x1,y1,x2,y2)

            # Periodic embedding refresh
            state["embed_count"] += 1
            if (state["embed_count"] % REID_UPDATE_EVERY == 0
                    and state.get("face_embedding") is not None):
                fc = get_face_crop(frame, x1, y1, x2, y2)
                new_emb = reid.get_embedding(fc)
                if new_emb is not None:
                    old = state["face_embedding"]
                    merged = 0.7*old + 0.3*new_emb
                    state["face_embedding"] = merged / np.linalg.norm(merged)

            # Submit pose job — pass full frame crop
            crop = frame[y1:y2, x1:x2].copy()
            pose_futures[tid] = pose_pool.submit(extract_joints, crop, x1, y1, fw, fh)

        active_tracks = new_active

    # ---- Collect pose results ----
    for tid in list(pose_futures.keys()):
        fut = pose_futures[tid]
        if not fut.done(): continue
        joints = fut.result()
        del pose_futures[tid]

        if joints is None or tid not in track_to_user: continue
        uid   = track_to_user[tid]
        state = user_registry[uid]

        # Compute metrics
        metrics = compute_metrics(joints)
        if metrics is None: continue

        # Update EWM smoother
        smooth_update(state, metrics)

        # Store joints for drawing
        latest_joints[uid] = joints

        # Calibration
        if not state["calibrated"]:
            state["calib_buf"].append(dict(state["smooth"]))
            if time.time()-state["calib_start"] >= CALIBRATION_SECS:
                # Average all calibration snapshots as baseline
                baseline = {}
                keys = set(k for d in state["calib_buf"] for k in d)
                for k in keys:
                    vals = [d[k] for d in state["calib_buf"] if k in d]
                    if vals:
                        baseline[k] = float(np.mean(vals))
                state["baseline"]   = baseline
                state["calibrated"] = True
                print(f"{state['label']} calibrated. Baseline: { {k:f'{v:.4f}' for k,v in baseline.items()} }")
                speak(f"{state['label']} ready. Monitoring your posture.")
            continue

        # Classify
        score, posture, issues = classify(state)
        state["score"]   = score
        state["posture"] = posture
        state["issues"]  = issues
        state["session_issues"].extend(issues)
        state["bad_streak"] = state["bad_streak"]+1 if posture!="Good Posture" else 0

        # Voice
        t = time.time()
        if (posture=="Bad Posture"
                and state["bad_streak"]>BAD_STREAK_TRIGGER
                and t-state["last_voice_time"]>VOICE_COOLDOWN):
            speak(voice_msg(state["label"], issues))
            state["last_voice_time"]=t
        elif (posture=="Fair Posture"
                and state["bad_streak"]>BAD_STREAK_TRIGGER*2
                and t-state["last_voice_time"]>VOICE_COOLDOWN*1.5):
            speak(voice_msg(state["label"], issues))
            state["last_voice_time"]=t

        # Log
        if t-state["last_log_time"]>=LOG_INTERVAL_SECS:
            state["log"].append((posture,score,time.strftime("%H:%M:%S")))
            state["last_log_time"]=t

    # ---- Draw every frame ----
    for tid,(x1,y1,x2,y2) in active_tracks.items():
        if tid not in track_to_user: continue
        uid   = track_to_user[tid]
        state = user_registry[uid]

        # Draw joint skeleton
        if uid in latest_joints:
            draw_joints(frame, latest_joints[uid], fw, fh)

        # Draw bbox overlay
        draw_overlay(frame, x1, y1, x2, y2, state)

    # HUD
    reid_tag = "Re-ID:ON" if reid._enabled else "Re-ID:OFF"
    cv2.putText(frame,
                f"FPS:{fps:.1f}  Users:{len(active_tracks)}  {reid_tag}  ESC=quit",
                (10, fh-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (160,160,160), 1, cv2.LINE_AA)

    cv2.imshow("Agentic Posture Monitor — Phase 2", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pose_pool.shutdown(wait=False)
_face_det.close()

session_report()
print("\nSession complete.")
