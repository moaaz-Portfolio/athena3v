"""Local web app: live view + guided recording + one-click training. Only needs numpy; torch for train/live."""
import json
import os
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import data
from .live import Predictor, motion_energy
from .receiver import CsiReceiver
from .fusion import MotionPresenceFusion

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "activity_model.pt")
PRETRAINED = os.path.join(ROOT, "pretrained", "activity_model_esp_fi_har.pt")
FS, WIN, HOP = 50, 160, 25


class State:
    def __init__(self, port):
        self.rx = CsiReceiver(port)
        self.fusion = MotionPresenceFusion(os.path.join(DATA_DIR, "empty_room_calibration.json"))
        self.fusion_state = {"motion": "unknown", "presence": "unknown", "ready": False, "calibrated": False}
        self.predictor = None
        self.model_error = ""
        self.pred = None
        self.energy_hist = deque(maxlen=120)
        self.fall_events = deque(maxlen=30)
        self._last_event_alert = 0.0
        self._previous_fall_alert = False
        # recording
        self.rec = {"status": "idle", "msg": "", "label": "", "left": 0, "frames": 0}
        self._rec_stop = threading.Event()
        self.countdown_seconds = 3
        # training
        self.train = {"status": "idle", "epoch": 0, "epochs": 0, "log": [], "metrics": None}
        self._train_stop = threading.Event()
        self.load_model()

    # ------------------------------------------------------------- model
    def load_model(self):
        self.predictor, self.model_error = None, ""
        if not os.path.exists(MODEL_PATH):
            return
        try:
            self.predictor = Predictor(MODEL_PATH)
        except ModuleNotFoundError:
            self.model_error = "torch is not installed (pip install torch)"
        except Exception as e:
            self.model_error = f"could not load model: {e}"

    def loop(self):
        while True:
            time.sleep(0.5)
            ts, amp = self.rx.recent(5.0)
            self.energy_hist.append(round(motion_energy(amp), 3))
            if self.rx.connected() and len(amp) >= 20:
                self.fusion_state = self.fusion.update(ts, amp)
            elif not self.rx.connected():
                self.fusion_state = {"motion": "unknown", "presence": "unknown",
                                     "motion_score": None, "presence_score": None,
                                     "ready": False, "calibrated": self.fusion.empty_mean is not None}
            if self.predictor and self.rx.connected():
                ts, amp = self.rx.recent(self.predictor.seconds_needed + 1.0)
                try:
                    self.pred = self.predictor.predict(ts, amp)
                    alert_now = bool(self.pred and self.pred.get("fall_alert"))
                    if alert_now and not self._previous_fall_alert:
                        stamp = time.time()
                        self._last_event_alert = stamp
                        self.fall_events.appendleft({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "confidence": round(self.pred.get("fall_probability", 0) * 100, 1),
                            "status": "possible_fall_pattern",
                        })
                    self._previous_fall_alert = alert_now
                except Exception as e:
                    self.model_error = str(e)
            elif not self.rx.connected():
                self.pred = None
                self._previous_fall_alert = False

    # --------------------------------------------------------- recording
    def start_record(self, label, person, room, seconds):
        if self.rec["status"] not in ("idle", "done", "error"):
            return "already recording"
        if not self.rx.connected():
            return "no CSI stream - is the board in 'CSI stream' mode and on the same Wi-Fi?"
        if not data.safe_name(label) or not label.strip():
            return "choose a label"
        if data.safe_name(label).lower() == "fall" and int(seconds) < 8:
            return "fall recordings must be at least 8 seconds to capture the movement before and after the event"
        self._rec_stop.clear()
        threading.Thread(target=self._record, args=(label, person, room, seconds), daemon=True).start()
        return ""

    def _record(self, label, person, room, seconds):
        r = self.rec
        r.update(status="countdown", label=label, msg="", frames=0, left=3)
        for i in range(int(self.countdown_seconds), 0, -1):
            r["left"] = i
            time.sleep(1)
            if self._rec_stop.is_set():
                r.update(status="idle", msg="cancelled")
                return
        self.rx.start_recording()
        t0 = time.time()
        r["status"] = "recording"
        while time.time() - t0 < seconds and not self._rec_stop.is_set():
            r["left"] = int(seconds - (time.time() - t0)) + 1
            r["frames"] = self.rx.recording_count()
            time.sleep(0.1)
        res = self.rx.stop_recording()
        if self._rec_stop.is_set():
            r.update(status="idle", msg="cancelled")
            return
        if res is None or len(res[0]) < seconds * 15:
            r.update(status="error", msg="too few frames received - move the board closer to the router")
            return
        ts, rssi, amp = res
        data.save_take(DATA_DIR, label, person, room, ts, rssi, amp)
        rate = len(ts) / max(seconds, 1)
        r.update(status="done", msg=f"saved {len(ts)} frames ({rate:.0f}/s)", frames=len(ts))

    def stop_record(self):
        self._rec_stop.set()

    # ----------------------------------------------------------- training
    def start_train(self, epochs, use_pretrained):
        if self.train["status"] == "running":
            return "already training"
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            return "torch is not installed. Run:  pip install torch"
        self._train_stop.clear()
        self.train.update(status="running", epoch=0, epochs=epochs, log=[], metrics=None)
        threading.Thread(target=self._train, args=(epochs, use_pretrained), daemon=True).start()
        return ""

    def _train(self, epochs, use_pretrained):
        from .train import train
        t = self.train
        try:
            m = train(DATA_DIR, MODEL_DIR, epochs=epochs, fs=FS, win=WIN, hop=HOP,
                      init=PRETRAINED if use_pretrained else None,
                      log=lambda s: t["log"].append(s),
                      progress=lambda e, n: t.update(epoch=e, epochs=n),
                      should_stop=self._train_stop.is_set)
            t.update(status="done", metrics={"accuracy": m["test"]["accuracy"],
                                             "macro_f1": m["test"]["macro_f1"],
                                             "balanced_accuracy": m["test"].get("balanced_accuracy"),
                                             "best_val_macro_f1": m.get("best_val_macro_f1"),
                                             "fall": m["test"].get("fall"),
                                             "per_class": m["test"]["per_class"],
                                             "fall_threshold": m.get("fall_threshold")})
            self.load_model()
        except data.DatasetError as e:
            t["log"].append(f"ERROR: {e}")
            t["status"] = "error"
        except Exception as e:
            t["log"].append(f"ERROR: {type(e).__name__}: {e}")
            t["status"] = "error"

    # ------------------------------------------------------------- state
    @staticmethod
    def _readiness(takes):
        core = ("empty", "sitting_still", "walking", "sitting_down", "lying_down", "fall")
        warnings = []
        for label in core:
            count = int(takes.get(label, {}).get("takes", 0))
            if count == 0:
                warnings.append(f"missing {label}")
            elif count < (10 if label == "fall" else 5):
                warnings.append(f"{label}: {count} recordings; collect more independent takes")
        total = sum(v.get("takes", 0) for v in takes.values())
        ready = all(takes.get(label, {}).get("takes", 0) >= (10 if label == "fall" else 5)
                    for label in core)
        return {"ready": ready,
                "total_takes": total, "fall_takes": takes.get("fall", {}).get("takes", 0),
                "recommended_labels": list(core), "warnings": warnings}

    def calibrate_empty_room(self):
        if not self.rx.connected():
            return "no CSI stream"
        amp = self.rx.last_n(1000)
        try:
            result = self.fusion.calibrate_empty(amp)
            self.fusion_state = {**self.fusion_state, "presence": "empty", "ready": True, "calibrated": True}
            return ""
        except ValueError as e:
            return str(e)

    def acknowledge_fall(self):
        if self.predictor:
            self.predictor.acknowledge_fall()
        self._previous_fall_alert = False
        if self.pred:
            self.pred["fall_alert"] = False
            self.pred["fall_candidate"] = False

    def snapshot(self):
        rx = self.rx
        last = rx.last_n(80)
        takes = data.summary(DATA_DIR)
        readiness = self._readiness(takes)
        return {
            "connected": rx.connected(), "fps": round(rx.fps(), 1), "loss": round(rx.loss_pct(), 1),
            "rssi": rx.rssi, "port": rx.port, "error": rx.error,
            "energy": list(self.energy_hist),
            "fusion": dict(self.fusion_state),
            "heat": last.astype(int).tolist(),
            "pred": self.pred,
            "fall_events": list(self.fall_events),
            "readiness": readiness,
            "model": {"loaded": self.predictor is not None, "error": self.model_error,
                      "version": self.predictor.version if self.predictor else "",
                      "labels": self.predictor.labels if self.predictor else [],
                      "threshold": self.predictor.threshold if self.predictor else 0.65,
                      "pretrained_available": os.path.exists(PRETRAINED)},
            "rec": dict(self.rec),
            "train": {k: (v[-40:] if k == "log" else v) for k, v in self.train.items()},
            "takes": takes,
        }


def make_handler(state):
    html_path = os.path.join(HERE, "static", "index.html")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/api/state":
                return self._json(state.snapshot())
            if self.path in ("/", "/index.html"):
                b = open(html_path, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            self.send_error(404)

        def do_POST(self):
            if self.headers.get("X-Athena") != "1":
                return self._json({"ok": False, "error": "forbidden"}, 403)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._json({"ok": False, "error": "bad json"}, 400)
            err = ""
            if self.path == "/api/record":
                err = state.start_record(str(body.get("label", "")), str(body.get("person", "")),
                                         str(body.get("room", "")), max(4, min(300, int(body.get("seconds", 20)))))
            elif self.path == "/api/record/stop":
                state.stop_record()
            elif self.path == "/api/train":
                err = state.start_train(max(1, min(200, int(body.get("epochs", 60)))), bool(body.get("pretrained")))
            elif self.path == "/api/train/stop":
                state._train_stop.set()
            elif self.path == "/api/label/delete":
                data.delete_label(DATA_DIR, str(body.get("label", "")))
            elif self.path == "/api/model/reload":
                state.load_model()
            elif self.path == "/api/fall/ack":
                state.acknowledge_fall()
            elif self.path == "/api/fusion/calibrate-empty":
                err = state.calibrate_empty_room()
            else:
                return self.send_error(404)
            self._json({"ok": not err, "error": err})

    return H


def run(port_udp=5005, port_http=8000, open_browser=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    st = State(port_udp)
    st.rx.start()
    threading.Thread(target=st.loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port_http), make_handler(st))
    url = f"http://127.0.0.1:{port_http}"
    print(f"Athena is running:  {url}\n(listening for the ESP32-S3 on UDP {port_udp})   Ctrl+C to quit")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
