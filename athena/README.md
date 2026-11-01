# ATHENA 3.0 — Unified Wi-Fi CSI Motion, Presence & Fall Research Prototype

Athena receives 64-subcarrier Wi-Fi CSI amplitude frames from the ESP32-S3, records labelled examples, trains a temporal model, and presents activity probabilities plus a separate **possible-fall-pattern alert**.

> **Important safety limitation:** Athena is an experimental research prototype, not a medical device, emergency alarm, fall detector certified for real-world safety, or substitute for supervision. It can miss real events and generate false alarms. Never rely on it to protect a person. Do not perform real falls or risky movements to collect data. Any staged examples must be safe, authorized, performed on a padded surface, and supervised by a qualified adult.

## Fall AI and training improvements

- Fall-specific prominent alert panel, probability meter, acknowledgement action, and event history.
- 3.2-second temporal window (50 Hz × 160 frames) with 0.5-second hop, so the model can inspect context around a short event. This increases startup latency; the dashboard shows a warm-up state until a complete window exists.
- Multi-scale temporal residual blocks added to the existing CNN + BiLSTM backbone.
- Class-balanced loss, signal-preserving augmentation, gradient clipping, learning-rate scheduling, early stopping, and best-validation checkpoint restoration.
- Validation-only temperature calibration and a validation-only fall threshold; the test set is not used to choose either value.
- Recording-level splitting (all overlapping windows from one recording remain in the same split).
- Long CSI packet gaps are marked invalid rather than filled with invented linear transitions.
- Old `athena-own-1` checkpoints still load with their original architecture. They do not contain a fall class unless the old checkpoint actually has one.
- Dashboard dataset-readiness counts and per-class test metrics, including fall precision and recall.

## Run

1. Put the ESP32-S3 in **Unified ATHENA (role 3)** mode and direct its stream to the computer's UDP port `5005` (or use broadcast on the same LAN).
2. On Windows, double-click `run_athena.bat`; or open a terminal in the `athena` folder and run:

   ```powershell
   python -m pip install -r requirements.txt
   python -m athena
   ```

3. Open `http://127.0.0.1:8000` if it does not open automatically.
   The ESP32 firmware page also has a **Unified ATHENA** tab that embeds this unified dashboard when opened in a browser on the same computer.
4. Verify connection rate and packet loss before recording. Keep the board and router fixed while collecting one set of recordings.
5. Record several independent clips for every class, including negative/confounding activities. For `fall`, use only safe, authorized, supervised, padded-surface data; do not stage a real fall.
6. Train from the dashboard. The app saves `models/activity_model.pt` and `models/metrics.json`.

To run a synthetic stream without hardware:

```bash
python tools/fake_esp.py
```

The fake `fall` waveform is only a synthetic UI/pipeline smoke test. It is not realistic human CSI and must never be used to claim fall-detection accuracy or train a real deployment model.

## Data collection protocol

- Begin with at least 10–20 independent recordings per class; 20+ is better for a fall class. The application can train with fewer recordings for software testing, but the resulting performance is not trustworthy.
- Include `empty`, `sitting_still`, `walking`, `sitting_down`, `lying_down`, and `fall`. Add `bending`, `squatting`, `turning`, and ordinary transitions as hard negatives where relevant.
- Each fall sample should include a few seconds before and after the labelled event. The recording form requires at least 8 seconds for the `fall` label.
- Keep person and room identifiers consistent. Vary direction, distance, clothing, time, and permitted device placements across separate recordings.
- Obtain informed permission before recording. Do not record private spaces or bystanders without permission.
- A room/person/device held out from training is required for a genuine generalization claim. The ordinary train/validation/test split only holds out recordings, not necessarily people or rooms.

## Model and evaluation

- `athena/data.py`: timestamp unwrapping, gap-aware resampling, normalization, windows, and recording-level splits.
- `athena/model.py`: multi-scale temporal model and legacy checkpoint architecture.
- `athena/train.py`: training, early stopping, validation calibration, and test metrics.
- `athena/live.py`: live inference and a latched possible-fall alert.
- `athena/app.py`: local API, recording, training orchestration, readiness checks.
- `athena/static/index.html`: bilingual dashboard.
- `athena/crossval.py`: leave-one-room/person-out evaluation.

Metrics to report for the fall class are **precision, recall, F1, and false-positive windows**, in addition to overall accuracy and Macro-F1. A high overall accuracy can hide missed falls when fall examples are rare. Evaluate false alarms per hour using continuous, independently labelled negative recordings before making any claims about real deployment.

Run the checks:

```bash
python tests/test_pipeline.py
python tests/test_crossval.py
python tests/test_fall_model.py
```

## Stream packet

Little-endian UDP packet: `"CSI1" | seq u32 | ts_ms u32 | rssi i8 | n u8 | amp[n] u8`.
The expected CSI amplitude vector has 64 values.


## ATHENA 3.0 unified mode

Set the ESP32-S3 role to **Unified ATHENA (role 3)**. In this mode the firmware owns one CSI callback and streams the CSI frames to the PC; the PC runs three concurrent paths from that same stream: motion variation, calibrated empty-room presence estimation, and the trained fall/activity classifier. This avoids registering a second Wi-Fi CSI callback over the ESP Wi-Fi Sensing FSM callback. The original firmware-only motion/presence mode remains available as role 0.

Start the PC app with `python -m athena`, then set the board's PC IPv4 address and UDP port 5005. In the ATHENA dashboard, first calibrate an empty room from the Motion/Presence cards. Fall AI remains unavailable until a model trained with real, independently recorded fall and non-fall CSI examples is installed. The motion and presence outputs in unified mode are experimental signal estimators and require per-room validation; they are not safety alarms.
