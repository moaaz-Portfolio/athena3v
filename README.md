# 📡 ESP32-S3 WiFi Radar

كاشف حركة بالواي فاي (Wi-Fi CSI) على **ESP32-S3** — من غير كاميرا ولا حساسات.
Wi-Fi CSI motion sensor for **ESP32-S3** — no camera, no extra sensors.

مبني على مكتبة Espressif `esp_wifi_sensing` (ESP-CSI)، وفكرته مأخوذة من مشروع DrWalls (Apache-2.0).
Built on Espressif's `esp_wifi_sensing`; inspired by DrWalls (Apache-2.0).

## الميزات / Features

- **تنصيب من المتصفح** بزرار واحد (esp-web-tools) / one-click browser install
- **بوابة إعداد** فيها قائمة شبكات + كابتيف بورتال / setup portal with network list
- **لوحة تحكم عربي/إنجليزي** + رسم بياني لحظي / bilingual live dashboard
- **حساسية الحركة والتواجد**، وقت الإبقاء، معايرة للأوضة / motion + presence tuning, calibration
- **لمبة RGB (WS2812) أو GPIO LED**: لون وسطوع و GPIO من الصفحة
- **مخرج ريلاي/بازر** على أي GPIO / relay-buzzer output
- **MQTT + Home Assistant** (auto-discovery: motion + presence) و **Webhook**
- **وضع Pair**: لوحتين S3 (مُرسل + حساس) بدون الاعتماد على الراوتر / two-board mode
- **تحديث OTA** من الصفحة، وباسورد اختياري للوحة / OTA update + optional password
- زرار **BOOT** 3 ثواني = نسيان الواي فاي

## التنصيب (من غير ما تثبّت أي حاجة)

1. اعمل Fork/Upload للمشروع على GitHub (branch `main`).
2. من Settings → Pages اختار **Source: GitHub Actions**.
3. الـ Action (`build.yml`) بيبني الفيرموير وينشر صفحة التنصيب على `https://<user>.github.io/<repo>/`.
4. افتح الصفحة بـ Chrome/Edge، وصّل الـ S3 بـ USB، واضغط **Install**.

## الاستخدام

1. بعد التنصيب اتصل بشبكة **S3-Radar-Setup**، اختار شبكتك واكتب الباسورد.
2. افتح `http://<اسم-الجهاز>.local` (الاسم الافتراضي `s3radar-xxxxxx`).
3. اضبط الحساسية، واعمل **معايرة** والأوضة فاضية.

## البناء المحلي (اختياري) / Local build

```bash
idf.py set-target esp32s3
idf.py build flash monitor
```
ESP-IDF 5.4 أو أحدث. المكتبات المُدارة بتتنزل تلقائياً.

## الـ GPIO الافتراضية (DevKitC-1)

| الوظيفة | GPIO |
|---|---|
| LED (WS2812) | 48 (بعض النسخ 38) — غيّرها من تبويب «المخرجات» |
| زرار الريست للواي فاي | 0 (BOOT) |
| ريلاي | معطّل افتراضياً |

## وضع Pair (دقة أعلى)

1. لوحة 1: ثبّت الفيرموير واختار الدور **مُرسل** (تبويب الشبكة)، وانسخ الـ MAC بتاعها.
2. لوحة 2: دور **حساس**، حط الـ MAC في «MAC اللوحة المساعدة»، واختار «مصدر الإشارة» = لوحة مساعدة.
3. الاتنين لازم يكونوا على نفس شبكة الواي فاي. حطهم على جنبين مكان الحركة.

## 🧠 ATHENA 3.0 — الحركة والتواجد والسقوط في نظام موحّد

الدور الرابع «ATHENA موحّد» يبث CSI مرة واحدة إلى الكمبيوتر، ثم يشغّل محركات الحركة والتواجد وكشف السقوط/النشاط داخل لوحة واحدة. كاشف الحركة والتواجد الأصلي داخل ESP32 يظل متاحًا في الدور الأول. راجع `athena/README.md` للتوصيل والمعايرة والتدريب.

**تنبيه:** النظام نموذج بحثي تجريبي، وليس جهازًا طبيًا أو نظام إنقاذ أو إنذارًا موثوقًا للسلامة. لا تعتمد عليه لحماية أي شخص.

## REST API

| Endpoint | |
|---|---|
| `GET /api/status` | motion, presence, jitter, rssi… |
| `GET/POST /api/config` | كل الإعدادات (JSON) |
| `POST /api/calibrate` | `{"action":"start|stop|clear|baseline"}` |
| `POST /api/ota` | body = ملف `esp32s3-wifi-radar.bin` |

أي POST لازم يحمل الهيدر `X-Radar: 1` (حماية من CSRF). لو فعّلت باسورد: Basic Auth، المستخدم `admin`.

## Home Assistant

فعّل MQTT واكتب عنوان الـ broker — بيظهر الجهاز تلقائياً بـ `Motion` و `Presence`.

## ملاحظات

- الكشف بيعتمد على مكان اللوحة والراوتر والغرفة؛ جرّب أماكن مختلفة.
- تجنب GPIO 26–32 (فلاش/PSRAM) و19–20 (USB) للمخرجات.
- التحديث من الصفحة: استخدم `esp32s3-wifi-radar.bin` مش الـ merged.

## License

Apache-2.0. Uses Espressif ESP-CSI components; `components/dns_server` is from ESP-IDF examples (Unlicense/CC0).

## ATHENA unified dashboard

The ESP32 firmware page (`moaaz.local`) includes a **Unified ATHENA** tab that embeds the dashboard from `http://127.0.0.1:8000` (the browser must be running on the same computer as ATHENA). In unified role 3, the ESP32 streams CSI to the computer and the Python app estimates motion and calibrated presence while the trained model handles activity/fall recognition.

1. In the ESP32 page, open **Network & mode** and set **Role** to **Unified ATHENA (motion + presence + fall AI)**. Leave the PC address empty for UDP broadcast on the LAN, or enter the computer's IPv4 address. Use UDP port `5005`.
2. On that same computer, open a terminal in the `athena` folder and run `python -m pip install -r requirements.txt`, then `python -m athena`.
3. Return to the ESP32 page and open **Unified ATHENA**. It displays all detection panels in the same dashboard. The same dashboard is also available directly at `http://127.0.0.1:8000`.
4. Record and train a model that includes the `fall` class before interpreting live fall probabilities. The bundled public/pretrained weights are only a warm-start for training, not a validated detector for your room or hardware.

If the ESP32 page is opened on a different device (for example, a phone), `127.0.0.1` refers to that device. Run ATHENA on the same device as the browser, or open its dashboard directly on the computer where it is running. This is an experimental research system, not an emergency or medical alarm.


