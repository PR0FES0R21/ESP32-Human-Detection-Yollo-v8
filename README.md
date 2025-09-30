# ESP32-CAM Human Detection System

Proyek ini menyediakan sistem deteksi manusia real-time untuk ESP32-CAM dengan buffer pradeteksi, perekaman otomatis, dan antarmuka HTTP opsional.

## Struktur Proyek

`
.
├── esp32_cam/
│   ├── __init__.py              # Ekspor Config dan class deteksi
│   ├── config.py                # Konfigurasi terpusat & override lewat environment
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── system.py            # Sistem deteksi utama + recording & display
│   │   └── optimized.py         # Varian dengan dukungan ONNX
│   └── api/
│       ├── __init__.py
│       └── server.py            # FastAPI server untuk kontrol via HTTP
├── esp32_human_detection.py     # Entry point CLI (wrapper system.py)
├── optimized_detector.py        # Entry point CLI untuk varian optimized
├── api_server.py                # Entry point CLI untuk FastAPI
├── requirements.txt
├── setup.py
└── recordings/                  # Folder output rekaman (ignored oleh git)
`

Direktori dugaan/ dan sementara/ menyimpan rekaman eksperimen; keduanya di-ignore secara default.

## Instalasi

1. Pastikan Python ≥ 3.10 terpasang.
2. Install dependensi:
   `ash
   pip install -r requirements.txt
   `
3. Letakkan model YOLO (default yolo8n.pt) di root project atau sesuaikan MODEL_PATH pada konfigurasi.

## Konfigurasi

Semua konfigurasi berada pada esp32_cam.config.Config. Nilai default dapat dioverride dengan environment variable berikut:

- ESP32_URL – URL stream ESP32-CAM (default http://192.168.145.152:81/stream)
- ESP32_CONTROL_PORT, ESP32_CONTROL_PATH – pengaturan endpoint kontrol kamera
- ESP32_CAMERA_SETTINGS – JSON string untuk mapping pengaturan kamera
- MODEL_PATH, OUTPUT_DIR, CONFIDENCE_THRESHOLD, SKIP_FRAMES, RECORDING_DELAY, dll.

Contoh override di shell:
`ash
set ESP32_URL=http://192.168.1.50:81/stream
set OUTPUT_DIR=c:\\data\\rekaman
`

## Menjalankan Sistem Deteksi

Jalankan entry point CLI standar:
`ash
python esp32_human_detection.py
`

Aplikasi akan mencoba terhubung ke stream, menerapkan pengaturan kamera dari Config, menampilkan feed dengan bounding box, dan otomatis merekam ketika manusia terdeteksi.

### Varian Optimized / ONNX
Untuk memakai pipeline optimized:
`ash
python optimized_detector.py
`

### API HTTP
Aktifkan REST API (FastAPI + Uvicorn):
`ash
python api_server.py
# atau
uvicorn esp32_cam.api.server:app --host 0.0.0.0 --port 8000
`
Endpoint penting:
- POST /api/start – Mulai deteksi
- POST /api/stop – Hentikan
- GET /api/status – Status sistem & rekaman
- GET /api/snapshot – Ambil frame terbaru dalam JPEG
- GET /api/recordings – Daftar rekaman terbaru

## Output

Rekaman video disimpan di folder ecordings/ dengan nama ecord_YYYYmmdd_HHMMSS.mp4. Snapshot manual (s) juga disimpan di folder yang sama.

## Catatan

- Pengaturan kamera (mis. flip, hmirror, ramesize) dikirim ulang setiap kali koneksi stream berhasil dibuat.
- Pastikan ESP32_URL mengarah ke stream MJPEG yang aktif.
- Jika koneksi terputus, sistem akan mencoba reconnect dengan delay yang diatur Config.RECONNECT_DELAY.

Selamat menggunakan! Jika ada masalah, periksa log dan pastikan alamat IP serta kredensial kamera benar.
