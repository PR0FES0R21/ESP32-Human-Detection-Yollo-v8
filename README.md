# ESP32-CAM Human Detection System

Proyek ini menyediakan sistem deteksi manusia real-time untuk ESP32-CAM dengan buffer pradeteksi, perekaman otomatis, dan antarmuka HTTP opsional.

## Struktur Proyek

```
.
|-- esp32_cam/
|   |-- __init__.py                # Ekspor Config dan class deteksi
|   |-- config.py                  # Konfigurasi terpusat & override lewat environment
|   |-- detection/
|   |   |-- __init__.py
|   |   |-- system.py              # Sistem deteksi utama + recording & display
|   |   `-- optimized.py           # Varian dengan dukungan ONNX
|   `-- api/
|       |-- __init__.py
|       `-- server.py              # FastAPI server untuk kontrol via HTTP
|-- esp32_human_detection.py       # Entry point CLI (wrapper system.py)
|-- optimized_detector.py          # Entry point CLI varian optimized
|-- api_server.py                  # Entry point CLI FastAPI
|-- requirements.txt
|-- setup.py
|-- README.md
|-- yolo8n.pt                      # Model YOLO (opsional, dapat diganti)
|-- recordings/                    # Output rekaman (digunakan runtime)
|-- dugaan/                        # Rekaman eksperimen (ignored)
`-- sementara/                     # Rekaman sementara (ignored)
```

Direktori `dugaan/` dan `sementara/` menyimpan file uji coba; keduanya diabaikan oleh Git melalui `.gitignore`.

## Instalasi

1. Pastikan Python >= 3.10 terpasang.
2. Pasang dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Letakkan model YOLO (default `yolo8n.pt`) di root project atau sesuaikan `MODEL_PATH` pada konfigurasi.

## Konfigurasi

Semua konfigurasi berada pada `esp32_cam.config.Config`. Nilai default bisa dioverride via environment variable berikut (contoh penting):

- `ESP32_URL` - URL stream ESP32-CAM (default `http://192.168.145.152:81/stream`)
- `ESP32_CONTROL_PORT`, `ESP32_CONTROL_PATH` - pengaturan endpoint kontrol kamera
- `ESP32_CAMERA_SETTINGS` - JSON untuk pengaturan kamera (mis. `{"vflip": 1, "framesize": 10}`)
- `MODEL_PATH`, `OUTPUT_DIR`, `CONFIDENCE_THRESHOLD`, `SKIP_FRAMES`, `RECORDING_DELAY`, dll.

Contoh override (Windows PowerShell):
```powershell
$env:ESP32_URL = "http://192.168.1.50:81/stream"
$env:OUTPUT_DIR = "C:/data/rekaman"
python esp32_human_detection.py
```

## Menjalankan Sistem Deteksi

Jalankan entry point CLI standar:
```bash
python esp32_human_detection.py
```
Aplikasi akan mencoba terhubung ke stream, menerapkan pengaturan kamera dari `Config`, menampilkan feed dengan bounding box, dan otomatis merekam ketika manusia terdeteksi.

### Varian Optimized / ONNX
Untuk memakai pipeline optimized:
```bash
python optimized_detector.py
```
Tambahkan argumen baris perintah sesuai kebutuhan (lihat bantuan pada file).

### API HTTP
Aktifkan REST API (FastAPI + Uvicorn):
```bash
python api_server.py
# atau
uvicorn esp32_cam.api.server:app --host 0.0.0.0 --port 8000
```
Endpoint penting:
- `POST /api/start` - Mulai deteksi
- `POST /api/stop` - Hentikan
- `GET /api/status` - Status sistem & rekaman
- `GET /api/snapshot` - Ambil frame terbaru (JPEG)
- `GET /api/recordings` - Daftar rekaman terbaru

## Output

Rekaman video disimpan di folder `recordings/` dengan nama `record_YYYYmmdd_HHMMSS.mp4`. Snapshot manual (`s`) juga disimpan di folder yang sama.

## Catatan

- Pengaturan kamera (mis. `vflip`, `hmirror`, `framesize`) dikirim ulang setiap kali koneksi stream berhasil dibuat.
- Pastikan `ESP32_URL` mengarah ke stream MJPEG yang aktif.
- Jika koneksi terputus, sistem akan mencoba reconnect berdasarkan `Config.RECONNECT_DELAY`.

Selamat menggunakan! Jika ada masalah, periksa log dan pastikan alamat IP serta kredensial kamera benar.
