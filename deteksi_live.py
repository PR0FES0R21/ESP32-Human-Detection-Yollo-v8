# deteksi_live.py (dengan FPS)

import cv2
from ultralytics import YOLO
import time # BARU: Impor library 'time' untuk mengukur waktu

# --- Tahap 1: Konfigurasi dan Muat Model ---
URL_STREAM = "http://192.168.145.152:81/stream"
print("Memuat model YOLOv8...")
model = YOLO('yolov8n.pt')
print("Model berhasil dimuat.")

# --- Tahap 2: Buka Koneksi ke Video Stream ---
print(f"Mencoba menyambungkan ke stream di: {URL_STREAM}")
cap = cv2.VideoCapture(URL_STREAM)

if not cap.isOpened():
    print("!!! ERROR: Gagal membuka koneksi ke video stream.")
    exit()

print(">>> Koneksi berhasil! Memulai deteksi live...")
print(">>> Tekan tombol 'q' di jendela video untuk keluar.")

# --- BARU: Inisialisasi variabel untuk kalkulasi FPS ---
# prev_frame_time untuk menyimpan waktu dari frame sebelumnya
prev_frame_time = 0
# new_frame_time untuk menyimpan waktu dari frame saat ini
new_frame_time = 0

# --- Tahap 3: Loop untuk Memproses Setiap Frame Video ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("!!! Stream terputus. Menghentikan program.")
        break

    # --- BARU: Logika Kalkulasi FPS ---
    # 1. Ambil waktu saat ini
    new_frame_time = time.time()
    # 2. Hitung FPS. Rumusnya adalah 1 dibagi selisih waktu frame sekarang dan sebelumnya
    #    Kita tambahkan pengecekan untuk menghindari pembagian dengan nol di frame pertama.
    if (new_frame_time - prev_frame_time) > 0:
        fps = 1 / (new_frame_time - prev_frame_time)
    else:
        fps = 0 # Jika selisih waktu 0, set FPS ke 0
    # 3. Update waktu frame sebelumnya dengan waktu frame saat ini untuk iterasi berikutnya
    prev_frame_time = new_frame_time
    # 4. Ubah nilai FPS menjadi string agar bisa ditampilkan
    fps_text = f"FPS: {int(fps)}"
    # --- Akhir Logika Kalkulasi FPS ---

    # --- Proses Deteksi (Sama seperti sebelumnya) ---
    hasil_deteksi = model.predict(frame, classes=0, verbose=False)
    deteksi_pertama = hasil_deteksi[0]

    # --- Gambar Kotak (Sama seperti sebelumnya) ---
    for kotak in deteksi_pertama.boxes:
        koordinat = kotak.xyxy[0]
        x1, y1, x2, y2 = int(koordinat[0]), int(koordinat[1]), int(koordinat[2]), int(koordinat[3])
        skor = kotak.conf[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Manusia {skor:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- BARU: Tampilkan Teks FPS di Layar ---
    # Kita letakkan teks di pojok kiri atas (koordinat x=10, y=30)
    # Parameter: (gambar, teks, posisi, font, ukuran_font, warna_BGR, ketebalan)
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- Tahap 4: Tampilkan Hasil Live ---
    cv2.imshow("Deteksi Manusia Live - Tekan 'q' untuk Keluar", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Tombol 'q' ditekan. Menghentikan program.")
        break

# --- Tahap 5: Beres-beres ---
cap.release()
cv2.destroyAllWindows()