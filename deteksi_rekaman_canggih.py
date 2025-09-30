# deteksi_rekaman_canggih.py

import cv2
from ultralytics import YOLO
import time
import os
from datetime import datetime
from collections import deque # PENTING: Untuk buffer pre-roll

# --- KONFIGURASI ---
URL_STREAM = "http://192.168.145.152:81/stream"
RECORDING_DIR = "rekaman" # Folder untuk menyimpan video
FPS_TARGET = 15.0 # Target FPS untuk video yang disimpan
PRE_EVENT_SECONDS = 3 # Berapa detik rekaman SEBELUM orang terdeteksi
POST_EVENT_COOLDOWN = 4.0 # Berapa detik menunggu SETELAH orang hilang

# --- Setup Awal ---
# Buat folder rekaman jika belum ada
if not os.path.exists(RECORDING_DIR):
    os.makedirs(RECORDING_DIR)

# Muat model YOLOv8n yang ringan dan cepat
print("Memuat model YOLOv8...")
model = YOLO('yolov8n.pt')
print("Model berhasil dimuat.")

# Buka koneksi ke video stream
print(f"Mencoba menyambungkan ke stream di: {URL_STREAM}")
cap = cv2.VideoCapture(URL_STREAM)
if not cap.isOpened():
    print("!!! ERROR: Gagal membuka koneksi ke video stream.")
    exit()

# Dapatkan resolusi video dari frame pertama untuk setup VideoWriter
ret, frame = cap.read()
if not ret:
    print("!!! ERROR: Tidak bisa membaca frame pertama dari stream.")
    exit()
FRAME_WIDTH = frame.shape[1]
FRAME_HEIGHT = frame.shape[0]

print(">>> Koneksi berhasil! Memulai deteksi...")
print(">>> Tekan tombol 'q' di jendela video untuk keluar.")


# --- Inisialisasi Variabel untuk Logika Rekaman & FPS ---
# Buffer untuk menyimpan frame pre-roll (3 detik * 15 FPS = 45 frame)
buffer_size = PRE_EVENT_SECONDS * int(FPS_TARGET)
pre_event_buffer = deque(maxlen=buffer_size)

is_recording = False
video_writer = None
time_last_seen = 0

prev_frame_time = 0

# --- Loop Utama ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("!!! Stream terputus.")
        break

    current_time = time.time()
    
    # --- LOGIKA INTI: Perekaman Canggih ---
    
    # 1. Selalu tambahkan frame saat ini ke buffer pre-roll
    pre_event_buffer.append(frame.copy())
    
    # 2. Lakukan deteksi manusia
    hasil_deteksi = model.predict(frame, classes=0, verbose=False, conf=0.5)
    
    # Cek apakah ada manusia yang terdeteksi
    if len(hasil_deteksi[0].boxes) > 0:
        # Jika terdeteksi, update waktu terakhir terlihat
        time_last_seen = current_time
        
        # Jika kita BELUM merekam, ini adalah AWAL dari sebuah kejadian
        if not is_recording:
            is_recording = True
            
            # Buat nama file video yang unik berdasarkan waktu
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_path = os.path.join(RECORDING_DIR, f"event_{timestamp}.avi")
            
            # Siapkan objek untuk menulis video
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(file_path, fourcc, FPS_TARGET, (FRAME_WIDTH, FRAME_HEIGHT))
            
            print(f"--- [Mulai Merekam] Manusia terdeteksi! Menyimpan ke {os.path.basename(file_path)} ---")
            
            # PENTING: Tulis semua frame dari buffer ke video (pre-roll)
            print(f"    -> Menulis {len(pre_event_buffer)} frame dari buffer pre-roll...")
            for buffered_frame in list(pre_event_buffer):
                video_writer.write(buffered_frame)
    else:
        # Jika TIDAK ada manusia terdeteksi
        # Cek apakah kita SEDANG merekam DAN cooldown sudah terlewati
        if is_recording and (current_time - time_last_seen) > POST_EVENT_COOLDOWN:
            print(f"--- [Selesai Merekam] Cooldown {POST_EVENT_COOLDOWN} detik selesai. Video disimpan. ---")
            is_recording = False
            if video_writer:
                video_writer.release()
                video_writer = None
    
    # 3. Jika status sedang merekam, terus tulis frame ke video
    # Ini akan merekam frame saat orang terdeteksi DAN selama masa cooldown
    if is_recording and video_writer:
        video_writer.write(frame)

    # --- Logika Visualisasi (Tidak berubah) ---
    # Kalkulasi FPS
    if (current_time - prev_frame_time) > 0:
        fps = 1 / (current_time - prev_frame_time)
    else:
        fps = 0
    prev_frame_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Gambar kotak deteksi
    for kotak in hasil_deteksi[0].boxes:
        x1, y1, x2, y2 = [int(i) for i in kotak.xyxy[0]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    # Tampilkan status rekaman
    status_text = "RECORDING" if is_recording else "STANDBY"
    status_color = (0, 0, 255) if is_recording else (0, 255, 0)
    cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Tampilkan frame ke layar
    cv2.imshow("Deteksi & Rekaman Canggih - Tekan 'q' untuk Keluar", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Tombol 'q' ditekan. Menghentikan program.")
        break

# --- Beres-beres ---
# Jika program dihentikan saat sedang merekam, pastikan video terakhir disimpan
if video_writer:
    print("Menyimpan rekaman terakhir sebelum keluar...")
    video_writer.release()

cap.release()
cv2.destroyAllWindows()