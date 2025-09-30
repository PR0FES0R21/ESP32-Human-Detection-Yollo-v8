import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2  # Menggantikan imageio

# --- KONFIGURASI ---
FOLDER_REKAMAN = "recordings"
SEEK_SECONDS = 5
# --- AKHIR KONFIGURASI ---

class VideoReviewerApp:
    def __init__(self, master):
        self.master = master
        master.title("Video Reviewer v2.4 (OpenCV Engine)")
        master.configure(bg="#2e2e2e")

        # --- State Management ---
        self.video_list = []
        self.current_video_index = -1
        self.is_paused = True
        self.playback_speed = 1.0
        self.current_frame_num = 0
        self.total_frames = 0
        self.fps = 30
        self.playback_job = None
        self.is_seeking = False
        self.cap = None  # Objek VideoCapture dari OpenCV

        # --- GUI Elements (Sama seperti sebelumnya) ---
        self.video_canvas = tk.Label(master, bg="black")
        self.video_canvas.pack(pady=10, padx=10, fill="both", expand=True)

        self.progress_slider = ttk.Scale(master, from_=0, to=100, orient="horizontal", command=self.seek_from_slider_command)
        self.progress_slider.pack(fill="x", padx=10, pady=(0, 5))
        self.progress_slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.progress_slider.bind("<ButtonRelease-1>", self.on_slider_release)

        control_frame = tk.Frame(master, bg="#2e2e2e")
        control_frame.pack(fill="x", padx=10)

        self.time_label = tk.Label(control_frame, text="00:00 / 00:00", bg="#2e2e2e", fg="white")
        self.time_label.pack(side="left", padx=10)
        self.rewind_button = tk.Button(control_frame, text="◄◄ 5s", command=self.seek_backward, bg="#4f4f4f", fg="white", relief="flat")
        self.rewind_button.pack(side="left", padx=5)
        self.play_pause_button = tk.Button(control_frame, text="▶ Play", command=self.toggle_play_pause, width=10, bg="#4f4f4f", fg="white", relief="flat")
        self.play_pause_button.pack(side="left", padx=5)
        self.forward_button = tk.Button(control_frame, text="5s ►►", command=self.seek_forward, bg="#4f4f4f", fg="white", relief="flat")
        self.forward_button.pack(side="left", padx=5)
        self.speed_button = tk.Button(control_frame, text="1x Speed", command=self.toggle_speed, width=10, bg="#4f4f4f", fg="white", relief="flat")
        self.speed_button.pack(side="left", padx=5)
        tk.Frame(control_frame, bg="#2e2e2e").pack(side="left", fill="x", expand=True)
        self.next_button = tk.Button(control_frame, text="Next (Keep)", command=lambda: self.load_next_video(), bg="#28a745", fg="white", relief="flat")
        self.next_button.pack(side="right", padx=5)
        self.delete_button = tk.Button(control_frame, text="Delete & Next", command=lambda: self.delete_and_next(), bg="#dc3545", fg="white", relief="flat", font=("Helvetica", 9, "bold"))
        self.delete_button.pack(side="right", padx=5)
        
        self.disable_controls()
        self.load_videos()
        
        master.bind('<space>', self.toggle_play_pause)
        master.bind('<Right>', self.seek_forward)
        master.bind('<Left>', self.seek_backward)
        master.bind('<n>', lambda e: self.load_next_video())
        master.bind('<d>', lambda e: self.delete_and_next())
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- PERUBAHAN UTAMA: MENGGUNAKAN OPENCV ---

    def load_video_file(self, path):
        self.stop_playback()
        if self.cap is not None:
            self.cap.release()

        if not os.path.exists(path):
            print(f"File tidak ditemukan, skip: {path}")
            self.load_next_video()
            return
        
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"OpenCV gagal membuka video:\n{path}\n\nMungkin file corrupt atau codec tidak didukung.")
            self.load_next_video()
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  # Fallback FPS
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames < 1: self.total_frames = 1

        self.is_paused = True
        self.current_frame_num = 0
        self.playback_speed = 1.0
        self.progress_slider.config(to=self.total_frames - 1)
        self.master.title(f"Video Reviewer - {os.path.basename(path)}")
        self.enable_controls()
        self.update_gui_for_frame(0)

    def update_gui_for_frame(self, frame_num):
        if self.cap is None: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            self.update_canvas(frame)
            if not self.is_seeking:
                self.progress_slider.set(frame_num)
            self.update_time_label()
        else:
            # Jika gagal baca frame, mungkin sudah di akhir
            self.handle_playback_end()

    def playback_loop(self):
        if self.is_paused or self.is_seeking or self.cap is None:
            return

        self.current_frame_num += self.playback_speed
        if self.current_frame_num >= self.total_frames:
            self.handle_playback_end()
            return

        # Dengan OpenCV, lebih efisien set posisi dulu baru read
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.current_frame_num))
        ret, frame = self.cap.read()
        
        if ret:
            self.update_canvas(frame)
            if not self.is_seeking:
                self.progress_slider.set(self.current_frame_num)
            self.update_time_label()
            
            delay_ms = int(1000 / self.fps) # Delay tidak perlu memperhitungkan speed karena frame sudah di-skip
            self.playback_job = self.master.after(delay_ms, self.playback_loop)
        else:
            # Gagal membaca frame, anggap video selesai
            self.handle_playback_end()

    def update_canvas(self, frame_data):
        # OpenCV membaca warna sebagai BGR, perlu diubah ke RGB untuk Tkinter
        frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2: return
        
        frame_image.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=frame_image)
        self.video_canvas.config(image=self.photo)

    def on_closing(self):
        """Pastikan resource dilepaskan saat window ditutup."""
        self.stop_playback()
        if self.cap is not None:
            self.cap.release()
        self.master.destroy()

    # --- Sisa kode (tidak berubah) ---
    def on_slider_press(self, event):
        self.is_seeking = True
        self.stop_playback()
    def on_slider_release(self, event):
        self.is_seeking = False
        self.seek_from_slider_command(self.progress_slider.get())
    def seek_from_slider_command(self, value):
        if not self.is_seeking: self.seek_to_frame(int(float(value)))
    def seek_to_frame(self, frame_num):
        if self.total_frames <= 1: return
        self.current_frame_num = max(0, min(self.total_frames - 1, frame_num))
        self.update_gui_for_frame(self.current_frame_num)
        if not self.is_paused: self.start_playback()
    def seek_forward(self, event=None): self.seek_relative(SEEK_SECONDS)
    def seek_backward(self, event=None): self.seek_relative(-SEEK_SECONDS)
    def seek_relative(self, seconds):
        frame_offset = int(seconds * self.fps)
        target_frame = self.current_frame_num + frame_offset
        self.seek_to_frame(target_frame)
    def load_videos(self):
        try:
            files = [f for f in os.listdir(FOLDER_REKAMAN) if f.endswith(".mp4")]
            if not files: messagebox.showinfo("Info", f"Tidak ada file video (.mp4) di folder '{FOLDER_REKAMAN}'."); self.disable_controls(); return
            self.video_list = sorted([os.path.join(FOLDER_REKAMAN, f) for f in files])
            self.current_video_index = -1
            self.load_next_video()
        except FileNotFoundError: messagebox.showerror("Error", f"Folder '{FOLDER_REKAMAN}' tidak ditemukan!")
    def load_next_video(self, event=None):
        self.current_video_index += 1
        if self.current_video_index < len(self.video_list): self.load_video_file(self.video_list[self.current_video_index])
        else: self.end_of_list()
    def delete_and_next(self, event=None):
        if not self.video_list or self.current_video_index < 0: return
        file_to_delete = self.video_list[self.current_video_index]
        try:
            self.stop_playback()
            if self.cap: self.cap.release(); self.cap = None
            os.remove(file_to_delete)
            print(f"Deleted: {file_to_delete}")
            self.video_list.pop(self.current_video_index)
            self.current_video_index -= 1
            self.load_next_video()
        except Exception as e: messagebox.showerror("Error", f"Gagal menghapus file: {e}")
    def toggle_play_pause(self, event=None):
        if self.play_pause_button['state'] == 'disabled': return
        self.is_paused = not self.is_paused
        if self.is_paused: self.stop_playback(); self.play_pause_button.config(text="▶ Play")
        else:
            if self.current_frame_num >= self.total_frames - 1: self.current_frame_num = 0
            self.play_pause_button.config(text="⏸ Pause"); self.start_playback()
    def stop_playback(self):
        if self.playback_job: self.master.after_cancel(self.playback_job); self.playback_job = None
    def start_playback(self):
        self.stop_playback(); self.playback_loop()
    def toggle_speed(self):
        if self.playback_speed == 1.0: self.playback_speed = 2.0; self.speed_button.config(text="2x Speed")
        else: self.playback_speed = 1.0; self.speed_button.config(text="1x Speed")
    def update_time_label(self):
        current_time_s = self.current_frame_num / self.fps if self.fps > 0 else 0
        total_time_s = (self.total_frames - 1) / self.fps if self.fps > 0 else 0
        self.time_label.config(text=f"{self.format_time(current_time_s)} / {self.format_time(total_time_s)}")
    def format_time(self, seconds):
        minutes, seconds = divmod(int(seconds), 60); return f"{minutes:02d}:{seconds:02d}"
    def disable_controls(self):
        for btn in [self.play_pause_button, self.rewind_button, self.forward_button, self.speed_button, self.next_button, self.delete_button]: btn.config(state="disabled")
    def enable_controls(self):
        for btn in [self.play_pause_button, self.rewind_button, self.forward_button, self.speed_button, self.next_button, self.delete_button]: btn.config(state="normal")
    def end_of_list(self):
        self.disable_controls(); self.master.title("Video Reviewer - Selesai"); self.video_canvas.config(image=None); self.time_label.config(text="--:-- / --:--")
        messagebox.showinfo("Selesai", "Anda telah mencapai akhir dari daftar video.")
    def handle_playback_end(self):
        self.is_paused = True; self.stop_playback(); self.play_pause_button.config(text="▶ Play")
        self.current_frame_num = self.total_frames - 1
        if self.current_frame_num < 0: self.current_frame_num = 0
        self.progress_slider.set(self.current_frame_num)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = VideoReviewerApp(root)
    root.mainloop()