"""
Live Captions - Real-time English to Chinese subtitle
Mouse over to show controls
"""

import numpy as np
import sounddevice as sd
import mlx_whisper
from deep_translator import GoogleTranslator
import queue
import threading
import tkinter as tk
import time
import collections
import datetime
import wave
import os
import re

# Settings
SAMPLE_RATE = 16000
WINDOW_SECONDS = 6
STEP_SECONDS = 1.5
SILENCE_THRESHOLD = 0.015
MODEL = "mlx-community/whisper-large-v3-turbo"

SENTENCE_END = re.compile(r'[.!?]\s*$')

subtitle_queue = queue.Queue()
translator = GoogleTranslator(source="en", target="zh-CN")

audio_buffer = collections.deque()
audio_lock = threading.Lock()
total_samples_count = 0

full_recording = []
recording_lock = threading.Lock()
is_recording = True


def audio_callback(indata, frames, time_info, status):
    global total_samples_count
    data = indata.copy()
    with audio_lock:
        audio_buffer.append(data)
        total_samples_count += frames
        max_samples = SAMPLE_RATE * WINDOW_SECONDS * 3
        while total_samples_count > max_samples and audio_buffer:
            removed = audio_buffer.popleft()
            total_samples_count -= removed.shape[0]
    with recording_lock:
        if is_recording:
            full_recording.append(data)


def get_window():
    with audio_lock:
        if not audio_buffer:
            return None
        chunks = list(audio_buffer)
    audio_np = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
    needed = SAMPLE_RATE * WINDOW_SECONDS
    if len(audio_np) >= needed:
        return audio_np[-needed:]
    return audio_np


last_text_en = ""
transcript_log = []


def transcribe_loop():
    global last_text_en
    print("Loading model...")
    time.sleep(WINDOW_SECONDS)
    print("Ready.")

    while True:
        start = time.time()
        audio_np = get_window()

        if audio_np is None or len(audio_np) < SAMPLE_RATE:
            time.sleep(0.3)
            continue

        rms = np.sqrt(np.mean(audio_np ** 2))
        if rms < SILENCE_THRESHOLD:
            time.sleep(STEP_SECONDS)
            continue

        try:
            result = mlx_whisper.transcribe(
                audio_np,
                path_or_hf_repo=MODEL,
                language="en",
                verbose=False,
                no_speech_threshold=0.4,
                hallucination_silence_threshold=1.0,
            )
            text_en = result["text"].strip()

            if not text_en or len(text_en) < 4:
                time.sleep(STEP_SECONDS)
                continue

            if text_en == last_text_en:
                time.sleep(STEP_SECONDS)
                continue

            last_text_en = text_en

            if SENTENCE_END.search(text_en):
                try:
                    text_zh = translator.translate(text_en)
                except Exception:
                    text_zh = "[Translation failed]"
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                transcript_log.append((ts, text_en, text_zh))
                subtitle_queue.put((ts, text_en, text_zh))

        except Exception as e:
            print(f"Error: {e}")

        elapsed = time.time() - start
        time.sleep(max(0, STEP_SECONDS - elapsed))


def save_transcript():
    if not transcript_log:
        return None
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(os.path.expanduser("~/Desktop"), f"transcript_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Live Captions Transcript\n")
        f.write(f"Saved: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        for ts_str, en, zh in transcript_log:
            f.write(f"[{ts_str}]\n")
            f.write(f"EN: {en}\n")
            f.write(f"ZH: {zh}\n\n")
    return path


def save_audio():
    with recording_lock:
        if not full_recording:
            return None
        data = np.concatenate(full_recording, axis=0).flatten()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(os.path.expanduser("~/Desktop"), f"recording_{ts}.wav")
    data_int16 = (data * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data_int16.tobytes())
    return path


class SubtitleWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Captions")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        self.root.configure(bg="#111111")

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w, h = 980, 200
        x = (sw - w) // 2
        y = sh - h - 60
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.resizable(True, True)

        self.font_en = 16
        self.font_zh = 20
        # 用户是否手动往上翻了
        self.user_scrolled_up = False
        self.ctrl_visible = False

        # 控制栏（默认隐藏）
        self.top = tk.Frame(self.root, bg="#1c1c1c", pady=5)

        btn_cfg = dict(bg="#1c1c1c", fg="#777777", relief="flat",
                       font=("Helvetica", 12), padx=4,
                       activebackground="#2a2a2a", activeforeground="white",
                       cursor="hand2", borderwidth=0)

        tk.Label(self.top, text="EN", bg="#1c1c1c", fg="#555555",
                 font=("Helvetica", 11)).pack(side="left", padx=(10, 1))
        tk.Button(self.top, text="－", command=self.dec_en, **btn_cfg).pack(side="left")
        self.lbl_en = tk.Label(self.top, text=str(self.font_en),
                               bg="#1c1c1c", fg="#555555", font=("Helvetica", 11), width=3)
        self.lbl_en.pack(side="left")
        tk.Button(self.top, text="＋", command=self.inc_en, **btn_cfg).pack(side="left")

        tk.Label(self.top, text="|", bg="#1c1c1c", fg="#333333",
                 font=("Helvetica", 13)).pack(side="left", padx=6)

        tk.Label(self.top, text="ZH", bg="#1c1c1c", fg="#555555",
                 font=("Helvetica", 11)).pack(side="left", padx=(0, 1))
        tk.Button(self.top, text="－", command=self.dec_zh, **btn_cfg).pack(side="left")
        self.lbl_zh = tk.Label(self.top, text=str(self.font_zh),
                               bg="#1c1c1c", fg="#555555", font=("Helvetica", 11), width=3)
        self.lbl_zh.pack(side="left")
        tk.Button(self.top, text="＋", command=self.inc_zh, **btn_cfg).pack(side="left")

        tk.Label(self.top, text="|", bg="#1c1c1c", fg="#333333",
                 font=("Helvetica", 13)).pack(side="left", padx=6)

        tk.Button(self.top, text="Save", command=self.on_save_transcript, **btn_cfg
                  ).pack(side="left", padx=2)
        tk.Button(self.top, text="Audio", command=self.on_save_audio, **btn_cfg
                  ).pack(side="left", padx=2)

        tk.Label(self.top, text="|", bg="#1c1c1c", fg="#333333",
                 font=("Helvetica", 13)).pack(side="left", padx=6)

        tk.Button(self.top, text="Latest", command=self.scroll_to_bottom, **btn_cfg
                  ).pack(side="left", padx=2)

        self.status_label = tk.Label(self.top, text="", bg="#1c1c1c", fg="#2ecc71",
                                     font=("Helvetica", 11))
        self.status_label.pack(side="left", padx=10)

        # 文本区域
        frame = tk.Frame(self.root, bg="#111111")
        frame.pack(expand=True, fill="both")
        self.frame = frame

        scrollbar = tk.Scrollbar(frame, bg="#1a1a1a", troughcolor="#111111",
                                 relief="flat", width=6)
        scrollbar.pack(side="right", fill="y")

        self.text_box = tk.Text(
            frame,
            bg="#111111",
            fg="white",
            font=("Helvetica", self.font_en),
            wrap="word",
            state="disabled",
            relief="flat",
            highlightthickness=0,
            borderwidth=0,
            yscrollcommand=scrollbar.set,
            padx=16,
            pady=8,
            spacing1=2,
            spacing2=0,
            spacing3=8,
        )
        self.text_box.pack(expand=True, fill="both")
        scrollbar.config(command=self.text_box.yview)

        self.text_box.tag_config("time", foreground="#3a3a3a", font=("Helvetica", 10))
        self.text_box.tag_config("en", foreground="#666666",
                                 font=("Helvetica", self.font_en))
        self.text_box.tag_config("zh", foreground="white",
                                 font=("PingFang SC", self.font_zh, "bold"))
        self.text_box.tag_config("sep", foreground="#1e1e1e", font=("Helvetica", 3))
        self.text_box.tag_config("loading", foreground="#444444",
                                 font=("Helvetica", self.font_en))

        self.text_box.config(state="normal")
        self.text_box.insert("end", "Loading model...", "loading")
        self.text_box.config(state="disabled")

        # 用鼠标滚轮方向判断是否向上翻
        self.text_box.bind("<MouseWheel>", self._on_mousewheel)
        self.text_box.bind("<Button-4>", self._on_mousewheel)
        self.text_box.bind("<Button-5>", self._on_scroll_down)

        self.first_entry = True

        # 定时检测鼠标位置来控制控制栏显隐
        self._check_hover()
        self.root.after(150, self.poll)

    def _is_mouse_inside(self):
        try:
            mx = self.root.winfo_pointerx()
            my = self.root.winfo_pointery()
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            return rx <= mx <= rx + rw and ry <= my <= ry + rh
        except Exception:
            return False

    def _check_hover(self):
        inside = self._is_mouse_inside()
        if inside and not self.ctrl_visible:
            self.top.pack(fill="x", before=self.frame)
            self.ctrl_visible = True
        elif not inside and self.ctrl_visible:
            self.top.pack_forget()
            self.ctrl_visible = False
        self.root.after(200, self._check_hover)

    def _on_mousewheel(self, event):
        # 向上滚
        delta = getattr(event, "delta", 0)
        num = getattr(event, "num", 0)
        if delta > 0 or num == 4:
            self.user_scrolled_up = True

    def _on_scroll_down(self, event):
        # 向下滚到底时恢复自动跟随
        self.root.after(100, self._check_at_bottom)

    def _check_at_bottom(self):
        pos = self.text_box.yview()
        if pos[1] >= 0.99:
            self.user_scrolled_up = False

    def scroll_to_bottom(self):
        self.user_scrolled_up = False
        self.text_box.yview_moveto(1.0)

    def _append(self, ts, text_en, text_zh):
        self.text_box.config(state="normal")
        self.text_box.insert("end", f"[{ts}]  ", "time")
        self.text_box.insert("end", f"{text_en}\n", "en")
        self.text_box.insert("end", f"{text_zh}\n", "zh")
        self.text_box.insert("end", "─" * 55 + "\n", "sep")
        self.text_box.config(state="disabled")
        if not self.user_scrolled_up:
            self.text_box.yview_moveto(1.0)

    def inc_en(self):
        self.font_en = min(self.font_en + 1, 60)
        self.text_box.tag_config("en", font=("Helvetica", self.font_en))
        self.lbl_en.config(text=str(self.font_en))

    def dec_en(self):
        self.font_en = max(self.font_en - 1, 8)
        self.text_box.tag_config("en", font=("Helvetica", self.font_en))
        self.lbl_en.config(text=str(self.font_en))

    def inc_zh(self):
        self.font_zh = min(self.font_zh + 1, 60)
        self.text_box.tag_config("zh", font=("PingFang SC", self.font_zh, "bold"))
        self.lbl_zh.config(text=str(self.font_zh))

    def dec_zh(self):
        self.font_zh = max(self.font_zh - 1, 8)
        self.text_box.tag_config("zh", font=("PingFang SC", self.font_zh, "bold"))
        self.lbl_zh.config(text=str(self.font_zh))

    def on_save_transcript(self):
        path = save_transcript()
        msg = "Saved to Desktop" if path else "Nothing to save"
        self.status_label.config(text=msg)
        self.root.after(3000, lambda: self.status_label.config(text=""))

    def on_save_audio(self):
        path = save_audio()
        msg = "Audio saved" if path else "No audio"
        self.status_label.config(text=msg)
        self.root.after(3000, lambda: self.status_label.config(text=""))

    def poll(self):
        try:
            ts, text_en, text_zh = subtitle_queue.get_nowait()
            if self.first_entry:
                self.text_box.config(state="normal")
                self.text_box.delete("1.0", "end")
                self.text_box.config(state="disabled")
                self.first_entry = False
            self._append(ts, text_en, text_zh)
        except queue.Empty:
            pass
        self.root.after(150, self.poll)


def main():
    global is_recording
    worker = threading.Thread(target=transcribe_loop, daemon=True)
    worker.start()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1),
    )
    stream.start()

    root = tk.Tk()
    SubtitleWindow(root)
    try:
        root.mainloop()
    finally:
        is_recording = False
        stream.stop()


if __name__ == "__main__":
    main()
