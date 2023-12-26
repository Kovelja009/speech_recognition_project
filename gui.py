import sounddevice as sd
import random
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, Canvas
import time
from sr_system import SR_System
import threading

class AudioRecorder:
    def __init__(self):
        self.fs = 22050
        self.channels = 1
        self.recording = np.array([])
        self.is_recording = False
        self.stream = None

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.recording = np.array([])
        self.stream = sd.InputStream(samplerate=self.fs, channels=self.channels, callback=self.audio_callback)
        self.stream.start()

    def stop_recording(self):
        self.is_recording = False
        if not self.stream: return
        self.stream.stop()
        self.stream.close()
        self.stream = None

    def audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.recording = np.append(self.recording, indata.copy())

    def play_recording(self):
        if self.recording.size > 0:
            sd.play(self.recording, self.fs)
            sd.wait()





class ActionPerformer:
    def __init__(self, canvas):
        self.canvas = canvas
        self.window = self.canvas.winfo_toplevel()
        self.shape_id = None

        self.action_map = {
            "kvadrat": self.draw_square,
            "krug": self.draw_circle,
            "trougao": self.draw_triangle,
            "oboji": self.color_shape,
            "izbrisi": self.delete_all,
            "animiraj": self.animate_shape
        }

    def perform_action(self, action):
        if action not in self.action_map: return
        self.action_map[action]()

    def draw_square(self, color="red"):
        self.delete_all()
        self.shape_id = self.canvas.create_rectangle(50, 50, 150, 150, fill=color)

    def draw_circle(self, color="red"):
        self.delete_all()
        self.shape_id = self.canvas.create_oval(50, 50, 150, 150, fill=color)

    def draw_triangle(self, color="red"):
        self.delete_all()
        self.shape_id = self.canvas.create_polygon(50, 150, 100, 50, 150, 150, fill=color)

    def delete_all(self):
        self.canvas.delete("all")

    def color_shape(self):
        if not self.shape_id: return
        color = self.get_random_color()
        self.canvas.itemconfig(self.shape_id, fill=color)
        

    def get_random_color(self):
        return random.choice(["green", "blue", "yellow", "orange", "purple", "pink"])
    
    def animate_shape(self):
        if not self.shape_id or not self.canvas.winfo_exists(): 
            return

        for _ in range(10):
            self.canvas.move(self.shape_id, 10, 0)
            self.window.update()
            time.sleep(0.02)

        for _ in range(10):
            self.canvas.move(self.shape_id, -20, 0)
            self.window.update()
            time.sleep(0.02)

        for _ in range(10):
            self.canvas.move(self.shape_id, 10, 0)
            self.window.update()
            time.sleep(0.02)




class GUI:
    def __init__(self, recorder):
        self.recorder = recorder
        self.sr_system = SR_System()
        

        self.window = tk.Tk()
        self.window.title("Audio Recorder")

        self.window.geometry("300x400")

        self.record_button = ttk.Button(self.window, text="Start Recording", command=self.toggle_record)
        self.record_button.pack(pady=10)

        self.play_button = ttk.Button(self.window, text="Play Recording", command=self.recorder.play_recording)
        self.play_button.pack(pady=10)

        self.execute_button = ttk.Button(self.window, text="Execute Command", command=self.execute_command)
        self.execute_button.pack(pady=10)

        self.new_word_button = ttk.Button(self.window, text="Train New Word", command=self.train_new_word)
        self.new_word_button.pack(pady=10)

        self.canvas = Canvas(self.window, width=200, height=200, bg="white")
        self.canvas.pack(pady=10)
        self.performer = ActionPerformer(self.canvas)

    def toggle_record(self):
        self.recorder.toggle_recording()
        btn_text = "Stop Recording" if self.recorder.is_recording else "Start Recording"
        self.record_button.config(text=btn_text)

    def execute_command(self):
        if self.recorder.recording.size > 0:
            thread = threading.Thread(target=self.run_prediction)
            thread.start()
        else:
            messagebox.showwarning("Warning", "No recording to analyze.")

    def run_prediction(self):
        result = self.sr_system.predict(self.recorder.recording)
        messagebox.showinfo("Recognition Result", f"Recognized command: {result}")
        self.performer.perform_action(result)

    def train_new_word(self):
        self.sr_system.add_new_sound(self.recorder.recording, 'animiraj')
        messagebox.showinfo("Training Result", "New word added to the vocabulary.")

    def run(self):
        self.window.mainloop()

recorder = AudioRecorder()
app = GUI(recorder)
app.run()