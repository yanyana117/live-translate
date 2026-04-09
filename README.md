# live-translate
Real-time English to Chinese speech translation using Whisper and Google Translate

# Live English to Chinese Speech Translator

Real-time speech recognition and translation tool built for lecture use.
Runs entirely on Apple Silicon (M4 Pro) with Metal GPU acceleration.

## How it works
- Captures microphone audio in real-time using sounddevice
- Accumulates a 6-second sliding window
- Transcribes English speech using mlx-whisper (Whisper large-v3-turbo, local, no internet required)
- Detects sentence boundaries before translating
- Translates to Chinese using Google Translate (deep-translator)
- Displays output in a floating tkinter window (always on top)

## Features
- Fully local speech recognition, no API cost
- Adjustable font size
- Save transcript and audio recording
- Metal GPU acceleration via mlx-whisper

## Stack
Python, mlx-whisper, sounddevice, deep-translator, tkinter
