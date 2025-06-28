import tkinter as tk
from tkinter import Label, messagebox, ttk
import threading
import queue
import cv2
import pytesseract
import pyttsx3
from ultralytics import YOLO
from PIL import Image, ImageTk
import torch
import speech_recognition as sr
from datetime import datetime
import cohere
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)
import os
from dotenv import load_dotenv
import time
import tempfile
import wave

load_dotenv()

# Configuration Paths
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Model instances
model = YOLO("yolov8l.pt")  
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Scene Labels for CLIP
SCENE_CATEGORIES = [
    "office", "kitchen", "living room", "street", "garden", "shop", "bedroom", 
    "hall", "outdoor", "indoor", "classroom", "lab", "bathroom", "restaurant", 
    "library", "gym",
]

# Speech Recognition Configuration
SPEECH_CONFIG = {
    'energy_threshold': 400,  # Increased for better sensitivity
    'dynamic_energy_threshold': True,
    'dynamic_energy_adjustment_damping': 0.15,
    'dynamic_energy_ratio': 1.5,
    'pause_threshold': 0.8,
    'phrase_threshold': 0.3,
    'non_speaking_duration': 0.5,
    'timeout': 3.0,
    'phrase_time_limit': 8.0
}

# Speech Recognition
recognizer = sr.Recognizer()
recognizer.energy_threshold = SPEECH_CONFIG['energy_threshold']
recognizer.dynamic_energy_threshold = SPEECH_CONFIG['dynamic_energy_threshold']
recognizer.pause_threshold = SPEECH_CONFIG['pause_threshold']
recognizer.phrase_threshold = SPEECH_CONFIG['phrase_threshold']
recognizer.non_speaking_duration = SPEECH_CONFIG['non_speaking_duration']
microphone = sr.Microphone()

# Threshold
CONF_THRESHOLD = 0.85  # Increased to reduce false positives

# Global variables
current_frame = None
frame_lock = threading.Lock()
voice_queue = queue.Queue()
speech_queue = queue.Queue()  # New speech queue
is_listening = False
is_speaking = False
speech_engine = None
DEBUG_MODE = True
last_analysis = {
    "objects": [],
    "text": "",
    "caption": "",
    "scene": "",
    "timestamp": None
}

# Voice commands
VOICE_COMMANDS = {
    "describe": ["describe", "what do you see", "tell me what you see", "what's in front of me"],
    "objects": ["objects", "what objects", "detect objects", "find objects"],
    "read_text": ["read text", "what text", "text detection", "read", "read the text", "what does it say"],
    "scene": ["scene", "where am i", "location", "environment"],
    "help": ["help", "commands", "what can you do", "instructions"],
    "stop": ["stop", "quit", "exit", "goodbye", "shut up", "be quiet", "stop talking"],
    "detailed": ["detailed", "more details", "explain", "elaborate"],
    "person": ["person", "people", "who is there", "describe person", "person description"],
    "comprehensive": ["comprehensive", "full description", "complete description", "everything", "all details"],
    
    # GUI Button Voice Commands
    "start_analysis": ["start analysis", "analyze", "begin analysis", "start", "analyze scene", "scan", "detect", "look around"],
    "follow_up": ["follow up", "ask question", "question", "ask", "followup", "more", "additional", "another question"],
    "clear_history": ["clear history", "clear", "reset", "clear memory", "forget", "start over", "reset history"],
    "stop_all": ["stop all", "stop everything", "halt", "pause", "stop processing", "stop analysis"],
    
    # Additional Voice Controls
    "camera_on": ["turn on camera", "enable camera", "show camera", "camera on", "start camera"],
    "camera_off": ["turn off camera", "disable camera", "hide camera", "camera off", "stop camera"],
    "debug_mode": ["debug mode", "toggle debug", "show debug", "debug on", "debug off"],
    "volume_up": ["volume up", "louder", "increase volume", "speak louder"],
    "volume_down": ["volume down", "quieter", "decrease volume", "speak quieter"],
    "speed_up": ["speed up", "faster", "speak faster", "increase speed"],
    "slow_down": ["slow down", "slower", "speak slower", "decrease speed"],
    "repeat": ["repeat", "say again", "repeat that", "what did you say", "say it again"],
    "status": ["status", "what's happening", "current status", "what are you doing", "system status"],
    "save_image": ["save image", "capture image", "take photo", "save photo", "save picture"],
    "export_results": ["export results", "save results", "export analysis", "save analysis", "download results"],
    "voice_test": ["voice test", "test voice", "test microphone", "test recognition", "can you hear me"]
}

# Initialize Cohere client
cohere_api_key = os.getenv('COHERE_API_KEY')
if not cohere_api_key:
    print("⚠️  Warning: COHERE_API_KEY not found in environment variables.")
    print("   Please create a .env file with your Cohere API key.")
    print("   Get your free API key from: https://cohere.ai/")
    co = None
else:
    try:
        co = cohere.Client(cohere_api_key)
        print("✅ Cohere API client initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing Cohere client: {e}")
        co = None

class AuraSightGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🕶️ AuraSight - Advanced Voice & Vision Assistant")
        self.root.geometry("1000x700")
        self.root.configure(bg="#121212")
        
        # GUI variables
        self.recording = False
        self.image_display = None
        self.image_label = None
        self.status_label = None
        self.camera_feed_label = None
        self.cap = None
        self.camera_available = False
        
        # Voice interruption
        self.is_speaking = False
        self.speech_thread = None
        
        self.setup_gui()
        self.initialize_tts()
        
        # Try to start camera, but continue if it fails
        self.camera_available = self.start_camera_feed()
        if not self.camera_available:
            print("Continuing without camera functionality")
        
        self.start_voice_recognition()
        
    def setup_gui(self):
        # Colors & Style
        CYAN = "#00bcd4"
        CYAN_DARK = "#0097a7"
        WHITE = "#ffffff"
        FONT = ("Segoe UI", 12)
        BUTTON_WIDTH = 18
        BUTTON_HEIGHT = 2
        BUTTON_STYLE = {
            "font": FONT,
            "width": BUTTON_WIDTH,
            "height": BUTTON_HEIGHT,
            "bd": 0,
            "relief": "flat",
            "bg": CYAN,
            "fg": WHITE,
            "activebackground": CYAN_DARK,
            "activeforeground": WHITE
        }

        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#121212")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Title
        title_label = tk.Label(main_frame, text="AuraSight - Advanced Voice Assistant", 
                              font=("Segoe UI", 16, "bold"), bg="#121212", fg=CYAN)
        title_label.pack(pady=(0, 20))
        
        # Three columns layout
        content_frame = tk.Frame(main_frame, bg="#121212")
        content_frame.pack(expand=True, fill="both")
        
        # Left: Camera Feed
        camera_frame = tk.Frame(content_frame, bg="#121212", width=400)
        camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        camera_title = tk.Label(camera_frame, text="Live Camera Feed", 
                               font=("Segoe UI", 12, "bold"), bg="#121212", fg=WHITE)
        camera_title.pack(pady=(0, 10))
        
        self.camera_feed_label = tk.Label(camera_frame, bg="#1e1e1e", relief="solid", bd=1)
        self.camera_feed_label.pack(expand=True, fill="both")
        
        # Center: Controls
        control_frame = tk.Frame(content_frame, bg="#121212", width=200)
        control_frame.pack(side="left", fill="y", padx=10)
        
        control_title = tk.Label(control_frame, text="Controls", 
                                font=("Segoe UI", 12, "bold"), bg="#121212", fg=WHITE)
        control_title.pack(pady=(0, 10))
        
        # Buttons
        self.start_btn = tk.Button(control_frame, text="Start Analysis", 
                                  command=self.handle_start, **BUTTON_STYLE)
        self.start_btn.pack(pady=6)
        
        self.stop_btn = tk.Button(control_frame, text="Stop", 
                                 command=self.handle_stop, **BUTTON_STYLE)
        self.stop_btn.pack(pady=6)
        
        self.followup_btn = tk.Button(control_frame, text="Follow Up", 
                                     command=self.handle_followup, **BUTTON_STYLE)
        self.followup_btn.pack(pady=6)
        
        self.clear_btn = tk.Button(control_frame, text="Clear History", 
                                  command=self.clear_history, **BUTTON_STYLE)
        self.clear_btn.pack(pady=6)
        
        # Voice commands info
        voice_info = tk.Label(control_frame, text="Voice Commands:\n• 'Start Analysis' - Begin analysis\n• 'Follow Up' - Ask questions\n• 'Clear History' - Reset memory\n• 'Stop All' - Stop everything\n• 'Volume Up/Down' - Adjust speech\n• 'Speed Up/Down' - Adjust speed\n• 'Repeat' - Hear again\n• 'Status' - System status\n• 'Save Image' - Capture photo\n• 'Export Results' - Save data\n• 'Help' - Show all commands", 
                             font=("Segoe UI", 9), bg="#121212", fg="#cccccc", 
                             anchor="w", justify="left", wraplength=180)
        voice_info.pack(pady=10)
        
        # Voice recognition status
        self.voice_status_label = tk.Label(control_frame, text="🎤 Listening...", 
                                          font=("Segoe UI", 10, "bold"), bg="#121212", fg="#00ff00")
        self.voice_status_label.pack(pady=5)
        
        # Right: Status and Analysis
        status_frame = tk.Frame(content_frame, bg="#121212", width=300)
        status_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        status_title = tk.Label(status_frame, text="Status & Analysis", 
                               font=("Segoe UI", 12, "bold"), bg="#121212", fg=WHITE)
        status_title.pack(pady=(0, 10))
        
        self.status_label = tk.Label(status_frame, text="Ready", 
                                    font=("Segoe UI", 11), bg="#121212", fg="#dddddd", 
                                    anchor="w", wraplength=280)
        self.status_label.pack(pady=6)
        
        # Analysis results
        self.analysis_text = tk.Text(status_frame, height=15, width=35, 
                                    bg="#1e1e1e", fg="#ffffff", font=("Consolas", 9),
                                    relief="solid", bd=1, wrap="word")
        self.analysis_text.pack(expand=True, fill="both", pady=(10, 0))
        
        # Scrollbar for analysis text
        scrollbar = tk.Scrollbar(status_frame, orient="vertical", command=self.analysis_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.analysis_text.configure(yscrollcommand=scrollbar.set)

    def initialize_tts(self):
        """Initialize text-to-speech engine."""
        global speech_engine
        try:
            speech_engine = pyttsx3.init('sapi5')
            speech_engine.setProperty('rate', 140)
            speech_engine.setProperty('volume', 0.9)
            
            voices = speech_engine.getProperty('voices')
            if voices:
                selected_voice = None
                for voice in voices:
                    voice_name = voice.name.lower()
                    if any(lang in voice_name for lang in ['english', 'en', 'us', 'uk']):
                        selected_voice = voice.id
                        break
                
                if not selected_voice:
                    selected_voice = voices[0].id
                
                speech_engine.setProperty('voice', selected_voice)
                print(f"TTS initialized with voice: {selected_voice}")
            
        except Exception as e:
            print(f"TTS initialization error: {e}")
            messagebox.showerror("TTS Error", f"Failed to initialize text-to-speech: {e}")
    
    def start_camera_feed(self):
        """Start the camera feed display."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Warning: Could not open camera. Camera features will be disabled.")
                self.status_label.config(text="Camera not available")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            def update_camera():
                global current_frame
                try:
                    if self.cap and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret:
                            with frame_lock:
                                current_frame = frame.copy()
                            
                            # Update GUI
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(rgb_frame)
                            img_resized = pil_image.resize((380, 280), Image.Resampling.LANCZOS)
                            img_tk = ImageTk.PhotoImage(img_resized)
                            
                            self.camera_feed_label.configure(image=img_tk)
                            self.camera_feed_label.image = img_tk
                        else:
                            print("Failed to read frame from camera")
                    else:
                        print("Camera not available")
                except Exception as e:
                    print(f"Camera update error: {e}")
                
                # Schedule next update only if application is still running
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(50, update_camera)
            
            # Start the camera update loop
            self.root.after(100, update_camera)
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            self.status_label.config(text="Camera error")
            return False
    
    def start_voice_recognition(self):
        """Start voice recognition in background using STT."""
        global is_listening
        
        def voice_thread():
            global is_listening
            is_listening = True
            
            # Initialize microphone with better settings
            try:
                with microphone as source:
                    print("Calibrating microphone for ambient noise...")
                    recognizer.adjust_for_ambient_noise(source, duration=2.0)
                    print(f"Microphone calibrated. Energy threshold: {recognizer.energy_threshold}")
            except Exception as e:
                print(f"Microphone calibration error: {e}")
            
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            while is_listening:
                try:
                    with microphone as source:
                        # Adjust for ambient noise periodically
                        if consecutive_errors > 2:
                            recognizer.adjust_for_ambient_noise(source, duration=1.0)
                            consecutive_errors = 0
                        
                        # Update status
                        self.root.after(0, lambda: self.voice_status_label.config(
                            text="🎤 Listening...", fg="#00ff00"))
                        
                        audio = recognizer.listen(
                            source, 
                            timeout=SPEECH_CONFIG['timeout'],
                            phrase_time_limit=SPEECH_CONFIG['phrase_time_limit']
                        )
                    
                    try:
                        # Use Google's speech recognition
                        command = recognizer.recognize_google(
                            audio, 
                            language='en-US',
                            show_all=False
                        ).lower()
                        
                        print(f"Heard: '{command}'")
                        consecutive_errors = 0  # Reset error counter on success
                        
                        # Update voice status
                        self.root.after(0, lambda: self.voice_status_label.config(
                            text="🎤 Heard: " + command[:20] + "...", fg="#00ff00"))
                        
                        voice_queue.put(command)
                        
                        # Check for stop command immediately
                        if any(keyword in command for keyword in VOICE_COMMANDS["stop"]):
                            self.interrupt_speech()
                        
                    except sr.UnknownValueError:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print("Multiple consecutive recognition failures. Recalibrating...")
                            consecutive_errors = 0
                            self.root.after(0, lambda: self.voice_status_label.config(
                                text="🎤 Recalibrating...", fg="#ffff00"))
                        else:
                            self.root.after(0, lambda: self.voice_status_label.config(
                                text="🎤 Listening...", fg="#00ff00"))
                        continue
                        
                    except sr.RequestError as e:
                        print(f"Speech recognition service error: {e}")
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print("Speech service unavailable. Waiting before retry...")
                            self.root.after(0, lambda: self.voice_status_label.config(
                                text="🎤 Service Error", fg="#ff0000"))
                            time.sleep(5)
                            consecutive_errors = 0
                        else:
                            self.root.after(0, lambda: self.voice_status_label.config(
                                text="🎤 Listening...", fg="#00ff00"))
                        continue
                        
                except sr.WaitTimeoutError:
                    consecutive_errors += 1
                    self.root.after(0, lambda: self.voice_status_label.config(
                        text="🎤 Listening...", fg="#00ff00"))
                    continue
                except Exception as e:
                    print(f"Voice recognition error: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many errors. Restarting voice recognition...")
                        self.root.after(0, lambda: self.voice_status_label.config(
                            text="🎤 Restarting...", fg="#ff0000"))
                        time.sleep(2)
                        consecutive_errors = 0
                    else:
                        self.root.after(0, lambda: self.voice_status_label.config(
                            text="🎤 Listening...", fg="#00ff00"))
                    continue
        
        threading.Thread(target=voice_thread, daemon=True).start()
    
    def interrupt_speech(self):
        """Interrupt current speech."""
        global is_speaking
        if is_speaking and speech_engine:
            try:
                speech_engine.stop()
                # Wait a moment for the stop to take effect
                time.sleep(0.1)
            except Exception as e:
                print(f"Error stopping speech: {e}")
            finally:
                is_speaking = False
                self.status_label.config(text="Speech interrupted")
                print("Speech interrupted by user")
        
        # Clear any pending speech requests
        try:
            while not speech_queue.empty():
                speech_queue.get_nowait()
                speech_queue.task_done()
        except queue.Empty:
            pass
    
    def speak(self, text):
        """Speak text with interruption capability."""
        if not text or not speech_engine:
            return
        
        # Add to speech queue instead of speaking directly
        speech_queue.put(text)
        
        # Start speech processor if not already running
        if not hasattr(self, 'speech_processor_running'):
            self.speech_processor_running = False
            self.start_speech_processor()
    
    def start_speech_processor(self):
        """Start the speech processing thread."""
        if self.speech_processor_running:
            return
        
        self.speech_processor_running = True
        
        def speech_processor():
            global is_speaking
            while self.speech_processor_running:
                try:
                    # Get next speech request from queue
                    text = speech_queue.get(timeout=0.1)
                    
                    # If already speaking, stop current speech
                    if is_speaking:
                        self.interrupt_speech()
                        time.sleep(0.2)  # Wait for clean stop
                    
                    # Speak the text
                    try:
                        is_speaking = True
                        self.status_label.config(text="Speaking...")
                        
                        # Create a new engine instance for this speech
                        temp_engine = pyttsx3.init('sapi5')
                        temp_engine.setProperty('rate', speech_engine.getProperty('rate'))
                        temp_engine.setProperty('volume', speech_engine.getProperty('volume'))
                        temp_engine.setProperty('voice', speech_engine.getProperty('voice'))
                        
                        temp_engine.say(text)
                        temp_engine.runAndWait()
                        temp_engine.stop()
                        
                    except Exception as e:
                        print(f"Speech error: {e}")
                        # Fallback to main engine
                        try:
                            speech_engine.say(text)
                            speech_engine.runAndWait()
                        except Exception as e2:
                            print(f"Fallback speech error: {e2}")
                    finally:
                        is_speaking = False
                        self.status_label.config(text="Ready")
                    
                    # Mark task as done
                    speech_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Speech processor error: {e}")
                    continue
        
        threading.Thread(target=speech_processor, daemon=True).start()
    
    def handle_start(self):
        """Handle start button click."""
        def task():
            self.recording = True
            self.status_label.config(text="Analyzing current view...")
            
            try:
                # Analyze current frame
                self.analyze_current_frame(include_text=False)
                
                # Get full description
                response = self.get_full_description()
                
                # Update analysis display
                self.update_analysis_display()
                
                # Speak response
                self.speak(response)
                
            except Exception as e:
                self.status_label.config(text="Analysis failed")
                messagebox.showerror("Error", f"Analysis failed: {e}")
            
            self.recording = False
        
        threading.Thread(target=task, daemon=True).start()
    
    def handle_stop(self):
        """Handle stop button click."""
        self.recording = False
        self.interrupt_speech()
        self.status_label.config(text="Stopped")
    
    def handle_followup(self):
        """Handle followup button click."""
        def task():
            self.recording = True
            self.status_label.config(text="Listening for followup question...")
            
            try:
                # Wait for voice input
                with microphone as source:
                    audio = recognizer.listen(source, timeout=5.0, phrase_time_limit=5.0)
                
                question = recognizer.recognize_google(audio, language='en-US')
                self.status_label.config(text="Processing followup...")
                
                # Process followup
                response = self.handle_question(question)
                self.speak(response)
                
            except Exception as e:
                self.status_label.config(text="Followup failed")
                messagebox.showerror("Error", f"Followup failed: {e}")
            
            self.recording = False
        
        threading.Thread(target=task, daemon=True).start()
    
    def clear_history(self):
        """Clear conversation history."""
        def task():
            global last_analysis
            last_analysis = {
                "objects": [],
                "text": "",
                "caption": "",
                "scene": "",
                "timestamp": None
            }
            self.analysis_text.delete(1.0, tk.END)
            self.status_label.config(text="History cleared")
        
        threading.Thread(target=task, daemon=True).start()
    
    def analyze_current_frame(self, include_text=False):
        """Analyze the current frame and update global results."""
        global current_frame, last_analysis
        
        # Check if camera is available
        if not self.camera_available:
            last_analysis = {
                "objects": [],
                "text": "",
                "caption": "",
                "scene": "",
                "timestamp": datetime.now()
            }
            return
        
        with frame_lock:
            if current_frame is None:
                return
            frame = current_frame.copy()
        
        if DEBUG_MODE:
            print("\n=== FRAME ANALYSIS ===")
        
        # Perform analyses (text only when requested)
        objects, results = self.detect_objects(frame, CONF_THRESHOLD)
        text = self.extract_text(frame) if include_text else ""
        caption = self.get_blip_caption(frame)
        scene = self.get_clip_classification(frame)
        
        if DEBUG_MODE:
            print(f"Objects detected: {objects}")
            if include_text:
                print(f"Text extracted: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            print(f"BLIP caption: '{caption}'")
            print(f"CLIP scene: '{scene}'")
            print("=====================\n")
        
        # Update last analysis
        last_analysis = {
            "objects": objects,
            "text": text,
            "caption": caption,
            "scene": scene,
            "timestamp": datetime.now()
        }
    
    def detect_objects(self, img, conf_thresh=0.3):
        """Detect objects using YOLO and filter by confidence."""
        results = model(img)
        labels = []
        for result in results:
            for box in result.boxes:
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                if confidence >= conf_thresh:
                    if self.should_include_object(class_name, confidence):
                        labels.append(class_name)
        return list(set(labels)), results
    
    def should_include_object(self, class_name, confidence):
        """Filter out objects that are likely false positives."""
        # Common false positive objects that need higher confidence
        high_confidence_objects = ['laptop', 'computer', 'tv', 'monitor', 'cell phone', 'remote', 'keyboard', 'mouse']
        
        # Very strict filtering for electronic devices
        strict_confidence_objects = ['laptop', 'computer', 'tv', 'monitor']
        
        if class_name in strict_confidence_objects:
            # Require very high confidence for these objects
            return confidence >= 0.90
        elif class_name in high_confidence_objects:
            # Require higher confidence for these objects
            return confidence >= 0.80
        else:
            # Normal confidence threshold for other objects
            return confidence >= CONF_THRESHOLD
    
    def extract_text(self, img):
        """Perform OCR using Tesseract with enhanced preprocessing."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = []
            
            # Multiple preprocessing attempts
            basic_text = pytesseract.image_to_string(gray, config='--psm 6')
            if basic_text.strip():
                results.append(basic_text.strip())
            
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
            enhanced_text = pytesseract.image_to_string(enhanced, config='--psm 6')
            if enhanced_text.strip():
                results.append(enhanced_text.strip())
            
            denoised = cv2.fastNlMeansDenoising(gray)
            denoised_text = pytesseract.image_to_string(denoised, config='--psm 6')
            if denoised_text.strip():
                results.append(denoised_text.strip())
            
            all_text = ' '.join(results)
            cleaned_text = self.clean_and_validate_text(all_text)
            
            return cleaned_text
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def clean_and_validate_text(self, text):
        """Clean and validate extracted text."""
        if not text:
            return ""
        
        import re
        
        text = re.sub(r'\b[a-zA-Z]\b', '', text)
        
        lines = text.split('\n')
        valid_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 2:
                if re.search(r'[a-zA-Z]{2,}', line) or re.search(r'\d{2,}', line):
                    valid_lines.append(line)
        
        cleaned_text = ' '.join(valid_lines)
        cleaned_text = ' '.join(cleaned_text.split())
        
        if len(cleaned_text) > 3:
            return cleaned_text
        else:
            return ""
    
    def get_blip_caption(self, frame):
        """Get scene caption using BLIP."""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = blip_processor(pil_image, return_tensors="pt")
            with torch.no_grad():
                output = blip_model.generate(**inputs, max_length=30)
            caption = blip_processor.decode(output[0], skip_special_tokens=True)
            return caption
        except:
            return ""
    
    def get_clip_classification(self, frame, labels=SCENE_CATEGORIES):
        """Classify scene using CLIP."""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = clip_processor(text=labels, images=pil_image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                top_prob, top_label_id = probs.max(dim=1)
            return labels[top_label_id]
        except:
            return ""
    
    def get_full_description(self):
        """Get complete scene description."""
        global last_analysis
        
        parts = []
        if last_analysis["objects"]:
            parts.append(f"I can see {', '.join(last_analysis['objects'])}")
        if last_analysis["text"] and last_analysis["text"].strip():
            parts.append(f"I can read this text: {last_analysis['text'][:100]}...")
        if last_analysis["caption"]:
            parts.append(f"It looks like {last_analysis['caption']}")
        if last_analysis["scene"]:
            parts.append(f"You're in what appears to be a {last_analysis['scene']}")
        
        if parts:
            return ". ".join(parts) + "."
        else:
            return "I'm not really seeing anything clear at the moment. Maybe try moving the camera around a bit?"
    
    def handle_question(self, question):
        """Handle natural language questions."""
        global last_analysis
        
        # Check if it's a visual analysis question
        visual_keywords = [
            "see", "look", "detect", "find", "spot", "notice", "observe",
            "what", "where", "how many", "describe", "tell me about",
            "person", "people", "object", "text", "scene", "room", "place"
        ]
        
        is_visual_question = any(keyword in question.lower() for keyword in visual_keywords)
        
        if is_visual_question:
            return self.handle_visual_question(question)
        else:
            return self.handle_general_question(question)
    
    def handle_visual_question(self, question):
        """Handle questions specifically about visual analysis."""
        global last_analysis
        
        if last_analysis["timestamp"] is None or (datetime.now() - last_analysis["timestamp"]).seconds > 3:
            self.analyze_current_frame(include_text=False)
        
        context = {
            "objects": last_analysis["objects"],
            "text": last_analysis["text"],
            "caption": last_analysis["caption"],
            "scene": last_analysis["scene"]
        }
        
        return self.get_refined_response(question, context, "detailed")
    
    def handle_general_question(self, question):
        """Handle general questions."""
        if not co:
            return "I'm sorry, but I can't process general questions right now because the AI service is not configured. Please set up your Cohere API key in the .env file."
        
        try:
            prompt = (
                f"Question: {question}\n\n"
                f"Instructions: You are Aurasight, a helpful voice assistant for visually impaired users. "
                f"Answer the question naturally and helpfully. Be friendly and conversational. "
                f"Keep responses concise and clear.\n\n"
                f"Response:"
            )
            
            response = co.generate(
                model='command',
                prompt=prompt,
                max_tokens=120,
                temperature=0.7,
                k=0,
                p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            answer = response.generations[0].text.strip()
            if not answer or len(answer) < 5:
                return "I'm not sure how to answer that. Could you rephrase your question?"
            
            return answer
            
        except Exception as e:
            print(f"General question error: {e}")
            return "I'm having trouble processing that question right now. Could you try asking something else?"
    
    def get_refined_response(self, user_query, context, detail_level="detailed"):
        """Generate refined response using Cohere."""
        if not co:
            return self.get_fallback_response(context, user_query)
        
        try:
            objects = context.get('objects', [])
            text = context.get('text', '')
            caption = context.get('caption', '')
            scene = context.get('scene', '')
            
            has_real_data = len(objects) > 0 or len(text.strip()) > 5 or len(caption.strip()) > 5
            
            if detail_level == "detailed":
                instruction = """IMPORTANT: Only describe what is actually detected. Do NOT make up or imagine details. Be factual and accurate. If you're not sure about something, say so. Include only:
                - Objects that are actually detected (if any)
                - Text that is actually readable (if any)
                - Scene description if clear and specific
                - Location if identifiable
                Use casual, friendly language but be honest about what you can and cannot see. Do not invent details."""
                max_tokens = 100
            else:
                instruction = "Respond naturally and helpfully, but only mention what is actually detected. Be honest about limitations."
                max_tokens = 60

            prompt = (
                f"Here's what I can actually detect:\n"
                f"Objects detected: {objects if objects else 'None detected'}\n"
                f"Text detected: '{text if text else 'No readable text'}'"
            )
            
            if caption and len(caption.strip()) > 5:
                prompt += f"\nScene description: {caption}"
            if scene:
                prompt += f"\nLocation: {scene}"
                
            prompt += f"\n\nQuestion: {user_query}\n\n"
            prompt += f"Instructions: {instruction}\n\n"
            prompt += "Factual Response (only mention what is actually detected):"
            
            temperature = 0.1 if has_real_data else 0.05
            
            response = co.generate(
                model='command',
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                k=0,
                p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            answer = response.generations[0].text.strip()
            
            if not answer or len(answer) < 5:
                return self.get_fallback_response(context, user_query)
            
            return answer
        except Exception as e:
            print(f"Cohere API Error: {e}")
            return self.get_fallback_response(context, user_query)
    
    def get_fallback_response(self, context, user_query):
        """Provide a fallback response when AI generation fails."""
        objects = context.get('objects', [])
        text = context.get('text', '')
        caption = context.get('caption', '')
        scene = context.get('scene', '')
        
        parts = []
        
        if objects:
            parts.append(f"I can detect {', '.join(objects)}")
        
        if text and len(text.strip()) > 3:
            parts.append(f"I can read some text: {text[:100]}...")
        
        if caption and len(caption.strip()) > 5:
            parts.append(f"The scene looks like {caption}")
        
        if scene:
            parts.append(f"You appear to be in a {scene}")
        
        if parts:
            return ". ".join(parts) + "."
        else:
            return "I'm not detecting anything specific in the current view. Try moving the camera or adjusting the lighting."
    
    def update_analysis_display(self):
        """Update the analysis display in the GUI."""
        global last_analysis
        
        self.analysis_text.delete(1.0, tk.END)
        
        if last_analysis["timestamp"]:
            timestamp = last_analysis["timestamp"].strftime("%H:%M:%S")
            self.analysis_text.insert(tk.END, f"Analysis Time: {timestamp}\n")
            self.analysis_text.insert(tk.END, "=" * 40 + "\n\n")
            
            if last_analysis["objects"]:
                self.analysis_text.insert(tk.END, f"Objects Detected:\n")
                for obj in last_analysis["objects"]:
                    self.analysis_text.insert(tk.END, f"• {obj}\n")
                self.analysis_text.insert(tk.END, "\n")
            
            if last_analysis["text"] and last_analysis["text"].strip():
                self.analysis_text.insert(tk.END, f"Text Extracted:\n{last_analysis['text']}\n\n")
            
            if last_analysis["caption"]:
                self.analysis_text.insert(tk.END, f"Scene Description:\n{last_analysis['caption']}\n\n")
            
            if last_analysis["scene"]:
                self.analysis_text.insert(tk.END, f"Location: {last_analysis['scene']}\n")
        else:
            self.analysis_text.insert(tk.END, "No analysis data available.\nClick 'Start Analysis' to begin.")
    
    def process_voice_commands(self):
        """Process voice commands from the queue."""
        try:
            while not voice_queue.empty():
                command = voice_queue.get_nowait()
                
                # Check for command matches
                for cmd_type, keywords in VOICE_COMMANDS.items():
                    if any(keyword in command for keyword in keywords):
                        response = self.execute_command(cmd_type, command)
                        if response == "exit":
                            self.root.quit()
                            return
                        if response:
                            self.speak(response)
                        break
                else:
                    # If no direct command match, treat as a question
                    response = self.handle_question(command)
                    if response:
                        self.speak(response)
                        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_voice_commands)
    
    def execute_command(self, cmd_type, command):
        """Execute specific voice commands."""
        global last_analysis, DEBUG_MODE, speech_engine
        
        # Handle stop commands first
        if cmd_type == "stop" or cmd_type == "stop_all":
            self.interrupt_speech()
            self.recording = False
            self.status_label.config(text="All operations stopped")
            return "All operations stopped"
        
        elif cmd_type == "help":
            return """I'm Aurasight, your voice-controlled assistant! I can help you in many ways:

VISUAL ANALYSIS:
Say 'describe' or 'what do you see' for a full description
Say 'objects' to hear what objects I can spot
Say 'read text' to read any text in the current view
Say 'scene' or 'where am I' for location information
Say 'detailed' for a more comprehensive description
Say 'person' or 'who is there' for detailed person descriptions
Say 'comprehensive' for complete scene analysis

GUI CONTROLS (Voice Commands):
Say 'start analysis' or 'analyze' to begin analysis
Say 'follow up' or 'ask question' for additional questions
Say 'clear history' or 'reset' to clear memory
Say 'stop all' or 'halt' to stop everything

SYSTEM CONTROLS:
Say 'camera on/off' to control camera display
Say 'volume up/down' to adjust speech volume
Say 'speed up/slow down' to adjust speech speed
Say 'repeat' to hear the last response again
Say 'status' to check current system status
Say 'save image' to capture current view
Say 'export results' to save analysis data
Say 'voice test' to test voice recognition

GENERAL CONVERSATION:
You can ask me ANY question! I'm not limited to just visual analysis.
Ask me about the weather, time, general knowledge, or just chat with me.

Say 'stop' to interrupt my speech at any time"""
        
        # GUI Button Voice Commands
        elif cmd_type == "start_analysis":
            self.handle_start()
            return "Starting analysis..."
        
        elif cmd_type == "follow_up":
            self.handle_followup()
            return "Listening for your question..."
        
        elif cmd_type == "clear_history":
            self.clear_history()
            return "History cleared"
        
        # System Control Commands
        elif cmd_type == "camera_on":
            if self.cap and not self.cap.isOpened():
                self.start_camera_feed()
            self.status_label.config(text="Camera enabled")
            return "Camera is now enabled"
        
        elif cmd_type == "camera_off":
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.status_label.config(text="Camera disabled")
            return "Camera is now disabled"
        
        elif cmd_type == "debug_mode":
            global DEBUG_MODE
            DEBUG_MODE = not DEBUG_MODE
            status = "enabled" if DEBUG_MODE else "disabled"
            self.status_label.config(text=f"Debug mode {status}")
            return f"Debug mode {status}"
        
        elif cmd_type == "volume_up":
            if speech_engine:
                current_volume = speech_engine.getProperty('volume')
                new_volume = min(1.0, current_volume + 0.1)
                speech_engine.setProperty('volume', new_volume)
                self.status_label.config(text=f"Volume: {new_volume:.1f}")
                return f"Volume increased to {new_volume:.1f}"
            return "Volume control not available"
        
        elif cmd_type == "volume_down":
            if speech_engine:
                current_volume = speech_engine.getProperty('volume')
                new_volume = max(0.0, current_volume - 0.1)
                speech_engine.setProperty('volume', new_volume)
                self.status_label.config(text=f"Volume: {new_volume:.1f}")
                return f"Volume decreased to {new_volume:.1f}"
            return "Volume control not available"
        
        elif cmd_type == "speed_up":
            if speech_engine:
                current_rate = speech_engine.getProperty('rate')
                new_rate = min(300, current_rate + 20)
                speech_engine.setProperty('rate', new_rate)
                self.status_label.config(text=f"Speed: {new_rate}")
                return f"Speech speed increased to {new_rate}"
            return "Speed control not available"
        
        elif cmd_type == "slow_down":
            if speech_engine:
                current_rate = speech_engine.getProperty('rate')
                new_rate = max(50, current_rate - 20)
                speech_engine.setProperty('rate', new_rate)
                self.status_label.config(text=f"Speed: {new_rate}")
                return f"Speech speed decreased to {new_rate}"
            return "Speed control not available"
        
        elif cmd_type == "repeat":
            # Get the last response from analysis
            if last_analysis["timestamp"]:
                response = self.get_full_description()
                return f"Repeating: {response}"
            else:
                return "No previous analysis to repeat"
        
        elif cmd_type == "status":
            status_parts = []
            status_parts.append(f"Camera: {'On' if self.cap and self.cap.isOpened() else 'Off'}")
            status_parts.append(f"Recording: {'Yes' if self.recording else 'No'}")
            status_parts.append(f"Speaking: {'Yes' if is_speaking else 'No'}")
            status_parts.append(f"Debug: {'On' if DEBUG_MODE else 'Off'}")
            if speech_engine:
                status_parts.append(f"Volume: {speech_engine.getProperty('volume'):.1f}")
                status_parts.append(f"Speed: {speech_engine.getProperty('rate')}")
            
            return ". ".join(status_parts)
        
        elif cmd_type == "save_image":
            try:
                global current_frame
                with frame_lock:
                    if current_frame is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"captured_image_{timestamp}.jpg"
                        cv2.imwrite(filename, current_frame)
                        self.status_label.config(text=f"Image saved: {filename}")
                        return f"Image saved as {filename}"
                    else:
                        return "No image available to save"
            except Exception as e:
                return f"Failed to save image: {e}"
        
        elif cmd_type == "export_results":
            try:
                if last_analysis["timestamp"]:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"analysis_results_{timestamp}.txt"
                    
                    with open(filename, 'w') as f:
                        f.write(f"AuraSight Analysis Results - {timestamp}\n")
                        f.write("=" * 50 + "\n\n")
                        
                        if last_analysis["objects"]:
                            f.write("Objects Detected:\n")
                            for obj in last_analysis["objects"]:
                                f.write(f"• {obj}\n")
                            f.write("\n")
                        
                        if last_analysis["text"]:
                            f.write(f"Text Extracted:\n{last_analysis['text']}\n\n")
                        
                        if last_analysis["caption"]:
                            f.write(f"Scene Description:\n{last_analysis['caption']}\n\n")
                        
                        if last_analysis["scene"]:
                            f.write(f"Location: {last_analysis['scene']}\n")
                    
                    self.status_label.config(text=f"Results exported: {filename}")
                    return f"Analysis results exported to {filename}"
                else:
                    return "No analysis results to export"
            except Exception as e:
                return f"Failed to export results: {e}"
        
        elif cmd_type == "voice_test":
            return "Voice recognition is working! I can hear you clearly. Try saying 'start analysis' or 'help' to see what I can do."
        
        # For all other commands that need visual analysis
        if last_analysis["timestamp"] is None or (datetime.now() - last_analysis["timestamp"]).seconds > 3:
            self.speak("Let me take a look at what's around you...")
            # Include text extraction only for commands that need it
            include_text = cmd_type in ["read_text"]
            self.analyze_current_frame(include_text=include_text)
        
        if cmd_type == "describe":
            return self.get_full_description()
        
        elif cmd_type == "objects":
            if last_analysis["objects"]:
                return f"I can spot {', '.join(last_analysis['objects'])} in front of you."
            else:
                return "I'm not picking up any objects right now. Try pointing the camera at something specific."
        
        elif cmd_type == "read_text":
            if last_analysis["text"]:
                return f"I can read this text: {last_analysis['text'][:200]}..."
            else:
                return "I don't see any readable text in the current view. Try holding something with text closer to the camera."
        
        elif cmd_type == "scene":
            parts = []
            if last_analysis["caption"]:
                parts.append(f"It looks like {last_analysis['caption']}")
            if last_analysis["scene"]:
                parts.append(f"You're in a {last_analysis['scene']}")
            
            if parts:
                return ". ".join(parts) + "."
            else:
                return "I'm having trouble figuring out what kind of place this is. Maybe try a different angle?"
        
        elif cmd_type == "detailed":
            detailed_parts = []
            if last_analysis["objects"]:
                obj_count = len(last_analysis["objects"])
                detailed_parts.append(f"I've found {obj_count} different things: {', '.join(last_analysis['objects'])}")
            if last_analysis["caption"]:
                detailed_parts.append(f"The overall scene looks like {last_analysis['caption']}")
            if last_analysis["scene"]:
                detailed_parts.append(f"Based on the surroundings, you seem to be in a {last_analysis['scene']}")
            
            if detailed_parts:
                return ". ".join(detailed_parts) + "."
            else:
                return "I'm not getting a clear picture of what's around you. Maybe try adjusting the lighting or camera position?"
        
        elif cmd_type == "person":
            if not last_analysis["objects"]:
                return "I don't see any people in the current view."
            
            people_detected = any(obj in ["person", "people"] for obj in last_analysis["objects"])
            if not people_detected:
                return "I can't spot any specific people in the current view."
            
            context = {
                "objects": last_analysis["objects"],
                "text": last_analysis["text"],
                "caption": last_analysis["caption"],
                "scene": last_analysis["scene"]
            }
            
            return self.get_refined_response("Describe the people in detail", context, "person")
        
        elif cmd_type == "comprehensive":
            context = {
                "objects": last_analysis["objects"],
                "text": last_analysis["text"],
                "caption": last_analysis["caption"],
                "scene": last_analysis["scene"]
            }
            
            return self.get_refined_response("Tell me everything you can see in detail", context, "detailed")
        
        return "I didn't quite catch that. Try saying 'help' to see what I can do for you."
    
    def run(self):
        """Start the GUI application."""
        try:
            # Start voice command processing
            self.process_voice_commands()
            
            # Welcome message
            self.speak("Hi there! I'm Aurasight, your advanced voice-controlled visual assistant. I have both GUI controls and voice commands. Just say 'help' if you want to know what I can do for you.")
            
            # Start the GUI
            self.root.mainloop()
            
        except Exception as e:
            print(f"Application error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            global is_listening
            is_listening = False
            
            # Stop speech processor
            if hasattr(self, 'speech_processor_running'):
                self.speech_processor_running = False
            
            # Clear speech queue
            try:
                while not speech_queue.empty():
                    speech_queue.get_nowait()
                    speech_queue.task_done()
            except queue.Empty:
                pass
            
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

    def on_closing(self):
        """Handle window close event."""
        try:
            # Stop all operations
            global is_listening
            is_listening = False
            
            # Stop speech processor
            if hasattr(self, 'speech_processor_running'):
                self.speech_processor_running = False
            
            # Clear speech queue
            try:
                while not speech_queue.empty():
                    speech_queue.get_nowait()
                    speech_queue.task_done()
            except queue.Empty:
                pass
            
            # Release camera
            if self.cap:
                self.cap.release()
            
            # Close GUI
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AuraSightGUI()
    app.run() 