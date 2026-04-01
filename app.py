import os
import time
import json
import base64
import threading
import logging
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import torch
import numpy as np
from gtts import gTTS
import ollama
import requests
from PIL import Image

# ---------------------------
# Flask App Initialization
# ---------------------------
# The main application runs on your computer and controls the Raspberry Pi.
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Configuration & Directories
# ---------------------------
RASPBERRY_PI_IP = "192.168.83.128"  # IMPORTANT: Change to your Pi's IP address
TTS_DIR = "static/tts"
CONVERSATION_FILE = "conversation_memory.json"
MEMORY_SIZE = 10  # Keep last 10 interactions

os.makedirs(TTS_DIR, exist_ok=True)

# ---------------------------
# Global Shared Resources
# ---------------------------
conversation_history = []
memory_lock = threading.Lock()
frame_lock = threading.Lock()
current_frame = None
detection_enabled = False
detected_objects = []

# ---------------------------
# Conversation Memory Functions
# ---------------------------
def save_conversation():
    with memory_lock:
        with open(CONVERSATION_FILE, 'w') as f:
            json.dump(conversation_history[-MEMORY_SIZE:], f)

def load_conversation():
    global conversation_history
    try:
        with open(CONVERSATION_FILE, 'r') as f:
            conversation_history = json.load(f)
    except FileNotFoundError:
        conversation_history = []

def update_memory(user_input, ai_response):
    timestamp = datetime.now().isoformat()
    with memory_lock:
        conversation_history.append({'timestamp': timestamp, 'user': user_input, 'ai': ai_response})
        if len(conversation_history) > MEMORY_SIZE:
            conversation_history.pop(0)
    save_conversation()

# ---------------------------
# YOLOv5 Model Initialization
# ---------------------------
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    model.conf = 0.5
    model.iou = 0.45
except Exception as e:
    logging.error(f"Failed to load YOLOv5 model: {e}. Object detection will be unavailable.")
    model = None

# ---------------------------
# Utility Functions
# ---------------------------
def generate_tts_for_browser(text, filename, lang='en'):
    """Generates TTS for playback in the user's web browser."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts_path = os.path.join(TTS_DIR, filename)
        tts.save(tts_path)
        return True
    except Exception as e:
        logging.error(f"Browser TTS generation failed: {str(e)}")
        return False

def get_dominant_color(frame, x1, y1, x2, y2):
    try:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return "unknown"
        avg_color_per_row = np.average(roi, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        dominant_color = tuple(map(int, avg_color))
        return f"rgb({dominant_color[2]}, {dominant_color[1]}, {dominant_color[0]})"
    except Exception as e:
        logging.error(f"Color detection error: {str(e)}")
        return "unknown"

# ---------------------------
# Video Processing
# ---------------------------
def process_frame(frame):
    global detected_objects
    if not detection_enabled or model is None:
        detected_objects = []
        return frame
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = model(img)
    df = results.pandas().xyxy[0]
    
    current_detected = []
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        obj_info = {
            'name': row['name'], 'confidence': round(row['confidence'], 2),
            'position': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'color': get_dominant_color(frame, x1, y1, x2, y2)
        }
        current_detected.append(obj_info)
    
    detected_objects = current_detected
    processed_rgb = np.squeeze(results.render())
    return cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)

def generate_frames():
    global current_frame
    cap = None
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "Feed Disconnected", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    while True:
        try:
            if cap is None:
                stream_url = f"http://{RASPBERRY_PI_IP}:5000/video_feed"
                logging.info(f"Attempting to connect to video stream at {stream_url}")
                cap = cv2.VideoCapture(stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened(): raise ConnectionError("VideoCapture not opened.")

            success, frame = cap.read()
            if not success: raise ConnectionError("Failed to read frame from stream.")

            processed_frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if ret:
                with frame_lock:
                    current_frame = frame.copy()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logging.warning(f"Video stream error: {str(e)}. Retrying in 5 seconds.")
            if cap: cap.release()
            cap = None
            with frame_lock: current_frame = None
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(5)

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_enabled
    detection_enabled = not detection_enabled
    logging.info(f"Object detection toggled: {'Enabled' if detection_enabled else 'Disabled'}")
    return jsonify({"status": "success", "detection_enabled": detection_enabled})

def tell_pi_to_speak(text_to_speak):
    """Sends a request to the Raspberry Pi to speak the given text."""
    try:
        requests.post(
            f"http://{RASPBERRY_PI_IP}:5000/speak",
            json={"text": text_to_speak},
            timeout=5
        )
        logging.info("Sent text to Raspberry Pi for speech.")
    except Exception as e:
        logging.error(f"Failed to send speech request to RPi: {str(e)}")

@app.route('/analyze', methods=['POST'])
def analyze_scene():
    try:
        with frame_lock:
            if current_frame is None:
                return jsonify({"error": "Video feed is currently unavailable."}), 408
            frame = current_frame.copy()

        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        analysis_prompt = "Analyze this image from a robot's perspective. Describe the scene, key objects, and potential obstacles in a concise paragraph."
        response = ollama.chat(
            model='llava:latest',
            messages=[{'role': 'user', 'content': analysis_prompt, 'images': [img_base64]}],
            options={'temperature': 0.2, 'timeout': 120}
        )
        description = response.get('message', {}).get('content', 'Analysis failed.')
        
        tell_pi_to_speak(description)

        filename = f"desc_{int(time.time())}.mp3"
        tts_url = f"/tts/{filename}" if generate_tts_for_browser(description, filename) else None

        return jsonify({"description": description, "tts_url": tts_url, "objects": detected_objects})
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return jsonify({"error": "An internal error occurred during analysis."}), 500

@app.route('/control', methods=['POST'])
def control():
    try:
        command = request.json.get('command', '').lower()
        if command not in ['forward', 'backward', 'left', 'right', 'stop']:
            return jsonify({"error": "Invalid command"}), 400
        
        logging.info(f"Sending command to Pi: {command}")
        response = requests.post(f'http://{RASPBERRY_PI_IP}:5000/control', json={'command': command}, timeout=3)
        response.raise_for_status()
        return jsonify({"status": "success", "command": command})
    except requests.exceptions.RequestException as e:
        logging.error(f"Control error connecting to Pi: {str(e)}")
        return jsonify({"error": f"Connection to robot at {RASPBERRY_PI_IP} failed."}), 500

@app.route('/tts/<filename>')
def serve_tts(filename):
    return send_from_directory(TTS_DIR, filename)

@app.route('/conversation', methods=['GET', 'POST'])
def handle_conversation():
    if request.method == 'POST':
        data = request.get_json()
        user_input = data.get('message', '')
        lang = data.get('language', 'en')
        
        with frame_lock:
            frame = current_frame.copy() if current_frame is not None else None
        
        img_base64 = None
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

        system_prompt = """You are Infi Sage, a helpful AI assistant for a robot. Your capabilities are observation and movement.
        - You can see through a camera. When asked about what you see, describe it based on the image provided.
        - You can move. To move, you must issue a command.
        - Respond to the user's query. If the query implies movement, you MUST end your response with `COMMAND: <action>`, where <action> is one of: forward, backward, left, right, stop.
        - Example: User says "Move ahead." Your response should be "Okay, moving forward. COMMAND: forward".
        - Be concise and direct.
        """
        messages = [{'role': 'system', 'content': system_prompt}]
        
        if img_base64:
            messages.append({'role': 'user', 'content': '[Current Robot View]', 'images': [img_base64]})

        with memory_lock:
            for entry in conversation_history[-3:]: # Use recent history for context
                messages.extend([
                    {'role': 'user', 'content': entry.get('user', '')},
                    {'role': 'assistant', 'content': entry.get('ai', '')}
                ])
        
        messages.append({'role': 'user', 'content': user_input})

        try:
            response = ollama.chat(
                model='llava:latest',
                messages=messages,
                options={'temperature': 0.3, 'timeout': 60}
            )
            ai_response = response.get('message', {}).get('content', 'Sorry, I could not process that.')
            
            # Extract and send command if present
            if "COMMAND:" in ai_response:
                command_part = ai_response.split("COMMAND:")[1].strip().lower()
                if command_part in ['forward', 'backward', 'left', 'right', 'stop']:
                    control_response = requests.post(f'http://{RASPBERRY_PI_IP}:5000/control', json={'command': command_part}, timeout=3)
                    control_response.raise_for_status()

            update_memory(user_input, ai_response)
            tell_pi_to_speak(ai_response.split("COMMAND:")[0].strip()) # Speak the text part only

            filename = f"conv_{int(time.time())}.mp3"
            tts_url = f"/tts/{filename}" if generate_tts_for_browser(ai_response, filename, lang=lang) else None

            return jsonify({"response": ai_response, "tts_url": tts_url})
        except Exception as e:
            logging.error(f"Conversation error: {str(e)}")
            return jsonify({"error": "An error occurred in conversation."}), 500
    else: # GET request
        with memory_lock:
            return jsonify({"conversation": conversation_history})

# ---------------------------
# Cleanup & Initialization
# ---------------------------
def cleanup_old_files():
    while True:
        try:
            now = time.time()
            for f in os.listdir(TTS_DIR):
                path = os.path.join(TTS_DIR, f)
                if os.stat(path).st_mtime < now - (3600 * 6): # 6 hours old
                    os.remove(path)
            time.sleep(3600) # Check every hour
        except Exception as e:
            logging.error(f"Cleanup error: {str(e)}")

if __name__ == '__main__':
    load_conversation()
    threading.Thread(target=cleanup_old_files, daemon=True).start()
    app.run(host='0.0.0.0', port=5005, debug=True)
