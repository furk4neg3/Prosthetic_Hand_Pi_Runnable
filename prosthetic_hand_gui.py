#!/usr/bin/env python3
"""
prosthetic_hand_gui.py
======================

Web-based GUI for the sEMG-based prosthetic hand control system.
Run this INSTEAD of prosthetic_hand.py for a visual interface.

Features:
  - Beautiful web interface accessible from any device on the network
  - Movement cards with NINAPRO DB1 images
  - Real-time inference results and servo status
  - Live EMG signal visualization
  - System health monitoring

Setup:
  pip install flask flask-socketio numpy scipy

  For servo control (Raspberry Pi only):
  pip install tflite-runtime adafruit-circuitpython-servokit

Usage:
  python3 prosthetic_hand_gui.py
  Then open http://<pi-ip>:5000 in your browser

Image Setup:
  Place NINAPRO DB1 movement images in:
    static/images/movements/
  Naming convention:
    movement_01.png  (original label 1: Index flexion)
    movement_02.png  (original label 2: Index extension)
    ...
    movement_40.png  (original label 40: Three Finger Sphere Grasp)
"""

import os
import sys
import time
import json
import threading
import numpy as np
from scipy.signal import butter, filtfilt

# ─────────────────────────────────────────────────────────────────────────────
# Try to import TFLite runtime (lightweight) first, fall back to full TF
# ─────────────────────────────────────────────────────────────────────────────
try:
    from ai_edge_litert.interpreter import Interpreter
    print("✅ Using ai-edge-litert")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        print("✅ Using tflite-runtime")
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            print("✅ Using tensorflow.lite")
        except ImportError:
            print("❌ No TFLite/LiteRT interpreter found!")
            print("   Install with: pip install ai-edge-litert")
            sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Try to import servo control libraries
# ─────────────────────────────────────────────────────────────────────────────
SERVO_AVAILABLE = False
try:
    from adafruit_servokit import ServoKit
    SERVO_AVAILABLE = True
    print("✅ Servo libraries loaded")
except ImportError:
    print("⚠️  Servo libraries not found — running in SIMULATION mode")

# ─────────────────────────────────────────────────────────────────────────────
# Flask & SocketIO
# ─────────────────────────────────────────────────────────────────────────────
from flask import Flask, render_template, jsonify, send_from_directory
try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("⚠️  flask-socketio not found, using polling fallback")


# =============================================================================
# CONFIGURATION (same as prosthetic_hand.py)
# =============================================================================
class Config:
    """All configurable parameters in one place."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "dann_inference_all_new_29.tflite")
    SAMPLES_PATH = os.path.join(SCRIPT_DIR, "pi_samples.npz")
    IMAGES_DIR = os.path.join(SCRIPT_DIR, "static", "images", "movements")

    SAMPLING_RATE = 100
    NUM_CHANNELS = 10
    WINDOW_SIZE = 50
    LOWCUT = 20.0
    HIGHCUT = 45.0
    FILTER_ORDER = 4

    NUM_SERVOS = 5
    PCA9685_I2C_ADDRESS = 0x40
    PCA9685_FREQUENCY = 50

    SERVO_CHANNELS = {
        "thumb":  0,
        "index":  1,
        "middle": 2,
        "ring":   3,
        "little": 4,
    }

    SERVO_INVERTED = {
        "thumb":  False,
        "index":  False,
        "middle": False,
        "ring":   True,
        "little": True,
    }

    SERVO_MIN_PULSE = 500
    SERVO_MAX_PULSE = 2500

    REST_ANGLES = {
        "thumb":  50,
        "index":  50,
        "middle": 40,
        "ring":   70,
        "little": 50,
    }

    HOLD_DURATION = 2.0
    SAMPLES_PER_MOVEMENT = 10


# =============================================================================
# MOVEMENT DEFINITIONS (same as prosthetic_hand.py)
# =============================================================================
MOVEMENTS = [
    # ─── Exercise 1: Basic finger movements (labels 1-12) ───
    {
        "encoded": 0, "original": 1,
        "name": "Index flexion",
        "exercise": 1,
        "angles": {"thumb": 50, "index": 180, "middle": 40, "ring": 70, "little": 50}
    },
    {
        "encoded": 1, "original": 2,
        "name": "Index extension",
        "exercise": 1,
        "angles": {"thumb": 50, "index": 0, "middle": 40, "ring": 70, "little": 50}
    },
    {
        "encoded": 2, "original": 3,
        "name": "Middle flexion",
        "exercise": 1,
        "angles": {"thumb": 50, "index": 50, "middle": 180, "ring": 70, "little": 50}
    },
    {
        "encoded": 3, "original": 4,
        "name": "Middle extension",
        "exercise": 1,
        "angles": {"thumb": 50, "index": 50, "middle": 0, "ring": 70, "little": 50}
    },
    {
        "encoded": 4, "original": 5,
        "name": "Ring flexion",
        "exercise": 1,
        "angles": {"thumb": 50, "index": 50, "middle": 40, "ring": 180, "little": 50}
    },
    {
        "encoded": 5, "original": 6,
        "name": "Ring extension",
        "exercise": 1,
        "angles": {"thumb": 50, "index": 50, "middle": 40, "ring": 0, "little": 50}
    },
    {
        "encoded": 6, "original": 7,
        "name": "Little finger flexion",
        "exercise": 1,
        "angles": {"thumb": 50, "index": 50, "middle": 40, "ring": 70, "little": 180}
    },
    {
        "encoded": 7, "original": 8,
        "name": "Little finger extension",
        "exercise": 1,
        "angles": {"thumb": 50, "index": 50, "middle": 40, "ring": 70, "little": 0}
    },
    {
        "encoded": 8, "original": 9,
        "name": "Thumb adduction",
        "exercise": 1,
        "angles": {"thumb": 0, "index": 50, "middle": 40, "ring": 70, "little": 50}
    },
    {
        "encoded": 9, "original": 10,
        "name": "Thumb abduction",
        "exercise": 1,
        "angles": {"thumb": 120, "index": 50, "middle": 40, "ring": 70, "little": 50}
    },
    {
        "encoded": 10, "original": 11,
        "name": "Thumb flexion",
        "exercise": 1,
        "angles": {"thumb": 0, "index": 50, "middle": 40, "ring": 70, "little": 50}
    },
    {
        "encoded": 11, "original": 12,
        "name": "Thumb extension",
        "exercise": 1,
        "angles": {"thumb": 180, "index": 50, "middle": 40, "ring": 70, "little": 50}
    },

    # ─── Exercise 2: Hand configurations (labels 13-20) ───
    {
        "encoded": 12, "original": 13,
        "name": "Thumb up",
        "exercise": 2,
        "angles": {"thumb": 0, "index": 180, "middle": 180, "ring": 180, "little": 180}
    },
    {
        "encoded": 13, "original": 14,
        "name": "Scissors",
        "exercise": 2,
        "angles": {"thumb": 180, "index": 0, "middle": 0, "ring": 180, "little": 180}
    },
    {
        "encoded": 14, "original": 15,
        "name": "Three move",
        "exercise": 2,
        "angles": {"thumb": 0, "index": 0, "middle": 0, "ring": 180, "little": 180}
    },
    {
        "encoded": 15, "original": 16,
        "name": "Thumb flexion",
        "exercise": 2,
        "angles": {"thumb": 180, "index": 0, "middle": 0, "ring": 0, "little": 0}
    },
    {
        "encoded": 16, "original": 17,
        "name": "Abduction of all fingers",
        "exercise": 2,
        "angles": {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "little": 0}
    },
    {
        "encoded": 17, "original": 18,
        "name": "Fist",
        "exercise": 2,
        "angles": {"thumb": 180, "index": 180, "middle": 180, "ring": 180, "little": 180}
    },
    {
        "encoded": 18, "original": 19,
        "name": "Pointing index",
        "exercise": 2,
        "angles": {"thumb": 180, "index": 0, "middle": 180, "ring": 180, "little": 180}
    },
    {
        "encoded": 19, "original": 20,
        "name": "Abduction of extended fingers",
        "exercise": 2,
        "angles": {"thumb": 50, "index": 50, "middle": 50, "ring": 90, "little": 50}
    },

    # ─── Exercise 3: Grasping / functional (labels 30-35, 38-40) ───
    {
        "encoded": 20, "original": 30,
        "name": "Large diameter grasp",
        "exercise": 3,
        "angles": {"thumb": 80, "index": 90, "middle": 100, "ring": 110, "little": 80}
    },
    {
        "encoded": 21, "original": 31,
        "name": "Small diameter grasp",
        "exercise": 3,
        "angles": {"thumb": 130, "index": 130, "middle": 130, "ring": 140, "little": 130}
    },
    {
        "encoded": 22, "original": 32,
        "name": "Fixed hook grasp",
        "exercise": 3,
        "angles": {"thumb": 0, "index": 110, "middle": 110, "ring": 130, "little": 110}
    },
    {
        "encoded": 23, "original": 33,
        "name": "Index finger extension grasp",
        "exercise": 3,
        "angles": {"thumb": 90, "index": 50, "middle": 150, "ring": 180, "little": 180}
    },
    {
        "encoded": 24, "original": 34,
        "name": "Medium wrap",
        "exercise": 3,
        "angles": {"thumb": 110, "index": 110, "middle": 110, "ring": 150, "little": 110}
    },
    {
        "encoded": 25, "original": 35,
        "name": "Ring grasp",
        "exercise": 3,
        "angles": {"thumb": 110, "index": 90, "middle": 0, "ring": 0, "little": 0}
    },
    {
        "encoded": 26, "original": 38,
        "name": "Writing tripod Grasp",
        "exercise": 3,
        "angles": {"thumb": 120, "index": 110, "middle": 150, "ring": 180, "little": 180}
    },
    {
        "encoded": 27, "original": 39,
        "name": "Power Sphere Grasp",
        "exercise": 3,
        "angles": {"thumb": 75, "index": 75, "middle": 75, "ring": 110, "little": 75}
    },
    {
        "encoded": 28, "original": 40,
        "name": "Three Finger Sphere Grasp",
        "exercise": 3,
        "angles": {"thumb": 80, "index": 80, "middle": 80, "ring": 160, "little": 180}
    },
]


# =============================================================================
# PREPROCESSING MODULE (same as prosthetic_hand.py)
# =============================================================================
class Preprocessor:
    """Applies the same preprocessing as the training pipeline."""

    def __init__(self, config, scaler_mean, scaler_scale):
        self.config = config
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        nyq = config.SAMPLING_RATE / 2.0
        self.b, self.a = butter(
            config.FILTER_ORDER,
            [config.LOWCUT / nyq, config.HIGHCUT / nyq],
            btype='band'
        )

    def normalize(self, emg_window):
        return (emg_window - self.scaler_mean) / self.scaler_scale

    def process(self, raw_window):
        normalized = self.normalize(raw_window)
        return normalized.astype(np.float32)


# =============================================================================
# TFLITE INFERENCE MODULE (same as prosthetic_hand.py)
# =============================================================================
class InferenceEngine:
    """Runs TFLite model inference on preprocessed sEMG windows."""

    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, preprocessed_window):
        sample = preprocessed_window[np.newaxis, :, :].astype(np.float32)
        t_start = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], sample)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        t_end = time.time()

        probs = output[0]
        if np.any(probs < 0):
            exp_p = np.exp(probs - np.max(probs))
            probs = exp_p / exp_p.sum()

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        inference_time = (t_end - t_start) * 1000

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': probs.tolist(),
            'inference_time_ms': inference_time,
        }


# =============================================================================
# SERVO CONTROL MODULE (same as prosthetic_hand.py)
# =============================================================================
class ServoController:
    """Controls 5 servo motors via PCA9685."""

    def __init__(self, config):
        self.config = config
        self.channels = config.SERVO_CHANNELS
        self.rest_angles = config.REST_ANGLES
        self.kit = None
        self.current_angles = dict(config.REST_ANGLES)

        if SERVO_AVAILABLE:
            try:
                self.kit = ServoKit(
                    channels=16,
                    address=config.PCA9685_I2C_ADDRESS,
                    frequency=config.PCA9685_FREQUENCY,
                )
                for name, channel in self.channels.items():
                    self.kit.servo[channel].set_pulse_width_range(
                        config.SERVO_MIN_PULSE,
                        config.SERVO_MAX_PULSE,
                    )
            except Exception as e:
                print(f"⚠️  Failed to initialize PCA9685: {e}")
                self.kit = None

    def set_angles(self, angles_dict):
        self.current_angles = dict(angles_dict)
        for name, channel in self.channels.items():
            angle = angles_dict.get(name, 0)
            angle = max(0, min(180, angle))
            inverted = self.config.SERVO_INVERTED.get(name, False)
            actual_angle = (180 - angle) if inverted else angle
            if self.kit is not None:
                try:
                    self.kit.servo[channel].angle = actual_angle
                except Exception as e:
                    print(f"  ⚠️  Error setting {name} (ch{channel}) to {angle}°: {e}")

    def go_to_rest(self):
        self.set_angles(self.rest_angles)

    def get_current_angles(self):
        return dict(self.current_angles)


# =============================================================================
# SAMPLE MANAGER (same as prosthetic_hand.py)
# =============================================================================
class SampleManager:
    """Manages pre-stored sEMG samples for each movement."""

    def __init__(self, samples_path):
        data = np.load(samples_path)
        self.samples = data['samples']
        self.encoded_labels = data['encoded_labels']
        self.original_labels = data['original_labels']
        self.scaler_mean = data['scaler_mean']
        self.scaler_scale = data['scaler_scale']
        self.n_movements = self.samples.shape[0]
        self.n_samples_per_movement = self.samples.shape[1]
        self.counters = np.zeros(self.n_movements, dtype=int)

    def get_next_sample(self, movement_index):
        idx = self.counters[movement_index]
        sample = self.samples[movement_index, idx]
        self.counters[movement_index] = (idx + 1) % self.n_samples_per_movement
        return sample, int(idx)


# =============================================================================
# FLASK APPLICATION
# =============================================================================
app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

if SOCKETIO_AVAILABLE:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
else:
    socketio = None

# Global state
config = Config()
hand_system = None
system_status = {
    "initialized": False,
    "servo_mode": "simulation",
    "model_loaded": False,
    "last_action": None,
    "busy": False,
}

# Cancellation event — set this to interrupt the current hold phase
cancel_event = threading.Event()
# Lock to serialize the actual servo access (inference is fast, hold is what we cancel)
execution_lock = threading.Lock()


def initialize_system():
    """Initialize all hardware/software components."""
    global hand_system, system_status

    try:
        sample_mgr = SampleManager(config.SAMPLES_PATH)
        preprocessor = Preprocessor(config, sample_mgr.scaler_mean, sample_mgr.scaler_scale)
        engine = InferenceEngine(config.MODEL_PATH)
        servo = ServoController(config)

        hand_system = {
            "sample_mgr": sample_mgr,
            "preprocessor": preprocessor,
            "engine": engine,
            "servo": servo,
        }

        system_status["initialized"] = True
        system_status["model_loaded"] = True
        system_status["servo_mode"] = "hardware" if servo.kit else "simulation"

        # Set servos to rest position
        servo.go_to_rest()

        print("✅ System initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_movement_image_path(movement):
    """Get the image path for a movement, checking multiple naming conventions."""
    original_label = movement["original"]
    possible_names = [
        f"movement_{original_label:02d}.png",
        f"movement_{original_label:02d}.jpg",
        f"movement_{original_label:02d}.jpeg",
        f"movement_{original_label:02d}.webp",
        f"movement_{original_label}.png",
        f"movement_{original_label}.jpg",
        f"movement_{original_label}.jpeg",
        f"movement_{original_label}.webp",
    ]

    for name in possible_names:
        full_path = os.path.join(config.IMAGES_DIR, name)
        if os.path.exists(full_path):
            return f"/static/images/movements/{name}"

    return None


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Get system status."""
    return jsonify(system_status)


@app.route('/api/movements')
def api_movements():
    """Get all movements with their image paths."""
    movements_data = []
    for mov in MOVEMENTS:
        img_path = get_movement_image_path(mov)
        movements_data.append({
            "encoded": mov["encoded"],
            "original": mov["original"],
            "name": mov["name"],
            "exercise": mov["exercise"],
            "image": img_path,
            "angles": mov["angles"],
        })
    return jsonify(movements_data)


@app.route('/api/execute/<int:movement_index>', methods=['POST'])
def api_execute(movement_index):
    """Execute a movement. If one is already running, cancel its hold and start the new one."""
    global system_status, cancel_event

    if not system_status["initialized"]:
        return jsonify({"error": "System not initialized"}), 503

    if movement_index < 0 or movement_index >= len(MOVEMENTS):
        return jsonify({"error": f"Invalid movement index: {movement_index}"}), 400

    # If a movement is currently executing, signal it to cancel its hold
    if system_status["busy"]:
        cancel_event.set()

    # Execute in background thread so the response returns immediately
    def execute_movement():
        global system_status, cancel_event
        with execution_lock:
            # Reset cancel event for this new execution
            cancel_event = threading.Event()
            system_status["busy"] = True
            movement = MOVEMENTS[movement_index]

            try:
                # Step 1: Get sample
                sample_mgr = hand_system["sample_mgr"]
                raw_window, sample_idx = sample_mgr.get_next_sample(movement_index)

                # Get EMG data for visualization
                emg_preview = raw_window[:, :10].tolist()

                # Step 2: Preprocess
                preprocessor = hand_system["preprocessor"]
                processed_window = preprocessor.process(raw_window)

                # Step 3: Inference
                engine = hand_system["engine"]
                result = engine.predict(processed_window)

                predicted_class = result['predicted_class']
                confidence = result['confidence']
                inference_time = result['inference_time_ms']
                predicted_movement = MOVEMENTS[predicted_class]

                # Emit inference result
                inference_data = {
                    "selected": {
                        "index": movement_index,
                        "name": movement["name"],
                        "original_label": movement["original"],
                    },
                    "prediction": {
                        "index": predicted_class,
                        "name": predicted_movement["name"],
                        "original_label": predicted_movement["original"],
                        "confidence": confidence,
                        "inference_time_ms": inference_time,
                        "correct": predicted_class == movement_index,
                    },
                    "sample_index": sample_idx + 1,
                    "total_samples": config.SAMPLES_PER_MOVEMENT,
                    "emg_data": emg_preview,
                    "probabilities": result['all_probabilities'],
                }

                emit_event("inference_result", inference_data)

                # Step 4: Execute servo movement
                target_angles = predicted_movement['angles']
                emit_status("moving", f"Moving servos: {predicted_movement['name']}")

                servo = hand_system["servo"]
                servo.set_angles(target_angles)

                emit_event("servo_update", {"angles": target_angles, "state": "active"})

                # Step 5: Hold — use cancel_event.wait() so it can be interrupted
                emit_status("holding", f"Holding for {config.HOLD_DURATION}s...")
                cancelled = cancel_event.wait(timeout=config.HOLD_DURATION)

                if cancelled:
                    # Another movement was requested — skip rest, let the new one take over
                    return

                # Step 6: Return to rest
                emit_status("resting", "Returning to rest position...")
                servo.go_to_rest()
                emit_event("servo_update", {"angles": config.REST_ANGLES, "state": "rest"})

                time.sleep(0.3)

                system_status["last_action"] = {
                    "movement": movement["name"],
                    "prediction": predicted_movement["name"],
                    "confidence": confidence,
                    "correct": predicted_class == movement_index,
                    "time": time.strftime("%H:%M:%S"),
                }

                emit_status("ready", "Ready for next movement")

            except Exception as e:
                emit_status("error", f"Error: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                system_status["busy"] = False

    thread = threading.Thread(target=execute_movement, daemon=True)
    thread.start()

    return jsonify({"status": "executing", "movement": MOVEMENTS[movement_index]["name"]})


@app.route('/api/rest', methods=['POST'])
def api_rest():
    """Return servos to rest position."""
    if hand_system and hand_system["servo"]:
        hand_system["servo"].go_to_rest()
        return jsonify({"status": "ok"})
    return jsonify({"error": "System not initialized"}), 503


@app.route('/api/servo_angles')
def api_servo_angles():
    """Get current servo angles."""
    if hand_system and hand_system["servo"]:
        return jsonify(hand_system["servo"].get_current_angles())
    return jsonify(Config.REST_ANGLES)


def emit_status(state, message):
    """Emit a status update via SocketIO or store for polling."""
    system_status["current_state"] = state
    system_status["current_message"] = message
    if socketio and SOCKETIO_AVAILABLE:
        socketio.emit('status_update', {"state": state, "message": message})


def emit_event(event, data):
    """Emit an event via SocketIO."""
    if socketio and SOCKETIO_AVAILABLE:
        socketio.emit(event, data)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🦾 PROSTHETIC HAND CONTROL SYSTEM — WEB GUI")
    print("=" * 60)
    print()

    # Create required directories
    os.makedirs(config.IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'), exist_ok=True)

    # Check required files
    if not os.path.exists(config.MODEL_PATH):
        print(f"❌ Model not found: {config.MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(config.SAMPLES_PATH):
        print(f"❌ Samples not found: {config.SAMPLES_PATH}")
        sys.exit(1)

    # Initialize the system
    if not initialize_system():
        sys.exit(1)

    # Check for movement images
    img_count = 0
    for mov in MOVEMENTS:
        if get_movement_image_path(mov):
            img_count += 1
    print(f"📸 Movement images found: {img_count}/{len(MOVEMENTS)}")
    if img_count < len(MOVEMENTS):
        print(f"   Place images in: {config.IMAGES_DIR}")
        print(f"   Naming: movement_01.png, movement_02.png, ...")

    print()
    print("=" * 60)
    print("🌐 Starting web server...")
    print("   Open in browser: http://localhost:5000")
    print("   Or from another device: http://<this-ip>:5000")
    print("=" * 60)
    print()

    if SOCKETIO_AVAILABLE:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    else:
        app.run(host='0.0.0.0', port=5000, debug=False)
