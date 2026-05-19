#!/usr/bin/env python3
"""
prosthetic_hand.py
==================

Main control script for the sEMG-based prosthetic hand.
Run this on the Raspberry Pi 5.

Hardware:
  - Raspberry Pi 5 (2GB)
  - PCA9685 servo driver (I2C)
  - 5 servo motors (thumb, index, middle, ring, little)
  - 5V 2A power supply (connected to PCA9685 V+ terminal)
  - 3D printed prosthetic hand with rope-pull mechanism

Flow:
  1. Show terminal UI with 29 NINAPRO DB1 movements
  2. User selects a movement by number
  3. System circularly picks one of 10 pre-stored raw sEMG windows
  4. Preprocesses: bandpass filter → StandardScaler normalization
  5. Runs TFLite inference → predicted label
  6. Maps predicted label to servo angles for 5 fingers
  7. Sends angles to PCA9685 → servos move
  8. Waits 2 seconds
  9. Returns all servos to rest position
  10. Loops back to step 1

Setup:
  pip install tflite-runtime numpy scipy adafruit-circuitpython-servokit

  Files needed in the same directory:
    - dann_inference_all_new_29.tflite  (the TFLite model)
    - pi_samples.npz                   (samples + scaler params)

Usage:
  python3 prosthetic_hand.py
"""

import os
import sys
import time
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
    print("   Install with: pip install adafruit-circuitpython-servokit")


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    """All configurable parameters in one place."""

    # ── File paths ──
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "dann_inference_all_new_29.tflite")
    SAMPLES_PATH = os.path.join(SCRIPT_DIR, "pi_samples.npz")

    # ── Signal processing ──
    SAMPLING_RATE = 100       # Hz
    NUM_CHANNELS = 10
    WINDOW_SIZE = 50          # samples (500ms at 100Hz)
    LOWCUT = 20.0             # Bandpass low cutoff (Hz)
    HIGHCUT = 45.0            # Bandpass high cutoff (Hz)
    FILTER_ORDER = 4          # Butterworth filter order

    # ── Servo configuration ──
    NUM_SERVOS = 5
    PCA9685_I2C_ADDRESS = 0x40
    PCA9685_FREQUENCY = 50     # Hz (standard for servos)

    # Servo channel assignments on PCA9685
    SERVO_CHANNELS = {
        "thumb":  0,
        "index":  1,
        "middle": 2,
        "ring":   3,
        "little": 4,
    }

    # ALL servos inverted to reverse rotation direction
    # REST_ANGLES = 180 compensates so rest position stays the same physically
    SERVO_INVERTED = {
        "thumb":  False,
        "index":  False,
        "middle": False,
        "ring":   True,
        "little": True,
    }

    # Minimum and maximum pulse widths (microseconds) — adjust for your servos
    SERVO_MIN_PULSE = 500    # µs — fully open / rest
    SERVO_MAX_PULSE = 2500   # µs — fully closed / max flex

    # Rest position angles (degrees) — hand fully open
    # Adjust these for your specific 3D-printed hand
    REST_ANGLES = {
        "thumb":  50,
        "index":  50,
        "middle": 40,
        "ring":   70,
        "little": 50,
    }

    # How long to hold the movement pose before returning to rest (seconds)
    HOLD_DURATION = 2.0

    # Number of samples per movement for circular selection
    SAMPLES_PER_MOVEMENT = 10


# =============================================================================
# MOVEMENT DEFINITIONS
# =============================================================================
# The 29 movements from NINAPRO DB1 after exclusions.
# Encoded label (0-28) maps to original NINAPRO label.
# Each entry: (encoded_label, original_label, name, servo_angles_dict)
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ IMPORTANT: The servo angles below are REASONABLE DEFAULTS.         │
# │ You MUST calibrate them for your specific 3D-printed hand.         │
# │                                                                    │
# │ Angles: 0° = finger fully open (rest)                              │
# │        180° = finger fully closed (max flexion)                     │
# │ Intermediate values = partial movement                              │
# │                                                                    │
# │ Your hand uses ropes pulled by servos, so the relationship         │
# │ between servo angle and finger position depends on your specific   │
# │ mechanical design.                                                 │
# └─────────────────────────────────────────────────────────────────────┘

MOVEMENTS = [
    # ─── Exercise 1: Basic finger movements (labels 1-12) ───
    {
        "encoded": 0, "original": 1,
        "name": "Index flexion",
        "angles": {"thumb": 50, "index": 180, "middle": 40, "ring": 70, "little": 50}
    },
    {
        "encoded": 1, "original": 2,
        "name": "Index extension",
        "angles": {"thumb": 50, "index": 0, "middle": 40, "ring": 70, "little": 50}
    },
    {
        "encoded": 2, "original": 3,
        "name": "Middle flexion",
        "angles": {"thumb": 50, "index": 50, "middle": 180, "ring": 70, "little": 50}
    },
    {
        "encoded": 3, "original": 4,
        "name": "Middle extension",
        "angles": {"thumb": 50, "index": 50, "middle": 0, "ring": 70, "little": 50}
    },
    {
        "encoded": 4, "original": 5,
        "name": "Ring flexion",
        "angles": {"thumb": 50, "index": 50, "middle": 40, "ring": 180, "little": 50}
    },
    {
        "encoded": 5, "original": 6,
        "name": "Ring extension",
        "angles": {"thumb": 50, "index": 50, "middle": 40, "ring": 0, "little": 50}
    },
    {
        "encoded": 6, "original": 7,
        "name": "Little finger flexion",
        "angles": {"thumb": 50, "index": 50, "middle": 40, "ring": 70, "little": 180}
    },
    {
        "encoded": 7, "original": 8,
        "name": "Little finger extension",
        "angles": {"thumb": 50, "index": 50, "middle": 40, "ring": 70, "little": 0}
    },
    {
        "encoded": 8, "original": 9,
        "name": "Thumb adduction",
        "angles": {"thumb": 0, "index": 50, "middle": 40, "ring": 70, "little": 50}
    },
    {
        "encoded": 9, "original": 10,
        "name": "Thumb abduction",
        "angles": {"thumb": 120, "index": 50, "middle": 40, "ring": 70, "little": 50}
    },
    {
        "encoded": 10, "original": 11,
        "name": "Thumb flexion",
        "angles": {"thumb": 0, "index": 50, "middle": 40, "ring": 70, "little": 50}
    },
    {
        "encoded": 11, "original": 12,
        "name": "Thumb extension",
        "angles": {"thumb": 180, "index": 50, "middle": 40, "ring": 70, "little": 50}
    },

    # ─── Exercise 2: Hand configurations (labels 13-20) ───
    {
        "encoded": 12, "original": 13,
        "name": "Thumb up",
        "angles": {"thumb": 0, "index": 180, "middle": 180, "ring": 180, "little": 180}
    },
    {
        "encoded": 13, "original": 14,
        "name": "Scissors",
        "angles": {"thumb": 180, "index": 0, "middle": 0, "ring": 180, "little": 180}
    },
    {
        "encoded": 14, "original": 15,
        "name": "Three move",
        "angles": {"thumb": 0, "index": 0, "middle": 0, "ring": 180, "little": 180}
    },
    {
        "encoded": 15, "original": 16,
        "name": "Thumb flexion",
        "angles": {"thumb": 180, "index": 0, "middle": 0, "ring": 0, "little": 0}
    },
    {
        "encoded": 16, "original": 17,
        "name": "Abduction of all fingers",
        "angles": {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "little": 0}
    },
    {
        "encoded": 17, "original": 18,
        "name": "Fist",
        "angles": {"thumb": 180, "index": 180, "middle": 180, "ring": 180, "little": 180}
    },
    {
        "encoded": 18, "original": 19,
        "name": "Pointing index",
        "angles": {"thumb": 180, "index": 0, "middle": 180, "ring": 180, "little": 180}
    },
    {
        "encoded": 19, "original": 20,
        "name": "Abduction of extended fingers",
        "angles": {"thumb": 50, "index": 50, "middle": 50, "ring": 90, "little": 50}
    },

    # ─── Exercise 3: Grasping / functional (labels 30-35, 38-40) ───
    {
        "encoded": 20, "original": 30,
        "name": "Large diameter grasp",
        "angles": {"thumb": 80, "index": 90, "middle": 100, "ring": 110, "little": 80}
    },
    {
        "encoded": 21, "original": 31,
        "name": "Small diameter grasp",
        "angles": {"thumb": 130, "index": 130, "middle": 130, "ring": 140, "little": 130}
    },
    {
        "encoded": 22, "original": 32,
        "name": "Fixed hook grasp",
        "angles": {"thumb": 0, "index": 110, "middle": 110, "ring": 130, "little": 110}
    },
    {
        "encoded": 23, "original": 33,
        "name": "Index finger extension grasp",
        "angles": {"thumb": 90, "index": 50, "middle": 150, "ring": 180, "little": 180}
    },
    {
        "encoded": 24, "original": 34,
        "name": "Medium wrap",
        "angles": {"thumb": 110, "index": 110, "middle": 110, "ring": 150, "little": 110}
    },
    {
        "encoded": 25, "original": 35,
        "name": "Ring grasp",
        "angles": {"thumb": 110, "index": 90, "middle": 0, "ring": 0, "little": 0}
    },
    {
        "encoded": 26, "original": 38,
        "name": "Writing tripod Grasp",
        "angles": {"thumb": 120, "index": 110, "middle": 150, "ring": 180, "little": 180}
    },
    {
        "encoded": 27, "original": 39,
        "name": "Power Sphere Grasp",
        "angles": {"thumb": 75, "index": 75, "middle": 75, "ring": 110, "little": 75}
    },
    {
        "encoded": 28, "original": 40,
        "name": "Three Finger Sphere Grasp",
        "angles": {"thumb": 80, "index": 80, "middle": 80, "ring": 160, "little": 180}
    },
]


# =============================================================================
# PREPROCESSING MODULE
# =============================================================================
class Preprocessor:
    """
    Applies the same preprocessing as the training pipeline:
      1. Bandpass filter (20-45 Hz, 4th order Butterworth)
      2. StandardScaler normalization (using saved mean/scale)
    """

    def __init__(self, config, scaler_mean, scaler_scale):
        self.config = config
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale

        # Pre-compute filter coefficients (done once)
        nyq = config.SAMPLING_RATE / 2.0
        self.b, self.a = butter(
            config.FILTER_ORDER,
            [config.LOWCUT / nyq, config.HIGHCUT / nyq],
            btype='band'
        )
        print("✅ Preprocessor initialized")
        print(f"   Bandpass: {config.LOWCUT}–{config.HIGHCUT} Hz, order {config.FILTER_ORDER}")

    def bandpass_filter(self, emg_window):
        """Apply bandpass filter to a single window (window_size, n_channels)."""
        filtered = np.zeros_like(emg_window)
        for ch in range(emg_window.shape[1]):
            filtered[:, ch] = filtfilt(self.b, self.a, emg_window[:, ch])
        return filtered

    def normalize(self, emg_window):
        """Apply StandardScaler normalization using saved parameters."""
        return (emg_window - self.scaler_mean) / self.scaler_scale

    def process(self, raw_window):
        """
        Full preprocessing pipeline for a single raw sEMG window.

        NOTE: The samples in pi_samples.npz are already bandpass-filtered
        (they went through bandpass during extraction in Colab).
        So at inference time, we only need to apply the scaler normalization.

        If you're using truly raw data (e.g., from a live sensor), you would
        call bandpass_filter() first, then normalize().
        """
        # The samples are already bandpass-filtered from the Colab extraction
        # We just need to apply the StandardScaler normalization
        normalized = self.normalize(raw_window)
        return normalized.astype(np.float32)


# =============================================================================
# TFLITE INFERENCE MODULE
# =============================================================================
class InferenceEngine:
    """Runs TFLite model inference on preprocessed sEMG windows."""

    def __init__(self, model_path):
        print(f"Loading model: {model_path}")
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]['shape']
        output_shape = self.output_details[0]['shape']
        input_dtype = self.input_details[0]['dtype']

        print(f"✅ Model loaded successfully")
        print(f"   Input:  shape={input_shape}, dtype={input_dtype}")
        print(f"   Output: shape={output_shape}")

    def predict(self, preprocessed_window):
        """
        Run inference on a single preprocessed sEMG window.

        Args:
            preprocessed_window: numpy array (window_size, n_channels), float32

        Returns:
            dict with:
                'predicted_class': int (0-28)
                'confidence': float (0-1)
                'all_probabilities': numpy array
                'inference_time_ms': float
        """
        # Add batch dimension: (50, 10) → (1, 50, 10)
        sample = preprocessed_window[np.newaxis, :, :].astype(np.float32)

        # Run inference
        t_start = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], sample)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        t_end = time.time()

        # Get probabilities (apply softmax if needed)
        probs = output[0]
        if np.any(probs < 0):
            # Output is logits, apply softmax
            exp_p = np.exp(probs - np.max(probs))
            probs = exp_p / exp_p.sum()

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        inference_time = (t_end - t_start) * 1000  # ms

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': probs,
            'inference_time_ms': inference_time,
        }


# =============================================================================
# SERVO CONTROL MODULE
# =============================================================================
class ServoController:
    """Controls 5 servo motors via PCA9685 for prosthetic hand finger movement."""

    def __init__(self, config):
        self.config = config
        self.channels = config.SERVO_CHANNELS
        self.rest_angles = config.REST_ANGLES
        self.kit = None

        if SERVO_AVAILABLE:
            try:
                self.kit = ServoKit(
                    channels=16,
                    address=config.PCA9685_I2C_ADDRESS,
                    frequency=config.PCA9685_FREQUENCY,
                )
                # Configure pulse width range for each servo
                for name, channel in self.channels.items():
                    self.kit.servo[channel].set_pulse_width_range(
                        config.SERVO_MIN_PULSE,
                        config.SERVO_MAX_PULSE,
                    )
                print("✅ Servo controller initialized (PCA9685)")
            except Exception as e:
                print(f"⚠️  Failed to initialize PCA9685: {e}")
                print("   Running in SIMULATION mode")
                self.kit = None
        else:
            print("⚠️  Running in SIMULATION mode (no servo hardware)")

    def set_angles(self, angles_dict):
        """
        Set servo angles for all 5 fingers.

        Args:
            angles_dict: dict mapping finger name to angle (0-180)
                e.g., {"thumb": 90, "index": 120, "middle": 0, "ring": 0, "little": 0}
        """
        for name, channel in self.channels.items():
            angle = angles_dict.get(name, 0)
            angle = max(0, min(180, angle))  # Clamp to valid range

            # Flip angle for anti-clockwise (inverted) servos
            inverted = self.config.SERVO_INVERTED.get(name, False)
            actual_angle = (180 - angle) if inverted else angle

            if self.kit is not None:
                try:
                    self.kit.servo[channel].angle = actual_angle
                except Exception as e:
                    print(f"  ⚠️  Error setting {name} (ch{channel}) to {angle}°: {e}")
            # Always print for visibility
            # (In simulation mode this is the only feedback)

    def go_to_rest(self):
        """Return all servos to rest position."""
        self.set_angles(self.rest_angles)

    def execute_movement(self, angles_dict, hold_duration=2.0):
        """
        Execute a movement: set angles, hold, then return to rest.

        Args:
            angles_dict: servo angles for the movement
            hold_duration: seconds to hold before returning to rest
        """
        print(f"  🦾 Moving servos...")
        self._print_angles(angles_dict)
        self.set_angles(angles_dict)

        print(f"  ⏱️  Holding for {hold_duration}s...")
        time.sleep(hold_duration)

        print(f"  🔄 Returning to rest position...")
        self.go_to_rest()
        self._print_angles(self.rest_angles)

    def _print_angles(self, angles_dict):
        """Print servo angles in a readable format."""
        parts = [f"{name}={angles_dict.get(name, 0):3d}°"
                 for name in self.channels.keys()]
        print(f"     [{', '.join(parts)}]")


# =============================================================================
# SAMPLE MANAGER
# =============================================================================
class SampleManager:
    """
    Manages pre-stored sEMG samples for each movement.
    Provides circular selection (so it's not the same sample every time).
    """

    def __init__(self, samples_path):
        print(f"Loading samples: {samples_path}")
        data = np.load(samples_path)

        self.samples = data['samples']               # (29, 10, 50, 10)
        self.encoded_labels = data['encoded_labels']  # (29,)
        self.original_labels = data['original_labels'] # (29,)
        self.scaler_mean = data['scaler_mean']        # (10,)
        self.scaler_scale = data['scaler_scale']      # (10,)

        self.n_movements = self.samples.shape[0]
        self.n_samples_per_movement = self.samples.shape[1]

        # Circular counters — one per movement
        self.counters = np.zeros(self.n_movements, dtype=int)

        print(f"✅ Samples loaded:")
        print(f"   Movements: {self.n_movements}")
        print(f"   Samples per movement: {self.n_samples_per_movement}")
        print(f"   Window shape: {self.samples.shape[2:]}")

    def get_next_sample(self, movement_index):
        """
        Get the next sample for a movement (circular).

        Args:
            movement_index: int (0-28)

        Returns:
            numpy array (50, 10) — raw sEMG window
        """
        idx = self.counters[movement_index]
        sample = self.samples[movement_index, idx]

        # Advance circular counter
        self.counters[movement_index] = (idx + 1) % self.n_samples_per_movement

        return sample


# =============================================================================
# MAIN APPLICATION
# =============================================================================
class ProstheticHand:
    """Main application that ties everything together."""

    def __init__(self):
        print("=" * 60)
        print("🦾 PROSTHETIC HAND CONTROL SYSTEM")
        print("=" * 60)
        print()

        self.config = Config()

        # Verify files exist
        self._check_files()

        # Initialize components
        self.sample_mgr = SampleManager(self.config.SAMPLES_PATH)
        self.preprocessor = Preprocessor(
            self.config,
            self.sample_mgr.scaler_mean,
            self.sample_mgr.scaler_scale,
        )
        self.engine = InferenceEngine(self.config.MODEL_PATH)
        self.servo = ServoController(self.config)

        # Self-test
        self._self_test()

        print()
        print("=" * 60)
        print("✅ System ready!")
        print("=" * 60)

    def _check_files(self):
        """Verify required files exist."""
        if not os.path.exists(self.config.MODEL_PATH):
            print(f"❌ Model not found: {self.config.MODEL_PATH}")
            sys.exit(1)
        if not os.path.exists(self.config.SAMPLES_PATH):
            print(f"❌ Samples not found: {self.config.SAMPLES_PATH}")
            sys.exit(1)

    def _self_test(self):
        """Run a quick inference to verify the model works."""
        print("\n--- Self-test ---")
        test_sample = self.sample_mgr.get_next_sample(0)
        processed = self.preprocessor.process(test_sample)
        result = self.engine.predict(processed)
        print(f"  Test inference: class={result['predicted_class']}, "
              f"conf={result['confidence']:.3f}, "
              f"time={result['inference_time_ms']:.1f}ms")
        # Reset counter
        self.sample_mgr.counters[0] = 0
        print("  ✅ Self-test passed!")

    def show_menu(self):
        """Display the movement selection menu."""
        print()
        print("─" * 60)
        print("  AVAILABLE MOVEMENTS")
        print("─" * 60)
        for i, mov in enumerate(MOVEMENTS):
            print(f"  [{i + 1:2d}] {mov['name']}")
        print(f"  [ 0] Exit")
        print("─" * 60)

    def run(self):
        """Main loop."""
        # Set servos to rest on startup
        print("\n🔄 Setting servos to rest position...")
        self.servo.go_to_rest()

        while True:
            self.show_menu()

            try:
                choice = input("\n  Select movement (0 to exit): ").strip()
                if not choice:
                    continue
                choice = int(choice)
            except ValueError:
                print("  ⚠️  Please enter a valid number.")
                continue
            except (KeyboardInterrupt, EOFError):
                print("\n\n  Shutting down...")
                break

            if choice == 0:
                print("\n  Shutting down...")
                break

            if choice < 1 or choice > len(MOVEMENTS):
                print(f"  ⚠️  Please enter a number between 0 and {len(MOVEMENTS)}.")
                continue

            movement_index = choice - 1
            movement = MOVEMENTS[movement_index]

            print()
            print(f"  ━━━ Selected: [{choice}] {movement['name']} ━━━")
            print(f"  (Original label: {movement['original']}, "
                  f"Encoded label: {movement['encoded']})")

            # Step 1: Get next sample (circular)
            sample_idx = self.sample_mgr.counters[movement_index]
            raw_window = self.sample_mgr.get_next_sample(movement_index)
            print(f"  📊 Using sample {sample_idx + 1}/{self.config.SAMPLES_PER_MOVEMENT}")

            # Step 2: Preprocess
            processed_window = self.preprocessor.process(raw_window)

            # Step 3: Run inference
            result = self.engine.predict(processed_window)
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            inference_time = result['inference_time_ms']

            predicted_movement = MOVEMENTS[predicted_class]
            print(f"  🧠 Model prediction: [{predicted_class + 1}] "
                  f"{predicted_movement['name']} "
                  f"(conf: {confidence:.1%}, time: {inference_time:.1f}ms)")

            # Step 4: Get servo angles for the PREDICTED movement
            target_angles = predicted_movement['angles']

            # Step 5: Execute movement (set angles → hold → rest)
            self.servo.execute_movement(
                target_angles,
                hold_duration=self.config.HOLD_DURATION,
            )

            print(f"  ✅ Movement complete!")

        # Final cleanup
        print("\n🔄 Returning to rest position...")
        self.servo.go_to_rest()
        print("👋 Goodbye!")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = ProstheticHand()
    app.run()
