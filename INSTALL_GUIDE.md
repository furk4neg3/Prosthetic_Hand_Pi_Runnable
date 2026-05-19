# 🦾 Prosthetic Hand — Raspberry Pi Installation Guide

## Prerequisites

- **Raspberry Pi 5** (2GB+ RAM) with Raspberry Pi OS (Bookworm, 64-bit recommended)
- **Python 3.11+** (comes with Bookworm)
- **I2C enabled** (for PCA9685 servo driver)

---

## Step 1: Enable I2C (if not already done)

```bash
sudo raspi-config
# → Interface Options → I2C → Enable
# Reboot after enabling
sudo reboot
```

Verify I2C is working:
```bash
sudo apt install -y i2c-tools
sudo i2cdetect -y 1
# You should see address 0x40 (PCA9685)
```

---

## Step 2: Install system-level dependencies

Some Python packages need C libraries to compile. Install them first:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev libatlas-base-dev libopenblas-dev
```

> **Why?**
> - `python3-venv` — needed to create virtual environments
> - `python3-dev` — C headers for building Python extensions
> - `libatlas-base-dev` / `libopenblas-dev` — needed by numpy/scipy

---

## Step 3: Create and activate a virtual environment

```bash
cd ~/Prosthetic_Hand-Runnable    # or wherever your project is
python3 -m venv servo_env
source servo_env/bin/activate
```

> ⚠️ **Always activate the venv before installing or running:**
> ```bash
> source servo_env/bin/activate
> ```

---

## Step 4: Upgrade pip first

```bash
pip install --upgrade pip setuptools wheel
```

> This avoids many "legacy setup.py" build failures.

---

## Step 5: Install requirements

```bash
pip install -r requirements.txt
```

**That's it if everything works!** But on a Raspberry Pi, some packages are likely to fail. Read on for solutions.

---

## ⚠️ Common Errors & Solutions

### Error 1: `numpy` or `scipy` fails to build (compilation errors)

**Symptom:**
```
error: subprocess-exited-with-error
Building wheel for numpy (pyproject.toml) did not run successfully.
```

**Solution — use the system packaged versions:**
```bash
# Exit venv first
deactivate

# Create venv WITH access to system packages
python3 -m venv servo_env --system-site-packages
source servo_env/bin/activate

# Install numpy and scipy from the system
sudo apt install -y python3-numpy python3-scipy

# Now install the rest
pip install -r requirements.txt
```

> When you use `--system-site-packages`, the venv can "see" the system-installed numpy/scipy,
> which are pre-compiled for your Pi's architecture.

---

### Error 2: `ai-edge-litert` has no wheel for your platform

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement ai-edge-litert
```
or
```
ERROR: No matching distribution found for ai-edge-litert
```

**Solution — use `tflite-runtime` instead:**

Edit `requirements.txt` — comment out `ai-edge-litert` and uncomment `tflite-runtime`:
```txt
# ai-edge-litert
tflite-runtime
```

Then:
```bash
pip install -r requirements.txt
```

If `tflite-runtime` also has no wheel, install from the Google Coral ppa:
```bash
# For Python 3.11 on aarch64 (Pi 5):
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime
```

**Last resort — install full TensorFlow (heavy but works):**
```bash
pip install tensorflow
```
> This is ~500MB and slow, but the code falls back to `tf.lite.Interpreter` automatically.

---

### Error 3: `lgpio` / GPIO errors when installing `adafruit-circuitpython-servokit`

**Symptom:**
```
RuntimeError: lgpio is not installed
```
or
```
ModuleNotFoundError: No module named 'lgpio'
```

**Solution:**
```bash
# Install lgpio from apt (the pip version often fails on Pi 5)
sudo apt install -y python3-lgpio

# If you're NOT using --system-site-packages, you need to manually symlink it:
# Find where lgpio is installed system-wide
python3 -c "import lgpio; print(lgpio.__file__)"
# Example output: /usr/lib/python3/dist-packages/lgpio.py

# Then symlink into your venv (adjust paths as needed)
ln -s /usr/lib/python3/dist-packages/lgpio.py servo_env/lib/python3.11/site-packages/
ln -s /usr/lib/python3/dist-packages/_lgpio*.so servo_env/lib/python3.11/site-packages/
```

> **Best approach:** Just use `--system-site-packages` when creating the venv (see Error 1 solution).
> That way lgpio is automatically available.

---

### Error 4: `adafruit-blinka` I2C permission denied

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: '/dev/i2c-1'
```

**Solution:**
```bash
# Add your user to the i2c group
sudo usermod -aG i2c $USER

# Log out and back in (or reboot)
sudo reboot
```

Or run with sudo (not recommended for long-term):
```bash
sudo servo_env/bin/python prosthetic_hand_gui.py
```

---

### Error 5: `libgpiod` errors with Adafruit libraries on Pi 5

**Symptom:**
```
ImportError: libgpiod.so.2: cannot open shared object file
```
or
```
NotImplementedError: No module named 'adafruit_blinka.microcontroller.bcm2712'
```

**Solution:**
```bash
sudo apt install -y python3-libgpiod libgpiod-dev

# Make sure Adafruit Blinka is up to date (Pi 5 support was added later)
pip install --upgrade adafruit-blinka adafruit-platformdetect
```

---

### Error 6: `flask-socketio` import error or async mode issues

**Symptom:**
```
ValueError: Invalid async_mode specified
```

**Solution:**
```bash
# Install the threading-compatible async backend
pip install simple-websocket
```

> The code uses `async_mode='threading'`, which requires the `simple-websocket` package.
> This is automatically pulled by newer flask-socketio versions, but if it's missing, install it manually.

---

## 🏁 Quick-Start (Recommended Full Flow)

If you want to avoid most of the above issues, here's the single recommended flow:

```bash
# 1. System dependencies
sudo apt update
sudo apt install -y python3-venv python3-dev python3-numpy python3-scipy \
    python3-lgpio python3-libgpiod libgpiod-dev libatlas-base-dev \
    libopenblas-dev i2c-tools

# 2. Create venv with system-site-packages
cd ~/Prosthetic_Hand-Runnable
python3 -m venv servo_env --system-site-packages
source servo_env/bin/activate

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install everything
pip install -r requirements.txt

# 5. Verify
python3 -c "import numpy; print('numpy OK:', numpy.__version__)"
python3 -c "from scipy.signal import butter; print('scipy OK')"
python3 -c "from ai_edge_litert.interpreter import Interpreter; print('TFLite OK')"
python3 -c "from flask import Flask; print('Flask OK')"
python3 -c "from adafruit_servokit import ServoKit; print('ServoKit OK')"
```

---

## ▶️ Running the Application

```bash
# Always activate venv first
source servo_env/bin/activate

# Terminal-only version
python3 prosthetic_hand.py

# Web GUI version (recommended)
python3 prosthetic_hand_gui.py
# Then open http://<pi-ip>:5000 in your browser

# Servo test script
python3 test_servos.py
```

---

## 📋 Summary of all packages

| Package | Purpose | Used by |
|---|---|---|
| `numpy` | Array operations, signal math | All scripts |
| `scipy` | Bandpass filter (`butter`, `filtfilt`) | prosthetic_hand.py, gui |
| `ai-edge-litert` | TFLite model inference (recommended) | prosthetic_hand.py, gui |
| `tflite-runtime` | TFLite inference (alternative) | prosthetic_hand.py, gui |
| `flask` | Web server for GUI | prosthetic_hand_gui.py |
| `flask-socketio` | Real-time WebSocket updates | prosthetic_hand_gui.py |
| `adafruit-circuitpython-servokit` | PCA9685 servo driver control | All scripts |
