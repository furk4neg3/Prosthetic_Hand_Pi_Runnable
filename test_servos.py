#!/usr/bin/env python3
"""
test_servos.py
==============

A simple test script to verify that all 5 servo motors work correctly
via the PCA9685 controller, both individually and together.

Usage:
  python3 test_servos.py

Tests:
  1. Each servo moves individually (0° → 90° → 180° → 0°)
  2. All servos move together (open → close → open)
  3. Interactive mode: manually set any servo to any angle
"""

import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# Try to import servo control libraries
# ─────────────────────────────────────────────────────────────────────────────
SERVO_AVAILABLE = False
try:
    from adafruit_servokit import ServoKit
    SERVO_AVAILABLE = True
except ImportError:
    print("⚠️  adafruit_servokit not found — running in SIMULATION mode")
    print("   Install with: pip install adafruit-circuitpython-servokit")


# =============================================================================
# CONFIGURATION — must match prosthetic_hand.py
# =============================================================================
PCA9685_I2C_ADDRESS = 0x40
PCA9685_FREQUENCY = 50
SERVO_MIN_PULSE = 500    # µs
SERVO_MAX_PULSE = 2500   # µs

# Servo channel assignments
# Servos 0, 3, 4 turn clockwise (normal)
# Servos 1, 2 turn anti-clockwise (inverted: angle is flipped to 180 - angle)
SERVOS = [
    {"name": "Thumb",  "channel": 0, "inverted": False},
    {"name": "Index",  "channel": 1, "inverted": False},
    {"name": "Middle", "channel": 2, "inverted": False},
    {"name": "Ring",   "channel": 3, "inverted": True},
    {"name": "Little", "channel": 4, "inverted": True},
]

# Test angles
REST_ANGLE = 0
MID_ANGLE = 90
MAX_ANGLE = 180

# Delay between movements (seconds)
STEP_DELAY = 1.0


# =============================================================================
# SERVO HELPERS
# =============================================================================
kit = None


def init_servos():
    """Initialize the PCA9685 and configure all servo channels."""
    global kit
    if not SERVO_AVAILABLE:
        print("  [SIM] Servos initialized (simulation)")
        return

    try:
        kit = ServoKit(
            channels=16,
            address=PCA9685_I2C_ADDRESS,
            frequency=PCA9685_FREQUENCY,
        )
        for servo in SERVOS:
            kit.servo[servo["channel"]].set_pulse_width_range(
                SERVO_MIN_PULSE, SERVO_MAX_PULSE
            )
        print("✅ PCA9685 initialized, all servo channels configured")
    except Exception as e:
        print(f"❌ Failed to initialize PCA9685: {e}")
        print("   Falling back to SIMULATION mode")
        kit = None


def set_servo(channel, angle, name=""):
    """Set a single servo to a given angle, respecting inversion."""
    angle = max(0, min(180, angle))

    # Find if this servo is inverted
    inverted = False
    for servo in SERVOS:
        if servo["channel"] == channel:
            inverted = servo.get("inverted", False)
            break

    # Flip angle for anti-clockwise servos
    actual_angle = (180 - angle) if inverted else angle

    inv_label = " [inv]" if inverted else ""
    label = f"{name} (ch{channel}){inv_label}" if name else f"ch{channel}{inv_label}"
    print(f"    {label} → {angle:3d}° (actual: {actual_angle:3d}°)")

    if kit is not None:
        try:
            kit.servo[channel].angle = actual_angle
        except Exception as e:
            print(f"    ⚠️  Error: {e}")


def set_all_servos(angle):
    """Set all servos to the same angle."""
    for servo in SERVOS:
        set_servo(servo["channel"], angle, servo["name"])


# =============================================================================
# TEST 1: Individual servo test
# =============================================================================
def test_individual():
    """Test each servo one by one: 0° → 90° → 180° → 0°."""
    print()
    print("=" * 50)
    print("  TEST 1: Individual Servo Test")
    print("  Each servo: 0° → 90° → 180° → 0°")
    print("=" * 50)

    for servo in SERVOS:
        name = servo["name"]
        ch = servo["channel"]

        print(f"\n  ── {name} (channel {ch}) ──")

        print(f"  → Moving to {REST_ANGLE}°")
        set_servo(ch, REST_ANGLE, name)
        time.sleep(STEP_DELAY)

        print(f"  → Moving to {MID_ANGLE}°")
        set_servo(ch, MID_ANGLE, name)
        time.sleep(STEP_DELAY)

        print(f"  → Moving to {MAX_ANGLE}°")
        set_servo(ch, MAX_ANGLE, name)
        time.sleep(STEP_DELAY)

        print(f"  → Returning to {REST_ANGLE}°")
        set_servo(ch, REST_ANGLE, name)
        time.sleep(STEP_DELAY)

        print(f"  ✅ {name} OK")

    print("\n✅ Individual test complete!")


# =============================================================================
# TEST 2: All servos together
# =============================================================================
def test_all_together():
    """Test all servos moving together: open → close → open."""
    print()
    print("=" * 50)
    print("  TEST 2: All Servos Together")
    print("  All at once: 0° → 90° → 180° → 0°")
    print("=" * 50)

    print(f"\n  → All to {REST_ANGLE}° (open)")
    set_all_servos(REST_ANGLE)
    time.sleep(STEP_DELAY * 1.5)

    print(f"\n  → All to {MID_ANGLE}° (half close)")
    set_all_servos(MID_ANGLE)
    time.sleep(STEP_DELAY * 1.5)

    print(f"\n  → All to {MAX_ANGLE}° (full close)")
    set_all_servos(MAX_ANGLE)
    time.sleep(STEP_DELAY * 1.5)

    print(f"\n  → All to {REST_ANGLE}° (open)")
    set_all_servos(REST_ANGLE)
    time.sleep(STEP_DELAY)

    print("\n✅ All-together test complete!")


# =============================================================================
# TEST 3: Interactive mode
# =============================================================================
def test_interactive():
    """Let the user manually control any servo."""
    print()
    print("=" * 50)
    print("  TEST 3: Interactive Mode")
    print("  Manually set any servo to any angle")
    print("=" * 50)

    print("\n  Commands:")
    print("    <servo#> <angle>  — Set servo to angle (e.g. '1 90')")
    print("    all <angle>       — Set ALL servos to angle (e.g. 'all 45')")
    print("    rest              — All servos to 0°")
    print("    q                 — Quit interactive mode")
    print()
    print("  Servo numbers:")
    for servo in SERVOS:
        print(f"    {servo['channel']} = {servo['name']}")
    print()

    while True:
        try:
            cmd = input("  > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if not cmd:
            continue

        if cmd == "q" or cmd == "quit" or cmd == "exit":
            break

        if cmd == "rest":
            print("  → All to rest (0°)")
            set_all_servos(REST_ANGLE)
            continue

        parts = cmd.split()
        if len(parts) != 2:
            print("  ⚠️  Usage: <servo#> <angle>  or  all <angle>  or  rest  or  q")
            continue

        try:
            angle = int(parts[1])
        except ValueError:
            print("  ⚠️  Angle must be a number (0-180)")
            continue

        if angle < 0 or angle > 180:
            print("  ⚠️  Angle must be between 0 and 180")
            continue

        if parts[0] == "all":
            print(f"  → All servos to {angle}°")
            set_all_servos(angle)
        else:
            try:
                ch = int(parts[0])
            except ValueError:
                print("  ⚠️  Servo must be a number (0-4)")
                continue

            if ch < 0 or ch > 4:
                print("  ⚠️  Servo must be between 0 and 4")
                continue

            name = SERVOS[ch]["name"]
            print(f"  → {name} to {angle}°")
            set_servo(ch, angle, name)

    # Return to rest on exit
    print("\n  → Returning all to rest")
    set_all_servos(REST_ANGLE)
    print("  ✅ Interactive mode ended")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print()
    print("🦾 PROSTHETIC HAND — SERVO TEST")
    print("=" * 50)

    init_servos()

    # Return all to rest first
    print("\n🔄 Setting all servos to rest position...")
    set_all_servos(REST_ANGLE)
    time.sleep(0.5)

    while True:
        print()
        print("─" * 50)
        print("  Select a test:")
        print("  [1] Individual servo test (one by one)")
        print("  [2] All servos together")
        print("  [3] Interactive mode (manual control)")
        print("  [0] Exit")
        print("─" * 50)

        try:
            choice = input("  Choice: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice == "1":
            test_individual()
        elif choice == "2":
            test_all_together()
        elif choice == "3":
            test_interactive()
        elif choice == "0":
            break
        else:
            print("  ⚠️  Enter 0, 1, 2, or 3")

    # Final cleanup
    print("\n🔄 Returning all servos to rest...")
    set_all_servos(REST_ANGLE)
    print("👋 Done!")


if __name__ == "__main__":
    main()