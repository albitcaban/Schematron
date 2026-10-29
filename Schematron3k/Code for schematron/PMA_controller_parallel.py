#!/usr/bin/env python3
"""
PMA behavioral controller + Arduino event logger + Point Grey/FLIR recorder
using pyflycap2.

PARALLELIZED VERSION
--------------------
This version separates the major jobs so they do not block one another:

1. CAMERA ACQUISITION THREAD
   - waits for each triggered frame
   - copies frame immediately
   - timestamps immediately
   - pushes video frame into video queue
   - pushes timing record into timing queue
   - updates a single "latest preview frame"

2. VIDEO WRITER THREAD
   - ONLY writes grayscale frames to AVI

3. TIMING LOGGER THREAD
   - ONLY writes camera frame timing rows to camera_frames.csv

4. PREVIEW THREAD
   - displays only the newest available frame at a low rate
   - skipping preview frames does NOT affect recording

5. SERIAL THREAD
   - logs Arduino behavioral events

The Arduino remains the master behavioral clock.

Outputs
-------
video.avi
behavioral_events.csv
camera_frames.csv
ttl_events.csv
raw_serial_log.txt
session_metadata.json

Before running
--------------
Configure camera in FlyCapture2 Viewer:
    Trigger ON
    Mode 0
    Source GPIO0 / Line0
    Active-high / rising-edge
    Shutter short enough for 30 fps

Then CLOSE FlyCapture2 Viewer before starting this script.

Example
-------
python PMA_controller_parallel.py ^
    --animal Mouse12 ^
    --protocol tone ^
    --port COM4 ^
    --output "D:\\PMA_Data"

Optional:
    --no-preview

During a run:
    type STOP and press Enter
"""

import argparse
import csv
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial
from serial.tools import list_ports

try:
    from pyflycap2.interface import Camera
except ImportError:
    Camera = None

BAUD = 115200
CAMERA_FPS = 30.0
CAMERA_FRAME_PERIOD_US = int(round(1_000_000 / CAMERA_FPS))
VIDEO_QUEUE_MAX = 900
TIMING_QUEUE_MAX = 10000
PREVIEW_FPS = 5.0
PREVIEW_PERIOD_S = 1.0 / PREVIEW_FPS

TTL_EVENTS = {
    "AUDIO_TRIGGER", "SHOCK_ON", "SHOCK_OFF", "CAMERA_START",
    "CAMERA_STOP", "TONE_ON", "TONE_OFF", "LIGHT_ON", "LIGHT_OFF",
    "VALVE_ON", "VALVE_OFF",
}
STOP_EVENTS = {"EXPERIMENT_END", "EXPERIMENT_ABORT"}

def iso_now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")

def find_arduino_port():
    ports = list(list_ports.comports())
    preferred, fallback = [], []
    for p in ports:
        text = " ".join(str(x or "") for x in [p.device, p.description, p.manufacturer]).lower()
        if "arduino" in text or "uno" in text:
            preferred.append(p.device)
        elif "usbmodem" in text or "usbserial" in text or "com" in text:
            fallback.append(p.device)
    if preferred:
        return preferred[0]
    if len(fallback) == 1:
        return fallback[0]
    return None

def make_session_folder(root, animal_id, protocol):
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{animal_id}_{protocol.lower()}PMA_{stamp}"
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder = root / name
    folder.mkdir(parents=False, exist_ok=False)
    return folder

def open_pyflycap2_camera(camera_index=0):
    if Camera is None:
        raise RuntimeError("pyflycap2 could not be imported.")
    cam = Camera(index=camera_index, context_type="IIDC")
    cam.connect()
    try:
        cam.set_drop_mode(False)
    except Exception:
        pass
    cam.start_capture()
    return cam

def image_info_to_frame_copy(image_info):
    rows = int(image_info["rows"])
    cols = int(image_info["cols"])
    stride = int(image_info["stride"])
    pix_fmt = str(image_info["pix_fmt"]).lower()
    buffer = image_info["buffer"]

    if pix_fmt in ("mono8", "raw8"):
        raw = np.frombuffer(buffer, dtype=np.uint8)
        needed = rows * stride
        frame = raw[:needed].reshape(rows, stride)[:, :cols]
        return frame.copy()

    if pix_fmt in ("mono16", "raw16", "s_mono16"):
        raw16 = np.frombuffer(buffer, dtype=np.uint16)
        stride_pixels = stride // 2 if stride >= cols * 2 else cols
        frame16 = raw16[:rows * stride_pixels].reshape(rows, stride_pixels)[:, :cols]
        return (frame16 >> 8).astype(np.uint8, copy=True)

    raise RuntimeError(f"Unsupported camera pixel format '{pix_fmt}'. Configure FlyCapture2 for Mono8.")

def extract_camera_timestamp(image_info):
    out = {
        "camera_seconds": "",
        "camera_microseconds": "",
        "camera_cycle_seconds": "",
        "camera_cycle_count": "",
        "camera_cycle_offset": "",
    }
    ts = image_info.get("ts")
    if ts is None:
        return out
    out["camera_seconds"] = getattr(ts, "seconds", "")
    out["camera_microseconds"] = getattr(ts, "micro_seconds", "")
    out["camera_cycle_seconds"] = getattr(ts, "cycle_seconds", "")
    out["camera_cycle_count"] = getattr(ts, "cycle_count", "")
    out["camera_cycle_offset"] = getattr(ts, "cycle_offset", "")
    return out

class SessionLogger:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.behavior_lock = threading.Lock()

        self.behavior_file = open(self.folder / "behavioral_events.csv", "w", newline="", buffering=1, encoding="utf-8")
        self.behavior_writer = csv.writer(self.behavior_file)
        self.behavior_writer.writerow(["arduino_ms","event","trial","host_monotonic_s","host_time_iso","raw_serial"])

        self.ttl_file = open(self.folder / "ttl_events.csv", "w", newline="", buffering=1, encoding="utf-8")
        self.ttl_writer = csv.writer(self.ttl_file)
        self.ttl_writer.writerow(["arduino_ms","event","trial","host_monotonic_s","host_time_iso"])

        self.frame_file = open(self.folder / "camera_frames.csv", "w", newline="", buffering=1, encoding="utf-8")
        self.frame_writer = csv.writer(self.frame_file)
        self.frame_writer.writerow([
            "frame_index","host_capture_monotonic_s","host_elapsed_from_first_frame_s",
            "host_capture_time_iso","host_interframe_ms","host_instantaneous_fps",
            "camera_seconds","camera_microseconds","camera_cycle_seconds","camera_cycle_count",
            "camera_cycle_offset","rows","cols","pixel_format","video_queue_size_at_capture",
            "timing_queue_size_at_capture","nominal_arduino_trigger_us_if_no_drops"
        ])

        self.raw_serial_file = open(self.folder / "raw_serial_log.txt", "w", buffering=1, encoding="utf-8")

    def log_raw_serial(self, host_iso, raw):
        with self.behavior_lock:
            self.raw_serial_file.write(f"{host_iso}\t{raw}\n")

    def log_behavior(self, arduino_ms, event, trial, host_monotonic_s, host_time_iso, raw):
        with self.behavior_lock:
            self.behavior_writer.writerow([arduino_ms,event,trial,f"{host_monotonic_s:.9f}",host_time_iso,raw])
            if event in TTL_EVENTS:
                self.ttl_writer.writerow([arduino_ms,event,trial,f"{host_monotonic_s:.9f}",host_time_iso])

    def log_frame_row(self, row):
        self.frame_writer.writerow(row)

    def close(self):
        for f in [self.behavior_file, self.ttl_file, self.frame_file, self.raw_serial_file]:
            try:
                f.flush()
                f.close()
            except Exception:
                pass

class PMASession:
    def __init__(self, serial_port, protocol, animal_id, output_root, camera_index=0, use_camera=True, show_preview=True):
        self.serial_port = serial_port
        self.protocol = protocol.upper()
        self.animal_id = animal_id
        self.output_root = output_root
        self.camera_index = camera_index
        self.use_camera = use_camera
        self.show_preview = show_preview

        self.folder = make_session_folder(output_root, animal_id, self.protocol)
        self.logger = SessionLogger(self.folder)

        self.ser = None
        self.cam = None
        self.video_writer = None

        self.stop_event = threading.Event()
        self.experiment_finished = threading.Event()
        self.camera_stop_requested = threading.Event()
        self.camera_done = threading.Event()

        self.video_queue = queue.Queue(maxsize=VIDEO_QUEUE_MAX)
        self.timing_queue = queue.Queue(maxsize=TIMING_QUEUE_MAX)

        self.serial_thread = None
        self.camera_thread = None
        self.video_thread = None
        self.timing_thread = None
        self.preview_thread = None
        self.input_thread = None

        self.camera_start_arduino_ms = None
        self.frames_acquired = 0
        self.frames_written = 0
        self.video_queue_overflow_frames = 0
        self.timing_queue_overflow_rows = 0

        self.first_frame_host = None
        self.previous_frame_host = None

        self.latest_preview_frame = None
        self.latest_preview_frame_index = 0
        self.preview_lock = threading.Lock()

        self.camera_error = None
        self.video_error = None
        self.timing_error = None
        self.preview_error = None

        self.session_start_iso = iso_now()

    def save_metadata(self):
        metadata = {
            "animal_id": self.animal_id,
            "protocol": self.protocol,
            "session_start_iso": self.session_start_iso,
            "serial_port": self.serial_port,
            "serial_baud": BAUD,
            "camera_enabled": self.use_camera,
            "camera_index": self.camera_index,
            "camera_library": "pyflycap2",
            "camera_target_fps": CAMERA_FPS,
            "camera_drop_mode": False,
            "video_queue_max": VIDEO_QUEUE_MAX,
            "timing_queue_max": TIMING_QUEUE_MAX,
            "video_mode": "grayscale_mjpg_avi",
            "live_preview": self.show_preview,
            "preview_target_fps": PREVIEW_FPS if self.show_preview else None,
            "output_folder": str(self.folder),
        }
        with open(self.folder / "session_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def open_serial(self):
        self.ser = serial.Serial(self.serial_port, BAUD, timeout=0.1)
        time.sleep(2.0)
        self.ser.reset_input_buffer()

    def send(self, command):
        self.ser.write((command.strip() + "\n").encode("ascii"))
        self.ser.flush()
        print(f"PYTHON  > {command}")

    def serial_loop(self):
        while not self.stop_event.is_set():
            try:
                raw_bytes = self.ser.readline()
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"\nSerial read error: {exc}")
                return
            if not raw_bytes:
                continue
            host_mono = time.perf_counter()
            host_iso = iso_now()
            raw = raw_bytes.decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            print(f"ARDUINO > {raw}")
            self.logger.log_raw_serial(host_iso, raw)
            parts = [x.strip() for x in raw.split(",")]
            try:
                arduino_ms = int(parts[0])
            except (ValueError, IndexError):
                continue
            event = parts[1] if len(parts) >= 2 else ""
            trial = parts[2] if len(parts) >= 3 else ""
            self.logger.log_behavior(arduino_ms, event, trial, host_mono, host_iso, raw)
            if event == "CAMERA_START":
                self.camera_start_arduino_ms = arduino_ms
            if event in STOP_EVENTS:
                self.experiment_finished.set()

    def input_loop(self):
        print("\nType STOP and press Enter at any time to abort cleanly.\n")
        while not self.stop_event.is_set():
            try:
                command = input().strip().upper()
            except (EOFError, KeyboardInterrupt):
                return
            if command in {"STOP", "ABORT", "Q", "QUIT"}:
                print("\nUser requested experiment stop.")
                try:
                    self.send("ABORT")
                except Exception as exc:
                    print(f"Could not send ABORT to Arduino: {exc}")
                if not self.experiment_finished.wait(timeout=1.0):
                    self.experiment_finished.set()
                return
            if command:
                print(f"Unknown command '{command}'. Type STOP and press Enter to abort.")

    def open_camera(self):
        self.cam = open_pyflycap2_camera(self.camera_index)
        print(f"Camera connected: serial={getattr(self.cam, 'serial', 'unknown')}, interface={getattr(self.cam, 'interface_type', 'unknown')}")

    def camera_acquisition_loop(self):
        try:
            while not self.camera_stop_requested.is_set():
                self.cam.read_next_image()
                if self.camera_stop_requested.is_set():
                    break

                capture_host = time.perf_counter()
                capture_iso = iso_now()
                image_info = self.cam.get_current_image()
                frame_gray = image_info_to_frame_copy(image_info)
                camera_ts = extract_camera_timestamp(image_info)

                self.frames_acquired += 1
                frame_index = self.frames_acquired

                if self.first_frame_host is None:
                    self.first_frame_host = capture_host

                if self.previous_frame_host is None:
                    interframe_ms = ""
                    instantaneous_fps = ""
                else:
                    dt = capture_host - self.previous_frame_host
                    interframe_ms = dt * 1000.0
                    instantaneous_fps = (1.0 / dt) if dt > 0 else ""

                self.previous_frame_host = capture_host

                if self.camera_start_arduino_ms is not None:
                    nominal_trigger_us = int(self.camera_start_arduino_ms * 1000) + (frame_index - 1) * CAMERA_FRAME_PERIOD_US
                else:
                    nominal_trigger_us = ""

                try:
                    self.video_queue.put_nowait((frame_index, frame_gray))
                except queue.Full:
                    self.video_queue_overflow_frames += 1
                    if self.video_queue_overflow_frames == 1:
                        print("\nWARNING: VIDEO QUEUE FILLED. Video writer is not keeping up.")

                elapsed = capture_host - self.first_frame_host
                timing_row = [
                    frame_index, f"{capture_host:.9f}", f"{elapsed:.9f}", capture_iso,
                    f"{interframe_ms:.6f}" if isinstance(interframe_ms, float) else "",
                    f"{instantaneous_fps:.6f}" if isinstance(instantaneous_fps, float) else "",
                    camera_ts["camera_seconds"], camera_ts["camera_microseconds"],
                    camera_ts["camera_cycle_seconds"], camera_ts["camera_cycle_count"],
                    camera_ts["camera_cycle_offset"], int(image_info["rows"]), int(image_info["cols"]),
                    str(image_info["pix_fmt"]), self.video_queue.qsize(), self.timing_queue.qsize(),
                    nominal_trigger_us
                ]

                try:
                    self.timing_queue.put_nowait(timing_row)
                except queue.Full:
                    self.timing_queue_overflow_rows += 1
                    if self.timing_queue_overflow_rows == 1:
                        print("\nWARNING: TIMING QUEUE FILLED. Camera timing logger is not keeping up.")

                if self.show_preview:
                    with self.preview_lock:
                        self.latest_preview_frame = frame_gray
                        self.latest_preview_frame_index = frame_index

        except Exception as exc:
            if not self.camera_stop_requested.is_set():
                self.camera_error = exc
                print(f"\nCAMERA ACQUISITION ERROR: {exc}")
                self.experiment_finished.set()
        finally:
            self.camera_done.set()

    def open_video_writer(self, frame_gray):
        height, width = frame_gray.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.video_writer = cv2.VideoWriter(
            str(self.folder / "video.avi"),
            fourcc,
            CAMERA_FPS,
            (width, height),
            False
        )
        if not self.video_writer.isOpened():
            raise RuntimeError("OpenCV could not create video.avi.")

    def video_writer_loop(self):
        try:
            while True:
                if self.camera_done.is_set() and self.video_queue.empty():
                    break
                try:
                    frame_index, frame_gray = self.video_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self.video_writer is None:
                    self.open_video_writer(frame_gray)
                self.video_writer.write(frame_gray)
                self.frames_written += 1
                self.video_queue.task_done()
        except Exception as exc:
            self.video_error = exc
            print(f"\nVIDEO WRITER ERROR: {exc}")
            self.experiment_finished.set()
        finally:
            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass

    def timing_logger_loop(self):
        try:
            while True:
                if self.camera_done.is_set() and self.timing_queue.empty():
                    break
                try:
                    row = self.timing_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                self.logger.log_frame_row(row)
                self.timing_queue.task_done()
        except Exception as exc:
            self.timing_error = exc
            print(f"\nTIMING LOGGER ERROR: {exc}")
            self.experiment_finished.set()

    def preview_loop(self):
        window_name = "PMA Camera Live View"
        try:
            last_displayed_index = -1
            while not self.stop_event.is_set():
                cycle_start = time.perf_counter()
                with self.preview_lock:
                    frame = self.latest_preview_frame
                    frame_index = self.latest_preview_frame_index

                if frame is not None and frame_index != last_displayed_index:
                    preview = cv2.resize(frame, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
                    cv2.putText(
                        preview,
                        f"{self.protocol} PMA   Frame {frame_index}   VideoQ {self.video_queue.qsize()}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        255,
                        2,
                    )
                    cv2.imshow(window_name, preview)
                    cv2.waitKey(1)
                    last_displayed_index = frame_index

                    try:
                        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                        if visible < 1:
                            print("Live preview closed. Recording continues normally.")
                            self.show_preview = False
                            break
                    except cv2.error:
                        self.show_preview = False
                        break

                elapsed = time.perf_counter() - cycle_start
                sleep_time = PREVIEW_PERIOD_S - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as exc:
            self.preview_error = exc
            print(f"\nPREVIEW ERROR: {exc}")
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def start(self):
        self.save_metadata()

        print("\n--------------------------------------------------")
        print("PMA SESSION")
        print("--------------------------------------------------")
        print(f"Animal:       {self.animal_id}")
        print(f"Protocol:     {self.protocol}")
        print(f"Serial port:  {self.serial_port}")
        print(f"Save folder:  {self.folder}")
        print(f"Camera FPS:   {CAMERA_FPS}")
        print(f"Video queue:  {VIDEO_QUEUE_MAX} frames")
        print(f"Timing queue: {TIMING_QUEUE_MAX} rows")
        print(f"Preview:      {'ON (~5 fps, separate thread)' if self.show_preview else 'OFF'}")
        print("--------------------------------------------------\n")

        print("Opening Arduino...")
        self.open_serial()

        if self.use_camera:
            print("Connecting to Point Grey camera...")
            self.open_camera()
            print("Camera is armed and waiting for Arduino frame TTLs.")

        self.serial_thread = threading.Thread(target=self.serial_loop, name="ArduinoSerial", daemon=True)
        self.serial_thread.start()

        if self.use_camera:
            self.video_thread = threading.Thread(target=self.video_writer_loop, name="VideoWriter", daemon=True)
            self.video_thread.start()

            self.timing_thread = threading.Thread(target=self.timing_logger_loop, name="TimingLogger", daemon=True)
            self.timing_thread.start()

            if self.show_preview:
                self.preview_thread = threading.Thread(target=self.preview_loop, name="Preview", daemon=True)
                self.preview_thread.start()

            self.camera_thread = threading.Thread(target=self.camera_acquisition_loop, name="CameraAcquisition", daemon=True)
            self.camera_thread.start()

        self.input_thread = threading.Thread(target=self.input_loop, name="KeyboardInput", daemon=True)
        self.input_thread.start()

        time.sleep(0.5)

        command = "START,TONE" if self.protocol == "TONE" else "START,LIGHT"
        print("\nEverything is armed.")
        print(f"Sending {command} to Arduino...\n")
        self.send(command)

        try:
            while True:
                if self.experiment_finished.wait(timeout=0.25):
                    break
                if self.camera_error is not None or self.video_error is not None or self.timing_error is not None:
                    break
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected.")
            try:
                self.send("ABORT")
                time.sleep(0.5)
            except Exception:
                pass
        finally:
            self.stop()

    def stop(self):
        print("\nStopping session cleanly...")
        self.stop_event.set()
        self.camera_stop_requested.set()

        if self.cam is not None:
            try:
                self.cam.stop_capture()
            except Exception:
                pass

        if self.camera_thread is not None:
            self.camera_thread.join(timeout=5.0)

        self.camera_done.set()

        if self.video_thread is not None:
            print(f"Draining video queue... {self.video_queue.qsize()} currently queued.")
            self.video_thread.join(timeout=30.0)

        if self.timing_thread is not None:
            print(f"Draining timing queue... {self.timing_queue.qsize()} currently queued.")
            self.timing_thread.join(timeout=10.0)

        if self.preview_thread is not None:
            self.preview_thread.join(timeout=2.0)

        if self.cam is not None:
            try:
                self.cam.disconnect()
            except Exception:
                pass

        time.sleep(0.2)

        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass

        if self.serial_thread is not None:
            self.serial_thread.join(timeout=1.0)

        self.logger.close()

        print("\n--------------------------------------------------")
        print("SESSION CLOSED")
        print("--------------------------------------------------")
        print(f"Frames acquired from camera: {self.frames_acquired}")
        print(f"Frames written to AVI:       {self.frames_written}")
        print(f"Video queue overflow frames: {self.video_queue_overflow_frames}")
        print(f"Timing queue overflow rows:   {self.timing_queue_overflow_rows}")
        print(f"Data saved to: {self.folder}")

        if self.first_frame_host is not None and self.previous_frame_host is not None and self.frames_acquired > 1:
            duration = self.previous_frame_host - self.first_frame_host
            measured_fps = (self.frames_acquired - 1) / duration if duration > 0 else 0.0
            print(f"Measured acquisition rate:    {measured_fps:.3f} fps")

        if self.camera_error is not None:
            print(f"Camera error: {self.camera_error}")
        if self.video_error is not None:
            print(f"Video error: {self.video_error}")
        if self.timing_error is not None:
            print(f"Timing logger error: {self.timing_error}")
        if self.preview_error is not None:
            print(f"Preview error: {self.preview_error}")

        print("--------------------------------------------------\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", required=True)
    parser.add_argument("--protocol", required=True, choices=["tone", "light"])
    parser.add_argument("--port", default=None)
    parser.add_argument("--output", default="./PMA_sessions")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    port = args.port or find_arduino_port()
    if not port:
        print(
            '\nCould not identify Arduino COM port.\n'
            'Example:\n'
            'python PMA_controller_parallel.py --animal Mouse12 --protocol tone --port COM4 --output "D:\\PMA_Data"\n'
        )
        return 1

    session = PMASession(
        serial_port=port,
        protocol=args.protocol,
        animal_id=args.animal,
        output_root=args.output,
        camera_index=args.camera_index,
        use_camera=not args.no_camera,
        show_preview=not args.no_preview,
    )
    session.start()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
