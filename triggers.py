"""
triggers.py

EEG event markers. Every marker is sent on the screen flip it belongs to and
written to an events CSV at the same moment, so the behavioural log and the
EEG recording share one set of event names and codes.

Usage inside a routine:
    markers.send_on_next_flip("recall_cue", category="face", detail=image_file)
    win.flip()
"""

import csv

from psychopy import core, logging

import settings


# -------------------------------------------------------------- codes ---
# 8-bit codes. Image onsets carry the category in the last digit
# (1 face, 2 object, 3 scene, 4 word, 5 body).

CATEGORY_DIGITS = {"face": 1, "object": 2, "scene": 3, "word": 4, "body": 5}

IMAGE_EVENT_BASES = {
    "study_image": 10,
    "recall_cue": 60,
}

EVENT_CODES = {
    "study_fixation": 16,
    "shades_start": 50,
    "shades_lighter": 51,
    "shades_darker": 52,
    "shades_end": 53,
    "recall_fixation": 70,
    "recall_offset": 77,
    "recall_offset_fixation": 78,
    "memory_judgement_blank": 71,
    "recall_remember": 72,
    "recall_forgotten": 73,
    "memory_timeout": 74,
    "choice_screen": 75,
    "choice_response": 76,
    "choice_timeout": 79,
    "block_start": 100,
    "block_end": 101,
    "break_start": 102,
    "break_end": 103,
    "stage_skipped": 104,
    "session_start": 250,
    "session_end": 251,
}


def get_event_code(event_name, category=None):
    if event_name in IMAGE_EVENT_BASES:
        return IMAGE_EVENT_BASES[event_name] + CATEGORY_DIGITS[category]
    return EVENT_CODES[event_name]


# -------------------------------------------------------------- ports ---

class TriggerPort:
    """Sends a short pulse on the configured port; 'none' only logs."""

    def __init__(self, backend):
        self.backend = backend
        self.port = None

        if backend == "parallel":
            from psychopy import parallel
            self.port = parallel.ParallelPort(address=settings.PARALLEL_PORT_ADDRESS)
            self.port.setData(0)
        elif backend == "serial":
            import serial
            self.port = serial.Serial(settings.SERIAL_PORT, baudrate=settings.SERIAL_BAUDRATE)
            self.port.write(bytes([0]))
        elif backend != "none":
            raise ValueError(f"Unknown trigger backend '{backend}'")

    def pulse(self, code):
        if self.backend == "parallel":
            self.port.setData(code)
            core.wait(settings.TRIGGER_PULSE_DURATION, hogCPUperiod=settings.TRIGGER_PULSE_DURATION)
            self.port.setData(0)
        elif self.backend == "serial":
            self.port.write(bytes([code]))
            core.wait(settings.TRIGGER_PULSE_DURATION, hogCPUperiod=settings.TRIGGER_PULSE_DURATION)
            self.port.write(bytes([0]))

    def close(self):
        if self.backend == "serial" and self.port is not None:
            self.port.close()


# ------------------------------------------------------------ markers ---

class EventMarkers:
    def __init__(self, win, events_file_path, backend=None):
        self.win = win
        self.port = TriggerPort(backend or settings.TRIGGER_BACKEND)
        self.context = {}

        self.events_file = open(events_file_path, "w", newline="")
        self.writer = csv.DictWriter(
            self.events_file,
            fieldnames=["time", "event", "code", "category", "detail", "phase", "session", "cycle", "trial"],
        )
        self.writer.writeheader()

    def set_context(self, **context):
        """Phase/session/cycle/trial labels written alongside every following event."""
        self.context.update(context)

    def send_now(self, event_name, category=None, detail=""):
        code = get_event_code(event_name, category)
        event_time = core.getTime()
        self.port.pulse(code)
        self.writer.writerow({
            "time": f"{event_time:.6f}",
            "event": event_name,
            "code": code,
            "category": category or "",
            "detail": detail,
            "phase": self.context.get("phase", ""),
            "session": self.context.get("session", ""),
            "cycle": self.context.get("cycle", ""),
            "trial": self.context.get("trial", ""),
        })
        self.events_file.flush()
        logging.data(f"EVENT {event_name} code={code} {detail}")

    def send_on_next_flip(self, event_name, category=None, detail=""):
        self.win.callOnFlip(self.send_now, event_name, category, detail)

    def close(self):
        self.port.close()
        self.events_file.close()
