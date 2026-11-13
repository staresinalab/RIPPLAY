"""
settings.py

Every timing, key, size and switch for cued_recall.py lives here. Design
parameters (offsets, runs, seeds, trial-order constraints) live in
make_conditions.py.
"""

from pathlib import Path


# ---------------------------------------------------------------- files ---

# The folder that contains stimuli/. Image paths in the stimulus list
# ("stimuli/f (03).jpg") are relative to it. "." means stimuli/ sits inside
# this task folder, next to cued_recall.py.
STIMULUS_ROOT = Path(".")
STIMULUS_LIST_FILE = STIMULUS_ROOT / "stimuli" / "stimulus_list.csv"

ASSIGNMENTS_FILE = Path("design/ripplay_25_participant_assignments_v3.csv")
CONDITIONS_FOLDER = Path("conditions")
DATA_FOLDER = Path("data")


# -------------------------------------------------------------- display ---

FULLSCREEN = True
WINDOW_SIZE = (1280, 800)        # used when not fullscreen
SCREEN_NUMBER = 0
MONITOR_NAME = "testMonitor"

BACKGROUND_COLOR = [-0.5, -0.5, -0.5]   # dark grey, so the mid-grey cross can brighten and darken
TEXT_COLOR = [1, 1, 1]
TEXT_HEIGHT = 0.035
FIXATION_HEIGHT = 0.06
FIXATION_COLOR = [0, 0, 0]              # mid grey


# ---------------------------------------------------------------- study ---
# Evening only: the 50 images in the participant's fixed order, once per
# cycle, before the shades-of-grey interval and the recall run.

STUDY_IMAGE_DURATION = 3.0
# jittered fixation between study images, so image onsets are aperiodic
# (stimulus-onset asynchrony 4-5 s) and cannot entrain ongoing oscillations
STUDY_FIXATION_RANGE = (1.0, 2.0)
STUDY_IMAGE_SIZE = (0.5, 0.5)           # height units


# ------------------------------------------------------- shades of grey ---
# Two minutes between every EVENING study block and its recall run
# (potential awake replay). Never before morning recall, per the study design.

RUN_SHADES_BEFORE_EVENING_RECALL = True
SHADES_DURATION = 120.0
SHADES_EVENT_DURATION = 0.3
SHADES_MIN_GAP = 6.0             # seconds between event onsets (sparse and jittered)
SHADES_MAX_GAP = 16.0
SHADES_EDGE_BUFFER = 4.0         # no events in the first/last few seconds
SHADES_LIGHTER_PROBABILITY = 0.5
SHADES_LIGHTER_COLOR = [0.5, 0.5, 0.5]
SHADES_DARKER_COLOR = [-0.3, -0.3, -0.3]


# ------------------------------------------------------------- cue ---

# OFFSET_ON_CUE = True (default): the signed offset sits on a translucent
# mask at the centre of the cue image, so number and image are ONE event
# (one EEG marker, one onset) and their evoked responses cannot blur into
# each other:
#     jittered fixation -> cue image + mask + offset number (CUE_DURATION)
# OFFSET_ON_CUE = False: the earlier sequence, number first, then the image:
#     jittered fixation -> offset number (OFFSET_DURATION) -> jittered fixation
#     (OFFSET_TO_CUE_RANGE) -> cue image (CUE_DURATION)
# which keeps a number-only period as a control for the left/right semantics
# of the - and + signs in gaze, at the cost of two overlapping EEG events.
OFFSET_ON_CUE = True
RECALL_ITI_RANGE = (1.0, 1.5)        # jittered fixation before each cue
OFFSET_DURATION = 0.5                # number alone (OFFSET_ON_CUE = False only)
OFFSET_TO_CUE_RANGE = (0.5, 0.8)     # fixation between number and image (OFFSET_ON_CUE = False only)
CUE_DURATION = 0.5                   # cue image (with the number on it when OFFSET_ON_CUE)
CUE_IMAGE_SIZE = (0.4, 0.4)

# translucent mask behind the number, so it reads against any image without hiding it
OFFSET_MASK_RADIUS = 0.085           # height units (the cue image is 0.4 wide)
OFFSET_MASK_COLOR = [1, 1, 1]        # white, as specified; see OFFSET_MASK_OPACITY
OFFSET_MASK_OPACITY = 0.5            # 0 = invisible, 1 = solid; the image shows through below 1

# A true minus sign (U+2212) has the same width as "+", keeping "-2" and "+2"
# visually symmetric.
OFFSET_TEXT_HEIGHT = 0.12
OFFSET_TEXT_COLOR = [1, 1, 1]        # white on the dark grey background
MINUS_SIGN = "−"


# --------------------------------------------- remembered / forgotten ---
# After the cue the screen goes blank (no text) until the participant presses
# R or F, or MEMORY_RESPONSE_TIMEOUT runs out. This blank is the retrieval /
# awake-replay window; the choice grid follows on the next frame after the key
# press (or the timeout), with no timed blank. A timeout leaves memory_report
# empty and sets memory_timed_out.

REMEMBER_KEY = "r"
FORGOTTEN_KEY = "f"
MEMORY_RESPONSE_TIMEOUT = 5.0        # seconds; None = fully self-paced


# -------------------------------------------------------- choice grid ---
# The number of answer choices and how foils are sampled are set in the start
# dialog (defaults in make_conditions.DEFAULT_CHOICE_OPTIONS). The grid is
# always a full rectangle, columns x rows = number of choices, in the shape
# that best fits the screen: 50 -> 10 x 5, 45 -> 9 x 5, 20 -> 5 x 4, 10 -> 5 x 2.

CHOICE_MAX_IMAGE_SIZE = 0.25         # height units; keeps small grids compact
CHOICE_GRID_MARGIN = 0.04            # height units kept free at every screen edge
CHOICE_IMAGE_FILL = 0.9              # image size as a proportion of its grid cell
CHOICE_TIMEOUT = 10.0                # seconds to click; None = waits for a click.
                                     # A timeout leaves the choice empty (choice_timed_out);
                                     # exact_correct etc. are then blank, not False
MOUSE_START_POSITION = (0, 0)        # same neutral start on every trial
RECORD_MOUSE_TRAJECTORY = True       # one row per frame in <data file>_mouse.csv


# ------------------------------------------------------------ localizer ---
# localizer.py: every image in stimuli/ shown LOCALIZER_REPETITIONS times in
# random order (shuffled in passes), passive viewing. Runs inside a session
# when 'localizer' is ticked (evening before study block 1, morning after the
# last recall run), or on its own with python localizer.py.
# 50 images x 10 x (1 s + 1-1.5 s) is about 19 minutes plus breaks.

LOCALIZER_REPETITIONS = 10
LOCALIZER_IMAGE_DURATION = 1.0
LOCALIZER_FIXATION_RANGE = (1.0, 1.5)   # jittered, so image onsets are aperiodic
LOCALIZER_IMAGE_SIZE = (0.5, 0.5)       # height units, as in study
LOCALIZER_BREAK_EVERY = 100             # presentations between short breaks


# --------------------------------------------------------------- breaks ---

BREAK_EVERY_TRIALS = 25              # 50 trials per run -> one break mid-run
BREAK_MINIMUM_SECONDS = 10


# -------------------------------------------------------- EEG triggers ---

TRIGGER_BACKEND = "none"             # "none", "parallel" or "serial"
PARALLEL_PORT_ADDRESS = 0x378
SERIAL_PORT = "COM3"
SERIAL_BAUDRATE = 115200
TRIGGER_PULSE_DURATION = 0.005


# ------------------------------------------------------------ quit key ---

QUIT_KEY = "escape"                  # ends the experiment, saving everything collected
QUIT_MODIFIERS = []                  # e.g. ["shift"] to require shift+escape

# Skips the rest of the current stage (an instruction screen, study block,
# shades of grey, recall run or reconstruction) and moves on to the next one.
# On a Mac keyboard F10 is the mute key: hold fn, or turn on "Use F1, F2, etc.
# keys as standard function keys" in System Settings > Keyboard.
SKIP_KEY = "f10"


# ----------------------------------------------------------- quick mode ---
# Ticking "quick_test" in the start dialog applies these overrides so the
# whole flow can be clicked through in a few minutes. Never use for data.

QUICK_TEST_OVERRIDES = {
    "STUDY_IMAGE_DURATION": 0.4,
    "STUDY_FIXATION_RANGE": (0.1, 0.3),
    "SHADES_DURATION": 15.0,
    "SHADES_MIN_GAP": 2.0,
    "SHADES_MAX_GAP": 4.0,
    "SHADES_EDGE_BUFFER": 1.0,
    "BREAK_MINIMUM_SECONDS": 1,
    "LOCALIZER_REPETITIONS": 1,
    "LOCALIZER_IMAGE_DURATION": 0.3,
    "LOCALIZER_FIXATION_RANGE": (0.1, 0.2),
    "QUICK_TEST_MAX_TRIALS": 6,      # per recall run (study always shows all 50)
}
QUICK_TEST_MAX_TRIALS = None


def apply_quick_test_overrides():
    for name, value in QUICK_TEST_OVERRIDES.items():
        globals()[name] = value
