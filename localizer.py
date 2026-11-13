"""
localizer.py

Image localizer for RIPPLAY: every image in stimuli/ is shown
LOCALIZER_REPETITIONS times (10), for LOCALIZER_IMAGE_DURATION (1 s) each,
with a jittered LOCALIZER_FIXATION_RANGE (1-1.5 s) fixation after every
image. Passive viewing: no task and no responses. Gives image- and
category-specific EEG patterns for training the replay classifiers.

Order
    Random, shuffled in passes: every image appears once per pass, in a fresh
    random order, so each image's ten presentations are spread evenly across
    the block and slow drifts (fatigue, electrode impedance) cannot line up
    with particular images. The same image never appears twice in a row,
    including across pass boundaries.

Images
    Every image file in stimuli/ (50 now, or however many are there). The
    category comes from stimuli/stimulus_list.csv, or for unlisted files from
    the first letter of the file name (f, o, s, w, b), as in make_conditions.py.

Running
    Inside a session: cued_recall.py runs it when 'localizer' is ticked in the
        start dialog: evening before the first study block, morning after the
        last recall run (as in the study design). F10 skips it, Escape quits.
    On its own: python localizer.py (its own start dialog and data files).

Data (same CSV as the rest of the session, phase = "localizer")
    one row per presentation: presentation number, pass, image, image ID,
    category, how many times this image has been shown so far, image onset
    (flip time) and the fixation duration that followed it

Everything specific to the localizer (EEG codes, instructions, the image
list) lives in this file; timings and sizes are in settings.py.
"""

import csv
import os
import random
from pathlib import Path

from psychopy import core, data, event, gui, logging, visual

import make_conditions
import settings
import triggers


# --------------------------------------------------------- EEG triggers ---
# Added to triggers.py's tables when this module is imported, so triggers.py
# does not need editing. Codes 20-26 are unused by the rest of the task.
# Image onsets carry the category in the last digit (21 face ... 25 body).

triggers.IMAGE_EVENT_BASES["localizer_image"] = 20
triggers.EVENT_CODES["localizer_fixation"] = 26

IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_ORDERING_ATTEMPTS = 1000

# "LOCALIZER" heads the instruction screen so the experimenter can tell this
# block from a study block, which looks the same (study opens with "Picture viewing.").
INSTRUCTIONS = (
    "LOCALIZER\n\n"
    "Picture viewing.\n\n"
    "Pictures will flash up one at a time.\n"
    "Keep your eyes on the centre of the screen and just look at each one.\n"
    "You do not need to press anything."
)


# ---------------------------------------------------------------- images ---

def get_localizer_images(stimuli_folder=settings.STIMULUS_ROOT / "stimuli",
                         stimulus_list_file=settings.STIMULUS_LIST_FILE):
    """Every image file in stimuli/, with its category and image ID.

    Paths are written as in the stimulus list ('stimuli/f (03).jpg'), i.e.
    relative to STIMULUS_ROOT.
    """
    listed_categories = {}
    if Path(stimulus_list_file).exists():
        for stimulus in make_conditions.load_stimulus_list(stimulus_list_file):
            listed_categories[Path(stimulus["image_file"]).name] = stimulus["category"]
    category_codes = {name: code for code, name in make_conditions.CATEGORY_NAMES.items()}

    images = []
    for image_path in sorted(Path(stimuli_folder).iterdir()):
        if image_path.suffix.lower() not in IMAGE_FILE_EXTENSIONS:
            continue
        category = listed_categories.get(image_path.name)
        if category is None:
            category = make_conditions.CATEGORY_NAMES.get(image_path.name[0].upper())
        if category is None:
            raise SystemExit(f"No category for {image_path.name}: add it to {stimulus_list_file}.")
        image_file = f"stimuli/{image_path.name}"
        images.append({
            "image_file": image_file,
            "category": category,
            "image_id": make_conditions.get_image_id(image_file, category_codes[category]),
        })

    if not images:
        raise SystemExit(f"No images found in {Path(stimuli_folder).resolve()}")
    return images


def build_presentation_order(images, repetitions, rng):
    """[(pass number, image), ...]: every image once per pass, random order
    within each pass, never the same image twice in a row."""
    presentations = []
    previous_image_file = None
    for pass_number in range(1, repetitions + 1):
        for _ in range(MAX_ORDERING_ATTEMPTS):
            pass_order = rng.sample(images, len(images))
            if len(images) == 1 or pass_order[0]["image_file"] != previous_image_file:
                break
        presentations += [(pass_number, image) for image in pass_order]
        previous_image_file = pass_order[-1]["image_file"]
    return presentations


# ---------------------------------------------------------------- block ---

def run_localizer(task, session):
    """The localizer block, using the session's window, markers and data file."""
    images = get_localizer_images()
    presentations = build_presentation_order(images, settings.LOCALIZER_REPETITIONS, task.rng)
    task.preload_images(image["image_file"] for image in images)

    block_name = f"localizer_{session}"
    task.set_context(phase="localizer", session=session, cycle="", trial="")
    task.show_message(INSTRUCTIONS, footer="Press SPACE to start")
    task.markers.send_now("block_start", detail=block_name)

    times_shown = {image["image_file"]: 0 for image in images}
    for presentation_number, (pass_number, image) in enumerate(presentations, start=1):
        task.set_context(trial=presentation_number)
        times_shown[image["image_file"]] += 1

        stimulus = task.place_image(image["image_file"], (0, 0), settings.LOCALIZER_IMAGE_SIZE)
        image_onset, _ = task.present_for(
            [stimulus], settings.LOCALIZER_IMAGE_DURATION, event_name="localizer_image",
            category=image["category"], detail=image["image_file"],
        )
        fixation_duration = task.jittered(settings.LOCALIZER_FIXATION_RANGE)
        task.present_for([task.fixation], fixation_duration, event_name="localizer_fixation")

        task.log_row(
            localizer_pass=pass_number,
            image_file=image["image_file"],
            image_id=image["image_id"],
            category=image["category"],
            image_repetition=times_shown[image["image_file"]],
            image_onset=image_onset,
            fixation_duration=round(fixation_duration, 4),
        )

        if presentation_number % settings.LOCALIZER_BREAK_EVERY == 0 and presentation_number < len(presentations):
            task.save_checkpoint()
            task.show_break()

    task.markers.send_now("block_end", detail=block_name)
    task.save_checkpoint()


# ------------------------------------------------------------ standalone ---

def show_standalone_dialog():
    session_info = {
        "participant": "P01",
        "session": ["evening", "morning", "standalone"],
        "quick_test": False,
    }
    dialog = gui.DlgFromDict(
        session_info, title="ripplay_localizer", sortKeys=False,
        tip={"quick_test": "Shortened timings for dry runs only - never for real data"},
    )
    if not dialog.OK:
        core.quit()
    session_info["participant"] = make_conditions.get_participant_id(
        make_conditions.get_participant_number(session_info["participant"])
    )
    return session_info


def main():
    """Run the localizer on its own, with its own data files."""
    # imported here, so cued_recall.py can import this module without a loop
    from cued_recall import CuedRecallRoutines
    from triggers import EventMarkers

    os.chdir(Path(__file__).resolve().parent)
    session_info = show_standalone_dialog()
    if session_info["quick_test"]:
        settings.apply_quick_test_overrides()
    participant_id, session = session_info["participant"], session_info["session"]

    settings.DATA_FOLDER.mkdir(exist_ok=True)
    date_string = data.getDateStr()
    quick_tag = "_quicktest" if session_info["quick_test"] else ""
    file_stem = str(settings.DATA_FOLDER / f"{participant_id}_{session}_localizer{quick_tag}_{date_string}")

    participant_number = make_conditions.get_participant_number(participant_id)
    runtime_seed = make_conditions.DESIGN_SEED + participant_number * 100 + 90
    experiment = data.ExperimentHandler(
        name="ripplay_localizer",
        extraInfo={**session_info, "date": date_string, "runtime_seed": runtime_seed,
                   "trigger_backend": settings.TRIGGER_BACKEND},
        dataFileName=file_stem, savePickle=False, saveWideText=True,
    )
    logging.LogFile(file_stem + ".log", level=logging.INFO)

    win = visual.Window(
        size=settings.WINDOW_SIZE, fullscr=settings.FULLSCREEN, screen=settings.SCREEN_NUMBER,
        monitor=settings.MONITOR_NAME, color=settings.BACKGROUND_COLOR, colorSpace="rgb",
        units="height", allowGUI=False,
    )
    win.mouseVisible = False

    settings.RECORD_MOUSE_TRAJECTORY = False   # no mouse responses in the localizer
    markers = EventMarkers(win, file_stem + "_events.csv")
    task = CuedRecallRoutines(win, experiment, markers, random.Random(runtime_seed), file_stem + "_mouse.csv",
                              make_conditions.SEQUENCE_LENGTH)
    task.set_context(participant=participant_id)

    def quit_session():
        markers.send_now("session_end", detail="quit key")
        task.save_checkpoint()
        experiment.abort()
        task.close()
        markers.close()
        win.close()
        core.quit()

    event.globalKeys.add(key=settings.QUIT_KEY, modifiers=settings.QUIT_MODIFIERS,
                         func=quit_session, name="quit")

    markers.send_now("session_start", detail=f"{session}_localizer")
    task.run_stage(f"localizer_{session}", run_localizer, task, session)
    task.run_stage("end", task.show_message, "That is the end of this part.\n\nThank you!", ("space",), "")
    markers.send_now("session_end", detail=f"{session}_localizer")

    task.save_checkpoint()
    experiment.abort()
    task.close()
    markers.close()
    win.close()
    core.quit()


if __name__ == "__main__":
    main()
