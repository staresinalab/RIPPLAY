
"""
cued_recall.py

RIPPLAY learning and cued-offset recall, PsychoPy Coder (boundary-agnostic
version). One launch runs a whole session.

EVENING (pre-sleep): six cycles of
    study           the 50 images in the participant's fixed order
                    (3 s each, jittered 1-2 s fixation between images)
    shades of grey  2 minutes: count the times the cross gets lighter
    recall run      50 cued-offset recall trials

MORNING (post-sleep): six recall runs, no re-study and no shades of grey

Every recall trial
    jittered fixation
    cue image at the centre, 500 ms, with the signed offset (e.g. +2, -3) on
    a translucent mask at its centre: one event (settings.OFFSET_ON_CUE;
    False gives the older number -> fixation -> image sequence)
    blank screen until R (remembered) or F (forgotten), at most
    MEMORY_RESPONSE_TIMEOUT (5 s)
    the answer grid for at most CHOICE_TIMEOUT (10 s), on the next frame after
    the key press or the timeout, freshly
    randomised on every trial: all 50 studied images, or fewer (target +
    random or banded foils), as chosen in the start dialog;
    the participant clicks one on BOTH remembered and forgotten trials

Before the first cycle (evening) or after the last recall run (morning), the
image localizer in localizer.py runs if 'localizer' is ticked.

start_run in the start dialog resumes a session after a crash: the session
restarts at the beginning of that cycle (for the evening, its study block).

Conditions come from make_conditions.py (run automatically if the files for
this participant do not exist yet). Run from this folder. Escape quits and
saves whatever has been collected; F10 skips to the next stage.
"""

import csv
import math
import os
import random
from pathlib import Path

from psychopy import core, data, event, gui, logging, visual
from psychopy.hardware import keyboard

import localizer
import make_conditions
import sequence_reconstruction
import settings
from triggers import EventMarkers


EXPERIMENT_NAME = "ripplay_cued_recall"

# grid sizes offered in the start dialog; 50 = every studied image (the original grid)
ANSWER_CHOICE_COUNTS = [50, 45, 40, 35, 30, 25, 20, 15, 10]

BAND_SIZE_TIPS = {
    "near_band_radius": "Every image within +-this many positions of the cue is offered (at least 3)",
    "mid_band_radius": "The mid band covers distances (near radius + 1) to this",
    "mid_band_count": "How many mid-band images, split across both sides of the cue",
    "camouflage_run_length": "One run of this many consecutive sequence images elsewhere (0 = none)",
    "camouflage_min_distance": "The camouflage run stays at least this far from the cue",
    "far_min_distance": "The remaining choices are drawn at least this far from the cue (at most 25)",
    "nearest_per_category_side": "Every category's nearest N images before and after the cue are offered",
    "camouflage_cycles": "Whole far five-image cycles offered (one image of each category each; 0 = none)",
}

DIGIT_KEYS = [str(digit) for digit in range(10)]

# how the cue is described depends on whether the number sits on the image
CUE_DESCRIPTION = (
    "One picture will briefly appear with a number on it, for example +3 or −2.\n"
    if settings.OFFSET_ON_CUE else
    "First you will see a number, for example +3 or −2. Then one picture will briefly appear.\n"
)
CUE_REMINDER = "the picture with its number" if settings.OFFSET_ON_CUE else "first the number, then the picture"

# how long participants have for R/F and for the click, as described to them
MEMORY_TIME_DESCRIPTION = (
    "Take as long as you need; the screen stays blank until you press R or F.\n\n"
    if settings.MEMORY_RESPONSE_TIMEOUT is None else
    f"You have {settings.MEMORY_RESPONSE_TIMEOUT:g} seconds to press R or F.\n\n"
)
CHOICE_TIME_DESCRIPTION = (
    "" if settings.CHOICE_TIMEOUT is None else
    f"\nYou have {settings.CHOICE_TIMEOUT:g} seconds to click."
)

# Learning is incidental: nothing before the first recall run mentions
# sequences or order. Order is first named in the recall instructions, where
# the +/- offsets have to be explained.
INSTRUCTIONS = {
    "evening_welcome": (
        "Welcome!\n\n"
        "Today you will look at pictures and do some short tasks.\n\n"
        "The experimenter will explain each part as we go."
    ),
    "first_study": (
        "Picture viewing.\n\n"
        "Pictures will appear one at a time. Please pay attention to each one.\n\n"
        "Afterwards there is a short counting task and then a memory test."
    ),
    "repeat_study": "Picture viewing.\n\nPictures will appear one at a time. Please pay attention to each one.",
    "shades": (
        "Keep your eyes on the cross.\n\n"
        "Silently count how many times it gets LIGHTER.\n"
        "Ignore the times it gets darker."
    ),
    "first_run": (
        "Memory test.\n\n"
        + CUE_DESCRIPTION +
        "+3 means the picture that was shown 3 pictures AFTER that picture.\n"
        "−2 means the picture that was shown 2 pictures BEFORE it.\n"
        "Keep your eyes on the centre of the screen.\n\n"
        "Then the screen goes blank. Try to bring the picture you are looking for to mind.\n"
        "While the screen is blank, tell us whether you remember it:\n"
        "press R if you REMEMBER it, or F if you have FORGOTTEN it.\n"
        + MEMORY_TIME_DESCRIPTION +
        "Then all the pictures appear: click the one you think it was.\n"
        "Always choose one, and guess if you are unsure."
        + CHOICE_TIME_DESCRIPTION
    ),
    "later_run": (
        "Memory test.\n\n"
        f"Same as before: {CUE_REMINDER}, then the screen goes blank.\n"
        "Press R if you remember it, F if you have forgotten it.\n"
        "Then click the picture you think it was, guessing if you are unsure."
        + CHOICE_TIME_DESCRIPTION
    ),
    "morning_welcome": (
        "Welcome back!\n\n"
        "This memory test is about the pictures you saw last time."
    ),
    "end": "That is the end of this session.\n\nThank you! Please let the experimenter know you have finished.",
}


# --------------------------------------------------------------- routines ---

class SkipStage(Exception):
    """Raised on the first flip after SKIP_KEY (F10); caught by run_stage."""

class CuedRecallRoutines:
    def __init__(self, win, experiment, markers, rng, mouse_file_path, number_of_choices):
        self.win = win
        self.experiment = experiment
        self.markers = markers
        self.rng = rng

        self.keyboard = keyboard.Keyboard()
        self.mouse = event.Mouse(win=win, visible=False)
        self.frame_duration = win.monitorFramePeriod or 1 / 60
        self.last_flip_time = core.getTime()
        self.context = {}
        self.image_stims = {}
        self.skip_requested = False

        self.mouse_file = None
        if settings.RECORD_MOUSE_TRAJECTORY:
            self.mouse_file = open(mouse_file_path, "w", newline="")
            self.mouse_writer = csv.writer(self.mouse_file)
            self.mouse_writer.writerow(["session", "cycle", "trial", "time_from_grid_onset", "x", "y"])

        self.fixation = visual.TextStim(
            win, text="+", height=settings.FIXATION_HEIGHT, color=settings.FIXATION_COLOR, font="Arial"
        )
        self.message = visual.TextStim(
            win, text="", height=settings.TEXT_HEIGHT, color=settings.TEXT_COLOR,
            wrapWidth=1.2, font="Arial",
        )
        self.small_message = visual.TextStim(
            win, text="", height=settings.TEXT_HEIGHT * 0.8, color=settings.TEXT_COLOR,
            pos=(0, -0.42), wrapWidth=1.4, font="Arial",
        )
        # signed offset, shown alone at the centre before the cue image
        self.offset_text = visual.TextStim(
            win, text="", height=settings.OFFSET_TEXT_HEIGHT, color=settings.OFFSET_TEXT_COLOR,
            pos=(0, 0), font="Arial", bold=True,
        )
        self.highlight_frame = visual.Rect(
            win, width=1, height=1, lineColor=settings.TEXT_COLOR, fillColor=None, lineWidth=4
        )
        # translucent disc behind the offset number on the cue image
        self.offset_mask = visual.Circle(
            win, radius=settings.OFFSET_MASK_RADIUS, edges=64, pos=(0, 0),
            fillColor=settings.OFFSET_MASK_COLOR, lineColor=None, opacity=settings.OFFSET_MASK_OPACITY,
        )

        self.grid_positions, self.grid_image_size, self.grid_columns = self.get_grid_layout(number_of_choices)

    # ------------------------------------------------------ basic tools ---

    def get_image(self, image_file):
        if image_file not in self.image_stims:
            image_path = settings.STIMULUS_ROOT / image_file
            self.image_stims[image_file] = visual.ImageStim(self.win, image=str(image_path), interpolate=True)
        return self.image_stims[image_file]

    def preload_images(self, image_files):
        """Load textures up front so no trial pays the file-loading delay."""
        for image_file in image_files:
            self.get_image(image_file)

    def place_image(self, image_file, position, size):
        image = self.get_image(image_file)
        image.pos = position
        image.size = size
        return image

    def draw_and_flip(self, stimuli):
        for stimulus in stimuli:
            stimulus.draw()
        self.last_flip_time = self.win.flip()
        self.check_for_skip()
        return self.last_flip_time

    # ---------------------------------------------------- skipping stages ---

    def check_for_skip(self):
        """Called after every flip: SKIP_KEY (F10) ends the current stage.

        PsychoPy's event.globalKeys only accepts letters, digits and a few
        named keys (not function keys), so the skip key is polled here
        instead. event.getKeys with a keyList removes only that key from the
        buffer, so no other responses are lost.
        """
        if self.skip_requested or event.getKeys(keyList=[settings.SKIP_KEY]):
            self.skip_requested = False
            raise SkipStage()

    def run_stage(self, stage_name, stage_function, *arguments):
        """Run one stage; SKIP_KEY ends it early and the session moves on.

        A skipped stage keeps every row and event logged before the key press,
        and gets a 'stage_skipped' row and EEG marker.
        """
        self.skip_requested = False
        event.getKeys(keyList=[settings.SKIP_KEY])   # drop presses from before this stage
        try:
            stage_function(*arguments)
        except SkipStage:
            self.win.mouseVisible = False
            self.skip_requested = False
            self.markers.send_now("stage_skipped", detail=stage_name)
            self.log_row(stage_skipped=stage_name)
            self.save_checkpoint()
            self.draw_and_flip([])

    def frames_remaining(self, onset, duration):
        """True while the NEXT flip still falls inside [onset, onset + duration)."""
        next_flip_time = self.last_flip_time + self.frame_duration
        return next_flip_time - onset < duration - self.frame_duration / 2

    def present_for(self, stimuli, duration, event_name=None, category=None, detail="",
                    response_keys=None):
        """Show stimuli for a fixed duration; return (onset, keys pressed meanwhile)."""
        if event_name:
            self.markers.send_on_next_flip(event_name, category, detail)
        if response_keys:
            self.keyboard.clearEvents()
            self.win.callOnFlip(self.keyboard.clock.reset)

        onset = self.draw_and_flip(stimuli)
        keys_pressed = []
        while self.frames_remaining(onset, duration):
            if response_keys:
                keys_pressed.extend(self.keyboard.getKeys(keyList=response_keys, waitRelease=False))
            self.draw_and_flip(stimuli)
        if response_keys:
            keys_pressed.extend(self.keyboard.getKeys(keyList=response_keys, waitRelease=False))
        return onset, keys_pressed

    def wait_for_key(self, stimuli, key_list, timeout=None, event_name=None, category=None, detail=""):
        """Show stimuli until one of key_list is pressed; return (onset, key name or None, rt)."""
        if event_name:
            self.markers.send_on_next_flip(event_name, category, detail)
        self.keyboard.clearEvents()
        self.win.callOnFlip(self.keyboard.clock.reset)

        onset = self.draw_and_flip(stimuli)
        while timeout is None or self.frames_remaining(onset, timeout):
            keys = self.keyboard.getKeys(keyList=key_list, waitRelease=False)
            if keys:
                return onset, keys[0].name, keys[0].rt
            self.draw_and_flip(stimuli)
        return onset, None, None

    def jittered(self, duration_range):
        return self.rng.uniform(*duration_range)

    def set_context(self, **context):
        """Labels (phase, session, cycle, trial...) added to every data row and event that follows."""
        self.context.update(context)
        self.markers.set_context(**context)

    def log_row(self, **columns):
        for column, value in {**self.context, **columns}.items():
            self.experiment.addData(column, value)
        self.experiment.nextEntry()

    def save_checkpoint(self):
        """Rewrite the data CSV so a crash loses at most the current block."""
        self.experiment.saveAsWideText(self.experiment.dataFileName + ".csv", fileCollisionMethod="overwrite")
        if self.mouse_file is not None:
            self.mouse_file.flush()

    def close(self):
        if self.mouse_file is not None:
            self.mouse_file.close()
            self.mouse_file = None

    # ---------------------------------------------------------- screens ---

    def show_message(self, text, continue_keys=("space",), footer="Press SPACE to continue"):
        self.message.text = text
        self.small_message.text = footer
        self.draw_and_flip([self.message, self.small_message])
        core.wait(0.5)  # ignore keys already held from the previous screen
        self.wait_for_key([self.message, self.small_message], list(continue_keys))
        self.draw_and_flip([])

    def show_break(self):
        self.markers.send_on_next_flip("break_start")
        self.message.text = "Short break.\n\nRest your eyes for a moment."
        self.present_for([self.message], settings.BREAK_MINIMUM_SECONDS)
        self.show_message("Ready to carry on?")
        self.markers.send_now("break_end")

    def ask_for_number(self, prompt):
        """Typed number entry (digits, backspace, return). Returns an int."""
        typed = ""
        self.keyboard.clearEvents()
        while True:
            self.message.text = f"{prompt}\n\n{typed or '_'}"
            self.small_message.text = "Type the number, then press RETURN"
            self.draw_and_flip([self.message, self.small_message])
            for key in self.keyboard.getKeys(keyList=DIGIT_KEYS + ["backspace", "return"], waitRelease=False):
                if key.name == "backspace":
                    typed = typed[:-1]
                elif key.name == "return" and typed:
                    self.draw_and_flip([])
                    return int(typed)
                elif key.name in DIGIT_KEYS and len(typed) < 3:
                    typed += key.name

    # ------------------------------------------------------------ study ---

    def run_study_block(self, sequence_rows, session, cycle):
        """The 50 images in the participant's fixed order, with a jittered
        fixation after each one so image onsets are aperiodic."""
        block_name = f"study_{session}_cycle{cycle}"
        self.preload_images(row["image_file"] for row in sequence_rows)
        self.set_context(phase="study", session=session, cycle=cycle, trial="")
        self.markers.send_now("block_start", detail=block_name)

        for row in sorted(sequence_rows, key=lambda row: row["position"]):
            self.set_context(trial=row["position"])
            stimulus = self.place_image(row["image_file"], (0, 0), settings.STUDY_IMAGE_SIZE)
            onset, _ = self.present_for(
                [stimulus], settings.STUDY_IMAGE_DURATION, event_name="study_image",
                category=row["category"], detail=row["image_file"],
            )
            fixation_duration = self.jittered(settings.STUDY_FIXATION_RANGE)
            self.present_for([self.fixation], fixation_duration, event_name="study_fixation")

            self.log_row(
                position=row["position"],
                sequence_cycle=row["cycle"],
                slot_in_cycle=row["slot_in_cycle"],
                category=row["category"],
                is_decoy=row["is_decoy"],
                image_file=row["image_file"],
                image_onset=onset,
                fixation_duration=round(fixation_duration, 4),
            )

        self.markers.send_now("block_end", detail=block_name)
        self.save_checkpoint()

    # --------------------------------------------------- shades of grey ---

    def schedule_shades_events(self):
        events = []
        onset = settings.SHADES_EDGE_BUFFER + self.rng.uniform(0, settings.SHADES_MAX_GAP - settings.SHADES_MIN_GAP)
        while onset < settings.SHADES_DURATION - settings.SHADES_EDGE_BUFFER:
            change = "lighter" if self.rng.random() < settings.SHADES_LIGHTER_PROBABILITY else "darker"
            events.append({"planned_onset": onset, "change": change, "actual_onset": None})
            onset += self.rng.uniform(settings.SHADES_MIN_GAP, settings.SHADES_MAX_GAP)
        return events

    def run_shades_of_grey(self, session, cycle):
        """Two-minute interval between study and recall: count the times the
        cross gets LIGHTER, ignore darker.

        Changes are sparse and jittered; every onset is logged and marked so
        event-related windows can be excluded from offline-replay analyses.
        """
        block_name = f"shades_{session}_cycle{cycle}"
        self.set_context(phase="shades_of_grey", session=session, cycle=cycle, trial="")
        self.show_message(INSTRUCTIONS["shades"], footer="Press SPACE to start")
        events = self.schedule_shades_events()

        self.fixation.color = settings.FIXATION_COLOR
        self.markers.send_on_next_flip("shades_start", detail=block_name)
        start = self.draw_and_flip([self.fixation])

        while self.frames_remaining(start, settings.SHADES_DURATION):
            next_frame_time = self.last_flip_time + self.frame_duration - start
            active_event = None
            for shades_event in events:
                if shades_event["planned_onset"] <= next_frame_time < shades_event["planned_onset"] + settings.SHADES_EVENT_DURATION:
                    active_event = shades_event

            if active_event is None:
                self.fixation.color = settings.FIXATION_COLOR
            else:
                lighter = active_event["change"] == "lighter"
                self.fixation.color = settings.SHADES_LIGHTER_COLOR if lighter else settings.SHADES_DARKER_COLOR
                if active_event["actual_onset"] is None:
                    self.markers.send_on_next_flip(f"shades_{active_event['change']}", detail=block_name)

            flip_time = self.draw_and_flip([self.fixation])
            if active_event is not None and active_event["actual_onset"] is None:
                active_event["actual_onset"] = flip_time

        self.fixation.color = settings.FIXATION_COLOR
        self.markers.send_now("shades_end", detail=block_name)

        true_count = sum(shades_event["change"] == "lighter" for shades_event in events)
        reported_count = self.ask_for_number("How many times did the cross get LIGHTER?")

        for event_number, shades_event in enumerate(events, start=1):
            self.log_row(
                trial=event_number,
                shades_change=shades_event["change"],
                shades_planned_onset=round(shades_event["planned_onset"], 4),
                shades_onset=shades_event["actual_onset"],
                shades_block_start=start,
            )
        self.log_row(
            trial="count",
            shades_block_start=start,
            shades_true_lighter_count=true_count,
            shades_reported_lighter_count=reported_count,
            shades_count_correct=reported_count == true_count,
        )
        self.save_checkpoint()

    # ------------------------------------------------------ choice grid ---

    def get_grid_layout(self, number_of_choices):
        """Centres of the grid cells (slot 1 = top left, row by row), the image
        size and the number of columns, fitted to this window's aspect ratio.

        The grid is always a full, regularly spaced rectangle: columns x rows
        equals the number of choices, using the factor pair whose shape is
        closest to the usable screen area (50 -> 10 x 5, 45 -> 9 x 5,
        20 -> 5 x 4, 10 -> 5 x 2). The dialog offers multiples of 5 only; a
        prime number of choices would give a single row. Images never grow
        beyond CHOICE_MAX_IMAGE_SIZE.
        """
        aspect_ratio = self.win.size[0] / self.win.size[1]
        usable_width = aspect_ratio - 2 * settings.CHOICE_GRID_MARGIN
        usable_height = 1 - 2 * settings.CHOICE_GRID_MARGIN

        factor_pairs = [
            (columns, number_of_choices // columns)
            for columns in range(1, number_of_choices + 1) if number_of_choices % columns == 0
        ]
        columns, rows = min(
            factor_pairs,
            key=lambda pair: abs(math.log((pair[0] / pair[1]) / (usable_width / usable_height))),
        )
        cell_size = min(
            usable_width / columns,
            usable_height / rows,
            settings.CHOICE_MAX_IMAGE_SIZE / settings.CHOICE_IMAGE_FILL,
        )

        positions = []
        for row in range(rows):
            y_position = (rows - 1) / 2 * cell_size - row * cell_size
            for column in range(columns):
                x_position = -(columns - 1) / 2 * cell_size + column * cell_size
                positions.append((x_position, y_position))
        image_size = cell_size * settings.CHOICE_IMAGE_FILL
        return positions, (image_size, image_size), columns

    def get_slot_coordinates(self, slot):
        """Centre of a grid slot in height units (origin at screen centre, y up)
        and in pixels (origin at the top-left corner, y down, as eye trackers
        report gaze). Pixels are the window's framebuffer pixels (win.size)."""
        x_height, y_height = self.grid_positions[slot - 1]
        window_width, window_height = self.win.size
        x_pixels = window_width / 2 + x_height * window_height
        y_pixels = window_height / 2 - y_height * window_height
        return {
            "x_height": round(x_height, 4), "y_height": round(y_height, 4),
            "x_pixels": round(x_pixels, 1), "y_pixels": round(y_pixels, 1),
        }

    def save_grid_layout(self, layout_file_path):
        """One row per grid slot: its centre and image size, in height units and pixels."""
        image_size_height = self.grid_image_size[0]
        with open(layout_file_path, "w", newline="") as layout_file:
            writer = csv.DictWriter(layout_file, fieldnames=[
                "slot", "row", "column", "x_height", "y_height", "x_pixels", "y_pixels",
                "image_size_height", "image_size_pixels", "window_width_pixels", "window_height_pixels",
            ])
            writer.writeheader()
            for slot in range(1, len(self.grid_positions) + 1):
                writer.writerow({
                    "slot": slot,
                    "row": (slot - 1) // self.grid_columns + 1,
                    "column": (slot - 1) % self.grid_columns + 1,
                    **self.get_slot_coordinates(slot),
                    "image_size_height": round(image_size_height, 4),
                    "image_size_pixels": round(image_size_height * self.win.size[1], 1),
                    "window_width_pixels": self.win.size[0],
                    "window_height_pixels": self.win.size[1],
                })

    def get_trial_coordinates(self, prefix, slot):
        """{prefix_x_height: ..., prefix_y_pixels: ...} for one slot, or empty values."""
        names = ["x_height", "y_height", "x_pixels", "y_pixels"]
        coordinates = self.get_slot_coordinates(slot) if slot else {}
        return {f"{prefix}_{name}": coordinates.get(name) for name in names}

    def wait_for_choice_click(self, choice_stimuli, timeout):
        """Mouse choice among the grid images. Returns (onset, slot or None, rt).

        The mouse starts at the same place on every trial; with
        RECORD_MOUSE_TRAJECTORY its position is written once per frame.
        """
        self.mouse.setPos(settings.MOUSE_START_POSITION)
        self.win.mouseVisible = True
        self.markers.send_on_next_flip("choice_screen")
        onset = self.draw_and_flip(choice_stimuli)

        button_released = not self.mouse.getPressed()[0]
        while timeout is None or self.frames_remaining(onset, timeout):
            mouse_x, mouse_y = self.mouse.getPos()
            if self.mouse_file is not None:
                self.mouse_writer.writerow([
                    self.context.get("session"), self.context.get("cycle"), self.context.get("trial"),
                    round(core.getTime() - onset, 4), round(mouse_x, 4), round(mouse_y, 4),
                ])

            hovered_slot = None
            for slot, stimulus in enumerate(choice_stimuli, start=1):
                if stimulus.contains(self.mouse):
                    hovered_slot = slot

            button_down = self.mouse.getPressed()[0]
            if button_down and button_released and hovered_slot is not None:
                rt = core.getTime() - onset
                self.markers.send_now("choice_response", detail=f"slot_{hovered_slot}")
                self.win.mouseVisible = False
                return onset, hovered_slot, rt
            if not button_down:
                button_released = True

            frame_stimuli = list(choice_stimuli)
            if hovered_slot is not None:
                self.highlight_frame.pos = choice_stimuli[hovered_slot - 1].pos
                self.highlight_frame.size = [dimension * 1.08 for dimension in self.grid_image_size]
                frame_stimuli.append(self.highlight_frame)
            self.draw_and_flip(frame_stimuli)

        self.win.mouseVisible = False
        return onset, None, None

    # ------------------------------------------------------- recall run ---

    def get_offset_label(self, offset):
        """'+2' or '−2' (true minus sign, same width as the plus)."""
        sign = "+" if offset > 0 else settings.MINUS_SIGN
        return f"{sign}{abs(offset)}"

    def run_recall_run(self, trials, sequence_rows, session, cycle):
        row_by_image = {row["image_file"]: row for row in sequence_rows}
        if settings.QUICK_TEST_MAX_TRIALS:
            trials = trials[:settings.QUICK_TEST_MAX_TRIALS]
        self.preload_images(row["image_file"] for row in sequence_rows)

        block_name = f"recall_{session}_cycle{cycle}"
        self.set_context(phase="recall", session=session, cycle=cycle, trial="")
        self.markers.send_now("block_start", detail=block_name)

        memory_keys = [settings.REMEMBER_KEY, settings.FORGOTTEN_KEY]
        for trial in trials:
            self.set_context(trial=trial["trial"])

            iti = self.jittered(settings.RECALL_ITI_RANGE)
            self.present_for([self.fixation], iti, event_name="recall_fixation")

            # R/F presses before the blank are too early and ignored
            offset_label = self.get_offset_label(trial["offset"])
            self.offset_text.text = offset_label
            cue = self.place_image(trial["cue_image"], (0, 0), settings.CUE_IMAGE_SIZE)

            if settings.OFFSET_ON_CUE:
                # one event: the cue image with the offset on a translucent mask at its centre
                cue_onset, early_keys = self.present_for(
                    [cue, self.offset_mask, self.offset_text], settings.CUE_DURATION,
                    event_name="recall_cue", category=trial["cue_category"],
                    detail=f"{trial['cue_image']}|{trial['offset']:+d}", response_keys=memory_keys,
                )
                offset_onset, offset_to_cue_duration = cue_onset, 0.0
            else:
                # signed offset alone, then jittered fixation, then the cue image alone
                offset_onset, early_keys = self.present_for(
                    [self.offset_text], settings.OFFSET_DURATION,
                    event_name="recall_offset", detail=f"{trial['offset']:+d}", response_keys=memory_keys,
                )
                offset_to_cue_duration = self.jittered(settings.OFFSET_TO_CUE_RANGE)
                _, early_keys_in_gap = self.present_for(
                    [self.fixation], offset_to_cue_duration,
                    event_name="recall_offset_fixation", response_keys=memory_keys,
                )
                early_keys += early_keys_in_gap
                cue_onset, early_keys_in_cue = self.present_for(
                    [cue], settings.CUE_DURATION,
                    event_name="recall_cue", category=trial["cue_category"],
                    detail=f"{trial['cue_image']}|{trial['offset']:+d}", response_keys=memory_keys,
                )
                early_keys += early_keys_in_cue

            # blank screen, self-paced, until the remembered / forgotten judgement.
            # The first blank frame is flipped before the grid is set up, so the blank
            # appears on time; the grid is then positioned while the participant decides,
            # so nothing is left to prepare after the key press.
            self.markers.send_on_next_flip("memory_judgement_blank")
            self.keyboard.clearEvents()
            self.win.callOnFlip(self.keyboard.clock.reset)  # memory_rt counts from the blank's flip
            memory_blank_onset = self.draw_and_flip([])

            if trial["number_of_choices"] != len(self.grid_positions):
                raise SystemExit(f"{trial['number_of_choices']}-choice trial on a "
                                 f"{len(self.grid_positions)}-slot grid: regenerate the conditions files.")
            choice_slots = range(1, trial["number_of_choices"] + 1)
            choice_files = [trial[f"choice_{slot:02d}"] for slot in choice_slots]
            choice_stimuli = [
                self.place_image(choice_file, position, self.grid_image_size)
                for choice_file, position in zip(choice_files, self.grid_positions)
            ]

            memory_key, memory_rt, memory_keypress_time = None, None, None
            while memory_key is None:
                keys = self.keyboard.getKeys(keyList=memory_keys, waitRelease=False)
                if keys:
                    memory_key, memory_rt = keys[0].name, keys[0].rt
                    memory_keypress_time = memory_blank_onset + memory_rt
                    break
                if (settings.MEMORY_RESPONSE_TIMEOUT is not None
                        and not self.frames_remaining(memory_blank_onset, settings.MEMORY_RESPONSE_TIMEOUT)):
                    break
                self.draw_and_flip([])

            memory_report = {settings.REMEMBER_KEY: "remember", settings.FORGOTTEN_KEY: "forgotten"}.get(memory_key)
            memory_timed_out = memory_key is None
            if memory_report:
                self.markers.send_now(f"recall_{memory_report}")
            else:
                self.markers.send_now("memory_timeout")

            # choice among the grid images, on the next frame after the key press or timeout
            choice_onset, chosen_slot, choice_rt = self.wait_for_choice_click(choice_stimuli, settings.CHOICE_TIMEOUT)
            choice_timed_out = chosen_slot is None
            if choice_timed_out:
                self.markers.send_now("choice_timeout")
            self.draw_and_flip([])
            grid_delay_after_keypress = choice_onset - memory_keypress_time if memory_keypress_time else None

            chosen_row = row_by_image[choice_files[chosen_slot - 1]] if chosen_slot else None
            if chosen_row:
                # chosen image relative to the cue, on the same circle as the offsets (-25 ... +24)
                chosen_offset_from_cue = (chosen_row["position"] - trial["cue_position"] + 25) % 50 - 25
            else:
                chosen_offset_from_cue = None

            self.log_row(
                **trial,
                offset_label=offset_label,
                offset_on_cue=settings.OFFSET_ON_CUE,
                early_memory_keys=len(early_keys),
                memory_report=memory_report,
                memory_rt=memory_rt,
                memory_timed_out=memory_timed_out,
                choice_timed_out=choice_timed_out,
                chosen_slot=chosen_slot,
                chosen_image=chosen_row["image_file"] if chosen_row else None,
                chosen_image_id=chosen_row["image_id"] if chosen_row else None,
                **self.get_trial_coordinates("target_grid", trial["target_grid_slot"]),
                **self.get_trial_coordinates("cue_grid", trial["cue_grid_slot"]),
                **self.get_trial_coordinates("chosen_grid", chosen_slot),
                chosen_position=chosen_row["position"] if chosen_row else None,
                chosen_category=chosen_row["category"] if chosen_row else None,
                chosen_is_decoy=chosen_row["is_decoy"] if chosen_row else None,
                chosen_offset_from_cue=chosen_offset_from_cue,
                chosen_band=trial[f"choice_{chosen_slot:02d}_band"] if chosen_slot else None,
                # blank (not False) when the choice timed out, so misses are not scored as errors
                chose_cue=chosen_row["position"] == trial["cue_position"] if chosen_row else None,
                exact_correct=chosen_row["image_file"] == trial["target_image"] if chosen_row else None,
                category_correct=chosen_row["category"] == trial["target_category"] if chosen_row else None,
                choice_rt=choice_rt,
                offset_onset=offset_onset,
                cue_onset=cue_onset,
                memory_blank_onset=memory_blank_onset,
                memory_keypress_time=memory_keypress_time,
                choice_onset=choice_onset,
                grid_delay_after_keypress=(
                    round(grid_delay_after_keypress, 4) if grid_delay_after_keypress is not None else None
                ),
                offset_to_cue_duration=round(offset_to_cue_duration, 4),
                iti=round(iti, 4),
            )

            if trial["trial"] % settings.BREAK_EVERY_TRIALS == 0 and trial["trial"] < len(trials):
                self.save_checkpoint()
                self.show_break()

        self.markers.send_now("block_end", detail=block_name)
        self.save_checkpoint()


# ------------------------------------------------------------- session ---

def show_session_dialog():
    run_choices = [str(run_number) for run_number in range(1, make_conditions.NUMBER_OF_RUNS + 1)]
    session_info = {
        "participant": "P01",
        "session": make_conditions.SESSIONS,
        "start_run": run_choices,
        "reconstruction": True,
        "localizer": True,
        "quick_test": False,
        "answer_choices": [str(count) for count in ANSWER_CHOICE_COUNTS],
        "choice_sampling": list(make_conditions.CHOICE_SAMPLING_STRATEGIES),
    }
    dialog = gui.DlgFromDict(
        session_info, title=EXPERIMENT_NAME, sortKeys=False,
        tip={
            "start_run": "Leave at 1. Only change it to resume after a crash: "
                         "the session restarts at the beginning of that cycle.",
            "reconstruction": "Full sequence reconstruction: evening after the sixth cycle, "
                              "morning before the first recall run. Untick when resuming "
                              "a morning session in which it was already done.",
            "localizer": "Image localizer (each image 10 x 1 s, about 19 min): evening before the "
                         "first study block, morning after the last recall run. In the evening it "
                         "only runs when start_run is 1.",
            "quick_test": "Shortened timings for dry runs only - never for real data",
            "answer_choices": "Images on the choice screen, target included. 50 = every studied "
                              "image, cue included (the original grid); fewer never include the cue.",
            "choice_sampling": "random: uniform foils; bands: foils built around the cue's position; "
                               "category_bands: the same near band with equal numbers per category "
                               "(band sizes asked next)",
        },
    )
    if not dialog.OK:
        core.quit()
    session_info["participant"] = make_conditions.get_participant_id(
        make_conditions.get_participant_number(session_info["participant"])
    )
    session_info["start_run"] = int(session_info["start_run"])
    return session_info


def list_with_first(options, first):
    """Dropdown list with the current choice first (the dialog shows the first item)."""
    return [first] + [option for option in options if option != first]


def get_choice_options(session_info):
    """Answer-grid settings from the start dialog, plus band sizes for 'bands' or 'category_bands'.

    Settings that don't fit together (e.g. bands bigger than the grid) are
    explained and asked for again.
    """
    choice_options = dict(make_conditions.DEFAULT_CHOICE_OPTIONS)
    choice_options["number_of_choices"] = int(session_info.pop("answer_choices"))
    choice_options["sampling"] = session_info.pop("choice_sampling")
    category_sequence = make_conditions.get_category_sequence(session_info["participant"])

    while True:
        band_size_names = make_conditions.BAND_SIZE_NAMES_BY_STRATEGY.get(choice_options["sampling"])
        if band_size_names:
            band_sizes = {name: choice_options[name] for name in band_size_names}
            band_dialog = gui.DlgFromDict(
                band_sizes, title=f"Answer-choice bands ({choice_options['number_of_choices']} choices)",
                sortKeys=False, tip=BAND_SIZE_TIPS,
            )
            if not band_dialog.OK:
                core.quit()
            for name, value in band_sizes.items():
                choice_options[name] = int(value)

        problems = make_conditions.check_choice_options(choice_options, category_sequence)
        if not problems:
            return choice_options

        grid_dialog = gui.Dlg(title="Answer-choice settings don't fit")
        grid_dialog.addText("\n".join(problems))
        grid_dialog.addField("answer_choices", choices=list_with_first(
            [str(count) for count in ANSWER_CHOICE_COUNTS], str(choice_options["number_of_choices"])
        ))
        grid_dialog.addField("choice_sampling", choices=list_with_first(
            make_conditions.CHOICE_SAMPLING_STRATEGIES, choice_options["sampling"]
        ))
        answers = grid_dialog.show()
        if not grid_dialog.OK:
            core.quit()
        if isinstance(answers, dict):
            answer_choices, choice_sampling = answers["answer_choices"], answers["choice_sampling"]
        else:
            answer_choices, choice_sampling = answers
        choice_options["number_of_choices"] = int(answer_choices)
        choice_options["sampling"] = choice_sampling


def load_participant_design(participant_id, choice_options):
    """Sequence rows and a note for the check screen, generating this
    participant's conditions files if they are missing, were made by an older
    version of make_conditions.py, or were made with other grid settings.

    Regenerating for new grid settings changes only the grids: the questions
    and their order come from a separate random generator.
    """
    if not hasattr(make_conditions, "DESIGN_VERSION"):
        raise SystemExit(
            "make_conditions.py in this folder is older than cued_recall.py "
            f"({Path(make_conditions.__file__).resolve()}). Replace it with the current version."
        )
    sequence_file = make_conditions.get_sequence_file(participant_id, settings.CONDITIONS_FOLDER)
    wanted_settings = make_conditions.get_choice_settings_label(choice_options)
    design_note = ""
    if sequence_file.exists():
        existing_rows = make_conditions.load_rows(sequence_file)
        file_version = existing_rows[0].get("design_version") if existing_rows else None
        file_settings = existing_rows[0].get("choice_settings") if existing_rows else None
        if file_version != make_conditions.DESIGN_VERSION:
            logging.warning(f"{sequence_file} is from design version {file_version}; regenerating")
            design_note = f"Conditions files regenerated (they were from design version {file_version})."
            make_conditions.make_participant_conditions(participant_id, choice_options)
        elif file_settings != wanted_settings:
            logging.warning(f"{sequence_file} has grid settings {file_settings}; regenerating for {wanted_settings}")
            design_note = (
                f"Conditions files regenerated: they had grid settings {file_settings}, now {wanted_settings}.\n"
                f"Questions and their order are unchanged; only the grids differ."
            )
            make_conditions.make_participant_conditions(participant_id, choice_options)
    else:
        make_conditions.make_participant_conditions(participant_id, choice_options)
    sequence_rows = make_conditions.load_rows(sequence_file)

    missing_images = [row["image_file"] for row in sequence_rows
                      if not (settings.STIMULUS_ROOT / row["image_file"]).exists()]
    if missing_images:
        raise SystemExit(
            f"{len(missing_images)} images not found under {settings.STIMULUS_ROOT.resolve()} "
            f"(first: {missing_images[0]}). Check STIMULUS_ROOT in settings.py."
        )
    return sequence_rows, design_note


def describe_session(session, start_run, run_reconstruction, run_localizer):
    if session == "evening":
        cycles = f"cycles {start_run}-6 of study -> shades of grey -> recall"
        reconstruction_note = "then sequence reconstruction" if run_reconstruction else "NO sequence reconstruction"
        localizer_note = (
            "image localizer BEFORE the first study block" if run_localizer and start_run == 1
            else "NO image localizer"
        )
    else:
        cycles = f"recall runs {start_run}-6 (no study, no shades of grey)"
        reconstruction_note = (
            "sequence reconstruction BEFORE the recall runs" if run_reconstruction else "NO sequence reconstruction"
        )
        localizer_note = "image localizer AFTER the last recall run" if run_localizer else "NO image localizer"
    resume_note = f"\n\nRESUMING at cycle {start_run}." if start_run > 1 else ""
    return f"{cycles}\n{reconstruction_note}\n{localizer_note}{resume_note}"


def confirm_session(session_info, sequence_rows, choice_options, design_note):
    categories_in_order = "".join(row["category_code"] for row in sequence_rows[:10])
    confirmation = gui.Dlg(title="Check session")
    confirmation.addText(
        f"{session_info['participant']}, {session_info['session']} session:\n"
        f"{describe_session(session_info['session'], session_info['start_run'], session_info['reconstruction'], session_info['localizer'])}\n\n"
        f"First two cycles of this participant's sequence: {categories_in_order}\n\n"
        f"Answer grid: {make_conditions.describe_choice_options(choice_options)}"
        + (f"\n\n{design_note}" if design_note else "")
    )
    confirmation.show()
    if not confirmation.OK:
        core.quit()


def run_evening_cycle(task, sequence_rows, trials, cycle, start_run):
    """study -> shades of grey -> recall run, one of the six evening cycles."""
    task.run_stage(f"study_instructions_cycle{cycle}", task.show_message,
                   INSTRUCTIONS["first_study"] if cycle == 1 else INSTRUCTIONS["repeat_study"],
                   ("space",), "Press SPACE to start")
    task.run_stage(f"study_evening_cycle{cycle}", task.run_study_block, sequence_rows, "evening", cycle)
    if settings.RUN_SHADES_BEFORE_EVENING_RECALL:
        task.run_stage(f"shades_evening_cycle{cycle}", task.run_shades_of_grey, "evening", cycle)
    task.run_stage(f"recall_instructions_cycle{cycle}", task.show_message,
                   INSTRUCTIONS["first_run"] if cycle == 1 else INSTRUCTIONS["later_run"],
                   ("space",), "Press SPACE to start")
    task.run_stage(f"recall_evening_cycle{cycle}", task.run_recall_run, trials, sequence_rows, "evening", cycle)


def run_morning_run(task, sequence_rows, trials, cycle, start_run):
    """One of the six morning recall runs (no re-study)."""
    first_of_this_launch = cycle == start_run
    task.run_stage(f"recall_instructions_cycle{cycle}", task.show_message,
                   INSTRUCTIONS["first_run"] if first_of_this_launch else INSTRUCTIONS["later_run"],
                   ("space",), "Press SPACE to start")
    task.run_stage(f"recall_morning_cycle{cycle}", task.run_recall_run, trials, sequence_rows, "morning", cycle)


# ------------------------------------------------------------------ main ---

def main():
    os.chdir(Path(__file__).resolve().parent)

    session_info = show_session_dialog()
    choice_options = get_choice_options(session_info)
    if choice_options["number_of_choices"] < make_conditions.SEQUENCE_LENGTH:
        # with fewer than 50 choices, don't tell participants they will see all the pictures
        INSTRUCTIONS["first_run"] = INSTRUCTIONS["first_run"].replace(
            "Then all the pictures appear:", "Then a set of pictures appears:"
        )
    if session_info["quick_test"]:
        settings.apply_quick_test_overrides()

    participant_id, session, start_run = session_info["participant"], session_info["session"], session_info["start_run"]
    sequence_rows, design_note = load_participant_design(participant_id, choice_options)
    confirm_session(session_info, sequence_rows, choice_options, design_note)

    settings.DATA_FOLDER.mkdir(exist_ok=True)
    date_string = data.getDateStr()
    quick_tag = "_quicktest" if session_info["quick_test"] else ""
    resume_tag = f"_from{start_run}" if start_run > 1 else ""
    file_stem = str(settings.DATA_FOLDER / f"{participant_id}_{session}{resume_tag}{quick_tag}_{date_string}")

    participant_number = make_conditions.get_participant_number(participant_id)
    runtime_seed = make_conditions.get_design_seed(participant_number, session) * 10 + start_run
    experiment_info = {
        **session_info,
        "date": date_string,
        "runtime_seed": runtime_seed,
        "design_seed": make_conditions.DESIGN_SEED,
        "trigger_backend": settings.TRIGGER_BACKEND,
        **{f"setup_{name}": value for name, value in choice_options.items()},
    }
    experiment = data.ExperimentHandler(
        name=EXPERIMENT_NAME, extraInfo=experiment_info, dataFileName=file_stem,
        savePickle=False, saveWideText=True,
    )
    logging.LogFile(file_stem + ".log", level=logging.INFO)

    win = visual.Window(
        size=settings.WINDOW_SIZE, fullscr=settings.FULLSCREEN, screen=settings.SCREEN_NUMBER,
        monitor=settings.MONITOR_NAME, color=settings.BACKGROUND_COLOR, colorSpace="rgb",
        units="height", allowGUI=False,
    )
    win.mouseVisible = False
    experiment_info["frame_rate"] = round(1 / win.monitorFramePeriod, 2)

    markers = EventMarkers(win, file_stem + "_events.csv")
    task = CuedRecallRoutines(win, experiment, markers, random.Random(runtime_seed), file_stem + "_mouse.csv",
                              choice_options["number_of_choices"])
    task.save_grid_layout(file_stem + "_grid_layout.csv")
    task.set_context(participant=participant_id)

    def quit_session():
        markers.send_now("session_end", detail="quit key")
        task.save_checkpoint()
        experiment.abort()  # already saved; stop the handler saving again at exit
        task.close()
        markers.close()
        win.close()
        core.quit()

    event.globalKeys.add(key=settings.QUIT_KEY, modifiers=settings.QUIT_MODIFIERS,
                         func=quit_session, name="quit")

    markers.send_now("session_start", detail=f"{session}_from{start_run}")
    if start_run == 1:
        task.run_stage("welcome", task.show_message,
                       INSTRUCTIONS["evening_welcome"] if session == "evening" else INSTRUCTIONS["morning_welcome"])

    # evening: image localizer before the first study block (not when resuming)
    if session == "evening" and session_info["localizer"] and start_run == 1:
        task.run_stage("localizer_evening", localizer.run_localizer, task, session)

    # morning: full sequence reconstruction before any further cued retrieval
    if session == "morning" and session_info["reconstruction"]:
        task.run_stage("reconstruction_morning", sequence_reconstruction.run_sequence_reconstruction,
                       task, sequence_rows, session)

    for cycle in range(start_run, make_conditions.NUMBER_OF_RUNS + 1):
        recall_file = make_conditions.get_recall_file(participant_id, session, cycle, settings.CONDITIONS_FOLDER)
        trials = make_conditions.load_rows(recall_file)
        if session == "evening":
            run_evening_cycle(task, sequence_rows, trials, cycle, start_run)
        else:
            run_morning_run(task, sequence_rows, trials, cycle, start_run)

    # evening: full sequence reconstruction after the sixth cycle
    if session == "evening" and session_info["reconstruction"]:
        task.run_stage("reconstruction_evening", sequence_reconstruction.run_sequence_reconstruction,
                       task, sequence_rows, session)

    # morning: image localizer last, so it cannot contaminate the memory tests
    if session == "morning" and session_info["localizer"]:
        task.run_stage("localizer_morning", localizer.run_localizer, task, session)

    task.run_stage("end", task.show_message, INSTRUCTIONS["end"], ("space",), "")
    markers.send_now("session_end", detail=f"{session}_from{start_run}")

    task.save_checkpoint()
    experiment.abort()
    task.close()
    markers.close()
    win.close()
    core.quit()


if __name__ == "__main__":
    main()
