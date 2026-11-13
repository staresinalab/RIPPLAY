"""
sequence_reconstruction.py

Full 50-item sequence reconstruction for RIPPLAY, called by cued_recall.py:
    evening    after the sixth study -> shades of grey -> recall cycle
    morning    before the first recall run

All 50 studied images appear in a grid, in one random arrangement that stays
fixed for the whole reconstruction. The participant clicks them in the order
they were shown, starting with the very first picture. A clicked image dims
and cannot be clicked again; the last picked image is shown at the top so the
participant can keep their place; UNDO takes back the last pick. After the
50th pick a DONE button appears. No time limit and no feedback.

Data (same CSV as the rest of the session, phase = "reconstruction")
    one row per pick or undo: reported rank, image, its true position, grid
        slot and coordinates, time since the screen appeared, time since the
        previous action
    one summary row (trial = "summary") with the scores listed in
        score_reconstruction(), plus the grid arrangement

Whether it runs is chosen with the "reconstruction" tick box in the
cued_recall.py start dialog. Everything else specific to the reconstruction
(layout, button labels, EEG codes) lives in this file, so cued_recall.py only
calls run_sequence_reconstruction().
"""

from psychopy import core, visual

import settings
import triggers


# ----------------------------------------------------------------- layout ---

GRID_COLUMNS = 10
GRID_ROWS = 5
GRID_MARGIN = 0.04                   # height units kept free at the left/right/bottom edges
HEADER_HEIGHT = 0.14                 # band at the top for the progress text and last pick
FOOTER_HEIGHT = 0.09                 # band at the bottom for the UNDO / DONE buttons
IMAGE_FILL = 0.9                     # image size as a proportion of its grid cell
PICKED_IMAGE_OPACITY = 0.2           # already-picked images stay visible but dimmed
LAST_PICK_SIZE = 0.09                # thumbnail of the last pick in the header

BUTTON_SIZE = (0.18, 0.06)
BUTTON_FILL_COLOR = [-0.2, -0.2, -0.2]
BUTTON_HOVER_COLOR = [0.2, 0.2, 0.2]
BUTTON_TEXT_HEIGHT = 0.03


# --------------------------------------------------------- EEG triggers ---
# Added to triggers.EVENT_CODES when this module is imported, so triggers.py
# does not need editing. Codes 80-83 are unused by the rest of the task.

RECONSTRUCTION_EVENT_CODES = {
    "reconstruction_screen": 80,
    "reconstruction_pick": 81,
    "reconstruction_undo": 82,
    "reconstruction_done": 83,
}
clashing_codes = set(RECONSTRUCTION_EVENT_CODES.values()) & (
    set(triggers.EVENT_CODES.values()) - set(RECONSTRUCTION_EVENT_CODES.values())
)
if clashing_codes:
    raise ValueError(f"Reconstruction trigger codes {clashing_codes} are already used in triggers.py")
triggers.EVENT_CODES.update(RECONSTRUCTION_EVENT_CODES)


# ---------------------------------------------------------- instructions ---

INSTRUCTIONS = {
    "evening": (
        "Picture order.\n\n"
        "All the pictures will appear on the screen. Click them in the order you saw them, "
        "starting with the very FIRST picture you saw, then the second, and so on.\n\n"
        "Each picture you click fades out. The last one you clicked is shown at the top.\n"
        "Click UNDO to take back your last click.\n\n"
        "Place all the pictures, guessing if you are unsure, then click DONE."
    ),
    "morning": (
        "Picture order.\n\n"
        "All the pictures from last time will appear on the screen. Click them in the order "
        "you saw them, starting with the very FIRST picture, then the second, and so on.\n\n"
        "Each picture you click fades out. The last one you clicked is shown at the top.\n"
        "Click UNDO to take back your last click.\n\n"
        "Place all the pictures, guessing if you are unsure, then click DONE."
    ),
}


# ---------------------------------------------------------------- scoring ---

def score_reconstruction(reported_positions, sequence_rows):
    """Scores for one reconstruction.

    reported_positions[k] is the TRUE position (1-50) of the image the
    participant placed at rank k + 1. Pure Python, so it can be checked
    without PsychoPy.

    exact_position_correct       images placed at exactly their true position
    mean_absolute_displacement   mean |reported rank - true position|
    spearman_rho, kendall_tau    rank correlation between reported and true order
    adjacent_pairs_correct       consecutive picks that are consecutive in the
                                 sequence (k then k + 1), out of 49
    best_circular_shift          the start point that best explains the order
                                 (0 = started at position 1), and the number of
    best_shift_exact_correct     exact matches after that shift; a participant
                                 who started at the wrong place but kept the
                                 order scores high here and low on exact matches
    core_transitions_correct     with decoys removed, consecutive reported
                                 categories that follow the four-category core
                                 order (incl. last -> first), out of 39
    decoy_exact_correct          decoy images at exactly their true position
    decoy_slot_correct           decoy images whose reported rank falls in the
                                 same within-cycle slot (1-5) as their true one
    """
    sequence_length = len(sequence_rows)
    row_by_position = {row["position"]: row for row in sequence_rows}
    reported_ranks = list(range(1, sequence_length + 1))

    exact_position_correct = sum(rank == position for rank, position in zip(reported_ranks, reported_positions))
    displacements = [abs(rank - position) for rank, position in zip(reported_ranks, reported_positions)]

    squared_rank_differences = sum(displacement ** 2 for displacement in displacements)
    spearman_rho = 1 - 6 * squared_rank_differences / (sequence_length * (sequence_length ** 2 - 1))

    concordant = discordant = 0
    for first_rank in range(sequence_length):
        for second_rank in range(first_rank + 1, sequence_length):
            if reported_positions[first_rank] < reported_positions[second_rank]:
                concordant += 1
            else:
                discordant += 1
    kendall_tau = (concordant - discordant) / (sequence_length * (sequence_length - 1) / 2)

    adjacent_pairs_correct = sum(
        reported_positions[rank + 1] == reported_positions[rank] + 1 for rank in range(sequence_length - 1)
    )

    exact_correct_by_shift = {
        shift: sum(
            position == (rank - 1 + shift) % sequence_length + 1
            for rank, position in zip(reported_ranks, reported_positions)
        )
        for shift in range(sequence_length)
    }
    best_circular_shift = max(exact_correct_by_shift, key=exact_correct_by_shift.get)

    core_order = [row["category_code"] for row in sequence_rows if not row["is_decoy"]][:4]
    next_core_category = {category: core_order[(index + 1) % 4] for index, category in enumerate(core_order)}
    reported_core_categories = [
        row_by_position[position]["category_code"] for position in reported_positions
        if not row_by_position[position]["is_decoy"]
    ]
    core_transitions_correct = sum(
        next_core_category[reported_core_categories[index]] == reported_core_categories[index + 1]
        for index in range(len(reported_core_categories) - 1)
    )

    decoy_exact_correct = decoy_slot_correct = 0
    for rank, position in zip(reported_ranks, reported_positions):
        if row_by_position[position]["is_decoy"]:
            decoy_exact_correct += rank == position
            decoy_slot_correct += (rank - 1) % 5 == (position - 1) % 5

    return {
        "exact_position_correct": exact_position_correct,
        "mean_absolute_displacement": round(sum(displacements) / sequence_length, 4),
        "spearman_rho": round(spearman_rho, 4),
        "kendall_tau": round(kendall_tau, 4),
        "adjacent_pairs_correct": adjacent_pairs_correct,
        "best_circular_shift": best_circular_shift,
        "best_shift_exact_correct": exact_correct_by_shift[best_circular_shift],
        "core_transitions_correct": core_transitions_correct,
        "decoy_exact_correct": decoy_exact_correct,
        "decoy_slot_correct": decoy_slot_correct,
    }


# ----------------------------------------------------------------- screen ---

class ReconstructionScreen:
    """The grid, header and buttons. Uses its own image stimuli, so dimming
    picked images never affects the images used by the rest of the task."""

    def __init__(self, task, sequence_rows):
        self.task = task
        self.win = task.win
        self.sequence_rows = sequence_rows

        # one random arrangement, fixed for the whole reconstruction
        self.grid_rows = task.rng.sample(sequence_rows, len(sequence_rows))
        self.grid_positions, self.image_size = self.get_grid_layout()
        self.image_stims = [
            visual.ImageStim(
                self.win, image=str(settings.STIMULUS_ROOT / row["image_file"]),
                pos=position, size=self.image_size, interpolate=True,
            )
            for row, position in zip(self.grid_rows, self.grid_positions)
        ]

        header_y_position = 0.5 - HEADER_HEIGHT / 2
        self.progress_text = visual.TextStim(
            self.win, text="", height=settings.TEXT_HEIGHT, color=settings.TEXT_COLOR,
            pos=(0, header_y_position), font="Arial",
        )
        self.last_pick_label = visual.TextStim(
            self.win, text="Last:", height=settings.TEXT_HEIGHT * 0.8, color=settings.TEXT_COLOR,
            pos=(-0.62, header_y_position), font="Arial", alignText="right", anchorHoriz="right",
        )
        self.last_pick_image = visual.ImageStim(
            self.win, image=None, pos=(-0.62 + 0.02 + LAST_PICK_SIZE / 2, header_y_position),
            size=(LAST_PICK_SIZE, LAST_PICK_SIZE), interpolate=True,
        )

        footer_y_position = -0.5 + FOOTER_HEIGHT / 2
        self.undo_button = self.make_button("UNDO", (-0.15, footer_y_position))
        self.done_button = self.make_button("DONE", (0.15, footer_y_position))
        self.highlight_frame = visual.Rect(
            self.win, width=self.image_size[0] * 1.08, height=self.image_size[1] * 1.08,
            lineColor=settings.TEXT_COLOR, fillColor=None, lineWidth=4,
        )

    def get_grid_layout(self):
        """Centres of the 50 cells (slot 1 = top left, row by row) between the
        header and footer bands, and the image size."""
        aspect_ratio = self.win.size[0] / self.win.size[1]
        usable_width = aspect_ratio - 2 * GRID_MARGIN
        usable_height = 1 - HEADER_HEIGHT - FOOTER_HEIGHT - GRID_MARGIN
        cell_size = min(usable_width / GRID_COLUMNS, usable_height / GRID_ROWS)
        grid_centre_y = (FOOTER_HEIGHT - HEADER_HEIGHT) / 2

        positions = []
        for row in range(GRID_ROWS):
            y_position = grid_centre_y + (GRID_ROWS - 1) / 2 * cell_size - row * cell_size
            for column in range(GRID_COLUMNS):
                x_position = -(GRID_COLUMNS - 1) / 2 * cell_size + column * cell_size
                positions.append((x_position, y_position))
        image_size = cell_size * IMAGE_FILL
        return positions, (image_size, image_size)

    def make_button(self, label, position):
        box = visual.Rect(
            self.win, width=BUTTON_SIZE[0], height=BUTTON_SIZE[1], pos=position,
            fillColor=BUTTON_FILL_COLOR, lineColor=settings.TEXT_COLOR, lineWidth=2,
        )
        text = visual.TextStim(
            self.win, text=label, height=BUTTON_TEXT_HEIGHT, color=settings.TEXT_COLOR,
            pos=position, font="Arial", bold=True,
        )
        return {"box": box, "text": text}

    def get_pixel_coordinates(self, position):
        """Height units (origin centre, y up) -> pixels (origin top left, y down)."""
        window_width, window_height = self.win.size
        return (
            round(window_width / 2 + position[0] * window_height, 1),
            round(window_height / 2 - position[1] * window_height, 1),
        )

    def draw(self, picked_slots, hovered_slot, hovered_button):
        number_picked = len(picked_slots)
        if number_picked < len(self.grid_rows):
            self.progress_text.text = f"Picture {number_picked + 1} of {len(self.grid_rows)}: click the next picture"
        else:
            self.progress_text.text = "All pictures placed. Click DONE to finish, or UNDO to change the last one."

        stimuli = [self.progress_text]
        for slot, image_stim in enumerate(self.image_stims, start=1):
            image_stim.opacity = PICKED_IMAGE_OPACITY if slot in picked_slots else 1.0
            stimuli.append(image_stim)
        if hovered_slot is not None:
            self.highlight_frame.pos = self.grid_positions[hovered_slot - 1]
            stimuli.append(self.highlight_frame)

        if picked_slots:
            self.last_pick_image.image = str(settings.STIMULUS_ROOT / self.grid_rows[picked_slots[-1] - 1]["image_file"])
            stimuli += [self.last_pick_label, self.last_pick_image]

        visible_buttons = []
        if picked_slots:
            visible_buttons.append(("undo", self.undo_button))
        if number_picked == len(self.grid_rows):
            visible_buttons.append(("done", self.done_button))
        for name, button in visible_buttons:
            button["box"].fillColor = BUTTON_HOVER_COLOR if name == hovered_button else BUTTON_FILL_COLOR
            stimuli += [button["box"], button["text"]]

        for stimulus in stimuli:
            stimulus.draw()
        self.task.last_flip_time = self.win.flip()
        self.task.check_for_skip()   # F10 ends the reconstruction (see cued_recall.SkipStage)
        return self.task.last_flip_time, [name for name, _ in visible_buttons]

    def find_hovered(self, picked_slots, visible_buttons):
        """(grid slot or None, button name or None) under the mouse. Picked
        images and hidden buttons cannot be hovered."""
        mouse = self.task.mouse
        for name, button in (("undo", self.undo_button), ("done", self.done_button)):
            if name in visible_buttons and button["box"].contains(mouse):
                return None, name
        if len(picked_slots) < len(self.grid_rows):
            for slot, image_stim in enumerate(self.image_stims, start=1):
                if slot not in picked_slots and image_stim.contains(mouse):
                    return slot, None
        return None, None


def run_sequence_reconstruction(task, sequence_rows, session):
    """Show the reconstruction screen until the participant clicks DONE with
    all 50 images placed; log every pick/undo and a scored summary row."""
    task.set_context(phase="reconstruction", session=session, cycle="", trial="")
    task.show_message(INSTRUCTIONS[session], footer="Press SPACE to start")

    screen = ReconstructionScreen(task, sequence_rows)
    position_by_slot = {slot: row["position"] for slot, row in enumerate(screen.grid_rows, start=1)}

    picked_slots = []
    action_number = 0
    undo_count = 0
    task.mouse.setPos(settings.MOUSE_START_POSITION)
    task.win.mouseVisible = True
    task.markers.send_on_next_flip("reconstruction_screen", detail=session)
    screen_onset, visible_buttons = screen.draw(picked_slots, None, None)
    last_action_time = screen_onset
    button_released = not task.mouse.getPressed()[0]

    while True:
        hovered_slot, hovered_button = screen.find_hovered(picked_slots, visible_buttons)
        button_down = task.mouse.getPressed()[0]

        if button_down and button_released and (hovered_slot is not None or hovered_button is not None):
            button_released = False
            action_time = core.getTime()
            action_number += 1
            task.set_context(trial=action_number)

            if hovered_button == "done":
                task.markers.send_now("reconstruction_done", detail=session)
                done_time = action_time
                break

            if hovered_button == "undo":
                undo_count += 1
                undone_slot = picked_slots.pop()
                task.markers.send_now("reconstruction_undo", detail=f"slot_{undone_slot}")
                action, slot = "undo", undone_slot
            else:
                picked_slots.append(hovered_slot)
                task.markers.send_now("reconstruction_pick", detail=f"slot_{hovered_slot}")
                action, slot = "pick", hovered_slot

            row = screen.grid_rows[slot - 1]
            x_pixels, y_pixels = screen.get_pixel_coordinates(screen.grid_positions[slot - 1])
            task.log_row(
                reconstruction_action=action,
                reported_rank=len(picked_slots) if action == "pick" else len(picked_slots) + 1,
                image_file=row["image_file"],
                image_id=row["image_id"],
                position=row["position"],
                category=row["category"],
                is_decoy=row["is_decoy"],
                reconstruction_grid_slot=slot,
                reconstruction_grid_x_pixels=x_pixels,
                reconstruction_grid_y_pixels=y_pixels,
                time_from_screen_onset=round(action_time - screen_onset, 4),
                time_from_previous_action=round(action_time - last_action_time, 4),
            )
            last_action_time = action_time
        elif not button_down:
            button_released = True

        _, visible_buttons = screen.draw(picked_slots, hovered_slot, hovered_button)

    task.win.mouseVisible = False
    task.draw_and_flip([])

    reported_positions = [position_by_slot[slot] for slot in picked_slots]
    scores = score_reconstruction(reported_positions, sequence_rows)
    task.log_row(
        trial="summary",
        reported_order_positions=" ".join(str(position) for position in reported_positions),
        reported_order_image_ids=" ".join(screen.grid_rows[slot - 1]["image_id"] for slot in picked_slots),
        grid_arrangement_image_ids=" ".join(row["image_id"] for row in screen.grid_rows),
        reconstruction_duration=round(done_time - screen_onset, 4),
        undo_count=undo_count,
        **scores,
    )
    task.save_checkpoint()
    return scores
