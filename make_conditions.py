"""
make_conditions.py

RIPPLAY cued-recall design -> PsychoPy conditions files. Pure Python, no
PsychoPy needed, so it can be run and checked on any machine.

Inputs
    design/ripplay_25_participant_assignments_v3.csv
        decoy, four-category core order and the complete 50-position
        category sequence for each of the 25 participants
    stimuli/stimulus_list.csv
        10 images per category (face, object, scene, word, body)

Outputs (in conditions/)
    P01_sequence.csv
        the 50-image study order: position, cycle, slot, category, decoy flag,
        image file
    P01_evening_run1.csv ... P01_morning_run6.csv
        50 cued-recall trials each, one row per trial, with the cue, offset,
        target and the randomised 50-image choice grid (choice_01 ... choice_50)

Boundary-agnostic offsets
    The 50-image sequence is treated as a circle: position 50 is followed by
    position 1. Every cue can then be probed at all six offsets (-3, -2, -1,
    +1, +2, +3), so every starting position gets exactly three forward and
    three backward questions per session and every position is the target
    exactly six times. Trials whose target lies across the 50 -> 1 join are
    flagged (wraps_sequence_end) so they can be modelled or excluded.

    Per session: each run probes all 50 cues once, with 8 or 9 trials at each
    offset; across the six runs every cue x offset pair occurs exactly once
    and every offset occurs exactly 50 times. The evening and morning
    schedules and grids are drawn independently.

Wrap-around only once the join has been seen
    The 12 cue x offset pairs that cross the 50 -> 1 join are never asked
    before the participant has seen position 1 follow position 50. In the
    evening that first happens when study block 2 starts (block 1 ended on
    position 50), so evening recall run 1 has no wrap-around trials and runs
    2-6 take all 12. Morning runs come after all six study blocks, so they
    are unrestricted. See FIRST_RUN_ALLOWING_WRAP.

Exemplars
    The assignments file is category-level. The k-th occurrence of a category
    gets exemplar base_order[(k + participant - 1) mod 10], where base_order
    is one fixed random order per category. Across 25 participants every
    exemplar occupies every occurrence (cycle) 2 or 3 times. Every image also
    gets an image_id in the localiser file's format (e.g. "stimuli/o (07).jpg"
    -> O07), so recall and localiser data join on the same key.

Choice grid
    The number of answer choices and how the foils are sampled are set in
    the start dialog (choice_options; defaults in DEFAULT_CHOICE_OPTIONS):
        50 choices, random   all 50 studied images, cue included (the original grid)
        n < 50, random       the target plus n-1 foils drawn uniformly from the
                             other studied images; the cue is never offered
        n < 50, bands        the target plus foils built around the cue's
                             position on the circle (see choose_band_positions)
        n < 50, category_bands
                             the same near band, plus equal numbers of images
                             per category (n / 5 each; see
                             choose_category_band_positions)
    A fresh random arrangement on every trial. Within each run the target
    lands in every grid slot equally often (once each with 50 choices), so
    target location (centre vs edge, left vs right) is balanced and cannot
    bias choice RT or gaze measures.

    Grids are drawn from their own random generator, after all trial orders
    are fixed, so changing the grid settings never changes which questions
    are asked or in what order. The sequence file records the grid settings
    (choice_settings); cued_recall.py regenerates the conditions files when
    they were made with different settings.

Run
    python make_conditions.py            all 25 participants
    python make_conditions.py P01 P07    selected participants
"""

import csv
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import settings


CATEGORY_NAMES = {"F": "face", "O": "object", "S": "scene", "W": "word", "B": "body"}
OFFSETS = [-3, -2, -1, 1, 2, 3]
SEQUENCE_LENGTH = 50
CYCLE_LENGTH = 5
NUMBER_OF_CYCLES = 10
NUMBER_OF_RUNS = 6
SESSIONS = ["evening", "morning"]

DESIGN_SEED = 20261026               # fixes exemplar base orders and all schedules

# Written into every sequence file. cued_recall.py regenerates a participant's
# conditions files when theirs carry a different version, so files made by an
# older version of this script can never be run by mistake. Change it whenever
# the generated design changes.
DESIGN_VERSION = "2026-10-30_adjustable_choice_grid"

# Answer choices on the recall grid (set in the start dialog; see the docstring).
CHOICE_SAMPLING_STRATEGIES = ["random", "bands", "category_bands"]
# the band sizes each strategy asks for in the start dialog
BAND_SIZE_NAMES_BY_STRATEGY = {
    "bands": [
        "near_band_radius", "mid_band_radius", "mid_band_count",
        "camouflage_run_length", "camouflage_min_distance", "far_min_distance",
    ],
    "category_bands": [
        "near_band_radius", "nearest_per_category_side",
        "camouflage_cycles", "camouflage_min_distance", "far_min_distance",
    ],
}
BAND_SIZE_NAMES = BAND_SIZE_NAMES_BY_STRATEGY["bands"]
DEFAULT_CHOICE_OPTIONS = {
    "number_of_choices": 50,
    "sampling": "random",
    # band sizes, used only with 'bands'; distances are positions on the circle from the cue
    "near_band_radius": 3,          # every image within +-3 of the cue (minus the target)
    "mid_band_radius": 6,           # the mid band covers distances near_band_radius+1 .. 6
    "mid_band_count": 3,            # mid-band images, split across both sides of the cue
    "camouflage_run_length": 6,     # one run of consecutive images elsewhere (0 = none)
    "camouflage_min_distance": 10,  # nearest camouflage image is at least this far from the cue
    "far_min_distance": 7,          # far images fill the rest, at least this far from the cue
    # category_bands only (near_band_radius, camouflage_min_distance and far_min_distance are shared)
    "nearest_per_category_side": 1, # every category's nearest image(s) before and after the cue
    "camouflage_cycles": 1,         # whole far five-image cycles (one image of each category each)
}
GRID_SEED_OFFSET = 50              # grid generator seed = session design seed + this

# First recall run (per session) that may ask a question across the 50 -> 1
# join. Evening run 1 follows study block 1 only, where position 50 has never
# been followed by position 1; from study block 2 on, it has.
FIRST_RUN_ALLOWING_WRAP = {"evening": 2, "morning": 1}
MAX_SCHEDULE_ATTEMPTS = 10000

# trial-order constraints within a run
MAX_SAME_OFFSET_IN_A_ROW = 2
MAX_ORDERING_ATTEMPTS = 10000


# ---------------------------------------------------------------- files ---

def get_participant_number(participant_id):
    """'P07' -> 7"""
    return int(participant_id.strip().upper().lstrip("P"))


def get_participant_id(participant_number):
    """7 -> 'P07'"""
    return f"P{participant_number:02d}"


def get_sequence_file(participant_id, conditions_folder):
    return Path(conditions_folder) / f"{participant_id}_sequence.csv"


def get_recall_file(participant_id, session, run_number, conditions_folder):
    return Path(conditions_folder) / f"{participant_id}_{session}_run{run_number}.csv"


def load_assignments(assignments_file):
    with open(assignments_file, newline="") as csv_file:
        return {int(row["participant"]): row for row in csv.DictReader(csv_file)}


def get_category_sequence(participant_id, assignments_file=settings.ASSIGNMENTS_FILE):
    """The participant's 50-letter category sequence, e.g. 'FOSBW...'."""
    return load_assignments(assignments_file)[get_participant_number(participant_id)]["category_sequence_50"]


def load_stimulus_list(stimulus_list_file):
    with open(stimulus_list_file, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_rows(conditions_file):
    """Rows of a conditions file, with whole numbers converted back to int."""
    with open(conditions_file, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    for row in rows:
        for column, value in row.items():
            if value.lstrip("-").isdigit():
                row[column] = int(value)
    return rows


def write_rows(rows, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------- sequence ---

def check_assignment(assignment):
    """Each five-image cycle must be the core order with the decoy inserted at
    the listed slot, and each slot must hold the decoy exactly twice."""
    category_sequence = assignment["category_sequence_50"]
    core_order = assignment["core_order"]
    decoy = assignment["decoy"]
    decoy_slots = [int(slot) for slot in assignment["decoy_slot_by_cycle"].split(",")]

    problems = []
    if len(category_sequence) != SEQUENCE_LENGTH:
        problems.append(f"sequence has {len(category_sequence)} positions")
    if sorted(Counter(decoy_slots).values()) != [2] * CYCLE_LENGTH:
        problems.append(f"decoy slots are not each used twice: {decoy_slots}")
    for cycle_index, decoy_slot in enumerate(decoy_slots):
        cycle_categories = category_sequence[cycle_index * CYCLE_LENGTH:(cycle_index + 1) * CYCLE_LENGTH]
        expected = core_order[:decoy_slot - 1] + decoy + core_order[decoy_slot - 1:]
        if cycle_categories != expected:
            problems.append(f"cycle {cycle_index + 1} is {cycle_categories}, expected {expected}")
    if problems:
        raise ValueError(f"Participant {assignment['participant']}: " + "; ".join(problems))


def get_exemplar_base_orders(stimuli):
    """One fixed random order of the 10 exemplars of each category."""
    rng = random.Random(DESIGN_SEED)
    base_orders = {}
    for category_code, category_name in CATEGORY_NAMES.items():
        image_files = sorted(image["image_file"] for image in stimuli if image["category"] == category_name)
        if len(image_files) != NUMBER_OF_CYCLES:
            raise ValueError(f"Expected {NUMBER_OF_CYCLES} {category_name} images, found {len(image_files)}")
        base_orders[category_code] = rng.sample(image_files, len(image_files))
    return base_orders


def get_image_id(image_file, category_code):
    """Image ID in the localiser file's format: 'stimuli/o (07).jpg' -> 'O07'.

    Uses the number in brackets in the file name, else the last number in it;
    falls back to the file name itself if it has no number.
    """
    file_name = Path(image_file).stem
    bracketed_number = re.search(r"\((\d+)\)", file_name)
    any_number = re.findall(r"\d+", file_name)
    if bracketed_number:
        return f"{category_code}{int(bracketed_number.group(1)):02d}"
    if any_number:
        return f"{category_code}{int(any_number[-1]):02d}"
    return file_name


def build_sequence_rows(participant_number, assignment, base_orders, choice_options):
    """The 50-image study order for one participant (with the grid settings its
    recall files were made with, so cued_recall.py can spot a mismatch)."""
    category_sequence = assignment["category_sequence_50"]
    occurrences_so_far = Counter()
    sequence_rows = []
    for position, category_code in enumerate(category_sequence, start=1):
        occurrence = occurrences_so_far[category_code]
        occurrences_so_far[category_code] += 1
        exemplar_index = (occurrence + participant_number - 1) % NUMBER_OF_CYCLES
        image_file = base_orders[category_code][exemplar_index]
        sequence_rows.append({
            "position": position,
            "cycle": (position - 1) // CYCLE_LENGTH + 1,
            "slot_in_cycle": (position - 1) % CYCLE_LENGTH + 1,
            "category_code": category_code,
            "category": CATEGORY_NAMES[category_code],
            "is_decoy": int(category_code == assignment["decoy"]),
            "category_occurrence": occurrence + 1,
            "image_id": get_image_id(image_file, category_code),
            "image_file": image_file,
            "design_version": DESIGN_VERSION,
            "choice_settings": get_choice_settings_label(choice_options),
        })

    image_ids = [row["image_id"] for row in sequence_rows]
    if len(set(image_ids)) != SEQUENCE_LENGTH:
        raise ValueError(f"Image IDs are not unique: {sorted(image_ids)}. Check the image file names.")
    return sequence_rows


# ---------------------------------------------------------- recall runs ---

def wrap_position(position):
    """Positions on the circle: 51 -> 1, 0 -> 50."""
    return (position - 1) % SEQUENCE_LENGTH + 1


def get_cycle(position):
    return (position - 1) // CYCLE_LENGTH + 1


def crosses_sequence_end(cue_position, offset):
    """True when the target lies across the 50 -> 1 join."""
    return not 1 <= cue_position + offset <= SEQUENCE_LENGTH


def get_design_seed(participant_number, session):
    return DESIGN_SEED + participant_number * 100 + SESSIONS.index(session)


def build_offset_schedule(rng):
    """{run number: [(cue position, offset), ...]} for one session.

    Cues are put in a random order and the six offsets in a random order;
    the cue at rank k gets offset number (k + rotation) mod 6 in each run, with
    a different rotation for every run. So every cue walks through all six
    offsets across the six runs, and the two offsets that get a ninth trial
    (50 = 6 x 8 + 2) change from run to run, giving 50 of each per session.
    """
    cue_order = rng.sample(range(1, SEQUENCE_LENGTH + 1), SEQUENCE_LENGTH)
    offset_order = rng.sample(OFFSETS, len(OFFSETS))
    rotations = rng.sample(range(len(OFFSETS)), NUMBER_OF_RUNS)

    schedule = {}
    for run_number, rotation in enumerate(rotations, start=1):
        schedule[run_number] = [
            (cue_position, offset_order[(rank + rotation) % len(OFFSETS)])
            for rank, cue_position in enumerate(cue_order)
        ]
    return schedule


def schedule_respects_wrap_rule(schedule, session):
    """No question across the 50 -> 1 join before that join has been seen."""
    for run_number, cue_offset_pairs in schedule.items():
        if run_number >= FIRST_RUN_ALLOWING_WRAP[session]:
            continue
        if any(crosses_sequence_end(cue_position, offset) for cue_position, offset in cue_offset_pairs):
            return False
    return True


def build_session_schedule(session, rng):
    """Redraw the offset schedule until early runs hold no wrap-around pairs.

    Only the six edge cues (1-3, 48-50) can wrap, so roughly one draw in
    thirteen already passes for the evening; the balance of the schedule is
    untouched because whole schedules are accepted or rejected.
    """
    for _ in range(MAX_SCHEDULE_ATTEMPTS):
        schedule = build_offset_schedule(rng)
        if schedule_respects_wrap_rule(schedule, session):
            return schedule
    raise RuntimeError(f"Could not find a {session} schedule that respects the wrap-around rule")


def order_is_acceptable(trials):
    """No offset more than twice in a row; no cue that was just the previous
    target (the answer would be on screen again); no target twice in a row."""
    for trial_index in range(1, len(trials)):
        previous_trial, trial = trials[trial_index - 1], trials[trial_index]
        if trial["cue_position"] == previous_trial["target_position"]:
            return False
        if trial["target_position"] == previous_trial["target_position"]:
            return False
        if trial_index >= MAX_SAME_OFFSET_IN_A_ROW:
            recent_offsets = {trials[earlier]["offset"] for earlier in range(trial_index - MAX_SAME_OFFSET_IN_A_ROW, trial_index + 1)}
            if len(recent_offsets) == 1:
                return False
    return True


def order_trials(trials, rng):
    for _ in range(MAX_ORDERING_ATTEMPTS):
        shuffled = rng.sample(trials, len(trials))
        if order_is_acceptable(shuffled):
            return shuffled
    raise RuntimeError("Could not find a trial order that meets the constraints")


def build_recall_trial(participant_id, session, run_number, cue_position, offset, row_by_position):
    target_position = wrap_position(cue_position + offset)
    cue_row, target_row = row_by_position[cue_position], row_by_position[target_position]
    return {
        "participant": participant_id,
        "session": session,
        "recall_run": run_number,
        "trial": None,                       # filled in after ordering
        "cue_position": cue_position,
        "offset": offset,
        "distance": abs(offset),
        "direction": "forward" if offset > 0 else "backward",
        "target_position": target_position,
        "wraps_sequence_end": int(crosses_sequence_end(cue_position, offset)),
        "cue_cycle": get_cycle(cue_position),
        "target_cycle": get_cycle(target_position),
        "crosses_five_item_boundary": int(get_cycle(cue_position) != get_cycle(target_position)),
        "cue_image": cue_row["image_file"],
        "cue_image_id": cue_row["image_id"],
        "cue_category": cue_row["category"],
        "cue_is_decoy": cue_row["is_decoy"],
        "target_image": target_row["image_file"],
        "target_image_id": target_row["image_id"],
        "target_category": target_row["category"],
        "target_is_decoy": target_row["is_decoy"],
    }


def get_circular_lag(position, cue_position):
    """Signed distance from the cue on the circle, -25 ... +24 (as chosen_offset_from_cue)."""
    return (position - cue_position + SEQUENCE_LENGTH // 2) % SEQUENCE_LENGTH - SEQUENCE_LENGTH // 2


def get_choice_settings_label(choice_options):
    """Short label for the grid settings, e.g. 'n50_random' or 'n20_bands_near3_mid6x3_camo6at10_far7'."""
    label = f"n{choice_options['number_of_choices']}_{choice_options['sampling']}"
    if choice_options["sampling"] == "bands":
        label += (
            f"_near{choice_options['near_band_radius']}"
            f"_mid{choice_options['mid_band_radius']}x{choice_options['mid_band_count']}"
            f"_camo{choice_options['camouflage_run_length']}at{choice_options['camouflage_min_distance']}"
            f"_far{choice_options['far_min_distance']}"
        )
    if choice_options["sampling"] == "category_bands":
        label += (
            f"_near{choice_options['near_band_radius']}"
            f"_side{choice_options['nearest_per_category_side']}"
            f"_cyc{choice_options['camouflage_cycles']}at{choice_options['camouflage_min_distance']}"
            f"_far{choice_options['far_min_distance']}"
        )
    return label


def count_band_foils(choice_options):
    """Near-band foils (one of the 2 * radius near images is the target whenever
    the offset is within the radius) and the far foils left over."""
    near_foils = 2 * choice_options["near_band_radius"] - 1
    fixed_foils = near_foils + choice_options["mid_band_count"] + choice_options["camouflage_run_length"]
    far_foils = choice_options["number_of_choices"] - 1 - fixed_foils
    return near_foils, far_foils


def check_choice_options(choice_options, category_sequence=None):
    """List of problems with the grid settings (empty if they work).

    category_sequence (the participant's 50-letter category string) lets the
    category_bands check confirm that no category's near images exceed its
    share of the grid for any cue.
    """
    problems = []
    number_of_choices = choice_options["number_of_choices"]
    sampling = choice_options["sampling"]
    half_circle = SEQUENCE_LENGTH // 2

    if sampling not in CHOICE_SAMPLING_STRATEGIES:
        problems.append(f"Unknown sampling strategy '{sampling}'.")
    if not 2 <= number_of_choices <= SEQUENCE_LENGTH:
        problems.append(f"The number of choices must be 2-{SEQUENCE_LENGTH}.")
    if sampling in ("bands", "category_bands") and number_of_choices == SEQUENCE_LENGTH:
        problems.append("With 50 choices every image is on screen, so bands make no difference: "
                        "pick 'random' (the original grid) or fewer choices.")
    if sampling == "category_bands":
        problems += check_category_band_options(choice_options, category_sequence)

    if sampling == "bands":
        near_radius = choice_options["near_band_radius"]
        mid_radius = choice_options["mid_band_radius"]
        if any(choice_options[name] < 0 for name in BAND_SIZE_NAMES):
            problems.append("Band sizes cannot be negative.")
        if near_radius < max(OFFSETS):
            problems.append(f"The near-band radius must be at least {max(OFFSETS)}, so the target "
                            f"always sits inside the near band and its presence gives nothing away.")
        if mid_radius < near_radius:
            problems.append("The mid-band radius must be at least the near-band radius.")
        largest_mid_band = 2 * (mid_radius - near_radius)
        if choice_options["mid_band_count"] > largest_mid_band:
            problems.append(f"The mid band holds at most {largest_mid_band} images with these radii.")
        if not mid_radius < choice_options["far_min_distance"] <= half_circle:
            problems.append(f"The far minimum distance must be above the mid-band radius and at most "
                            f"{half_circle} (the furthest any image is from the cue on the circle).")
        if choice_options["camouflage_run_length"]:
            if choice_options["camouflage_min_distance"] <= mid_radius:
                problems.append("The camouflage minimum distance must be above the mid-band radius.")
            room_for_run = SEQUENCE_LENGTH + 1 - 2 * choice_options["camouflage_min_distance"]
            if choice_options["camouflage_run_length"] > room_for_run:
                problems.append(f"Only {max(room_for_run, 0)} consecutive images are at least "
                                f"{choice_options['camouflage_min_distance']} from the cue: shorten the "
                                f"camouflage run or lower its minimum distance.")
        near_foils, far_foils = count_band_foils(choice_options)
        if far_foils < 0:
            problems.append(
                f"Near ({near_foils}) + mid ({choice_options['mid_band_count']}) + camouflage "
                f"({choice_options['camouflage_run_length']}) foils do not fit beside the target in "
                f"{number_of_choices} choices. Use more choices or smaller bands."
            )
    return problems


def get_category_near_counts(cue_position, category_by_position, choice_options):
    """{category code: images it gets from the near band and the nearest-per-side top-up}."""
    near_radius = choice_options["near_band_radius"]
    near_positions = {
        wrap_position(cue_position + lag) for lag in range(-near_radius, near_radius + 1) if lag
    }
    for category_code in CATEGORY_NAMES:
        near_positions.update(get_nearest_positions(cue_position, category_code, category_by_position,
                                                    choice_options["nearest_per_category_side"]))
    return Counter(category_by_position[position] for position in near_positions)


def check_category_band_options(choice_options, category_sequence=None):
    """Problems specific to category_bands (see choose_category_band_positions)."""
    problems = []
    number_of_choices = choice_options["number_of_choices"]
    number_of_categories = len(CATEGORY_NAMES)
    near_radius = choice_options["near_band_radius"]
    half_circle = SEQUENCE_LENGTH // 2

    if number_of_choices % number_of_categories:
        problems.append(f"'category_bands' needs a multiple of {number_of_categories} choices "
                        f"(the same number from every category).")
        return problems
    images_per_category = number_of_choices // number_of_categories
    if any(choice_options[name] < 0 for name in BAND_SIZE_NAMES_BY_STRATEGY["category_bands"]):
        problems.append("Band sizes cannot be negative.")
    if near_radius < max(OFFSETS):
        problems.append(f"The near-band radius must be at least {max(OFFSETS)}, so the target "
                        f"always sits inside the near band and its presence gives nothing away.")
    if not near_radius < choice_options["far_min_distance"] <= half_circle:
        problems.append(f"The far minimum distance must be above the near-band radius and at most {half_circle}.")
    if choice_options["camouflage_cycles"]:
        if choice_options["camouflage_min_distance"] <= near_radius:
            problems.append("The camouflage minimum distance must be above the near-band radius.")
        # cycles that can touch the stretch of positions closer than the minimum distance
        close_positions = 2 * choice_options["camouflage_min_distance"] - 1
        far_cycles = NUMBER_OF_CYCLES - (math.ceil(close_positions / CYCLE_LENGTH) + 1)
        if choice_options["camouflage_cycles"] > far_cycles:
            problems.append(f"Only {max(far_cycles, 0)} whole cycles are always at least "
                            f"{choice_options['camouflage_min_distance']} from the cue.")

    if category_sequence and not problems:
        category_by_position = {
            position: category_code for position, category_code in enumerate(category_sequence, start=1)
        }
        most_near_images = max(
            count
            for cue_position in range(1, SEQUENCE_LENGTH + 1)
            for count in get_category_near_counts(cue_position, category_by_position, choice_options).values()
        )
        needed = most_near_images + choice_options["camouflage_cycles"]
        if needed > images_per_category:
            problems.append(
                f"For some cues one category already gets {most_near_images} near image(s), plus "
                f"{choice_options['camouflage_cycles']} camouflage, more than its {images_per_category} "
                f"of {number_of_choices}. Use more choices, a smaller near band or fewer camouflage cycles."
            )
    return problems


def describe_choice_options(choice_options):
    """One line for the check screen and the summary printout."""
    number_of_choices = choice_options["number_of_choices"]
    chance = f"chance {100 / number_of_choices:.0f}%"
    if number_of_choices == SEQUENCE_LENGTH:
        return f"all {SEQUENCE_LENGTH} images, cue included ({chance})"
    if choice_options["sampling"] == "random":
        return f"{number_of_choices} choices: target + {number_of_choices - 1} random foils, no cue ({chance})"
    if choice_options["sampling"] == "category_bands":
        return (
            f"{number_of_choices} choices, {number_of_choices // len(CATEGORY_NAMES)} per category: "
            f"near +-{choice_options['near_band_radius']}, each category's nearest "
            f"{choice_options['nearest_per_category_side']} on each side, "
            f"{choice_options['camouflage_cycles']} far cycle(s) as camouflage "
            f"(distance >= {choice_options['camouflage_min_distance']}), rest far "
            f"(distance >= {choice_options['far_min_distance']}), no cue ({chance})"
        )
    near_foils, far_foils = count_band_foils(choice_options)
    return (
        f"{number_of_choices} choices, bands: near +-{choice_options['near_band_radius']} ({near_foils} foils), "
        f"{choice_options['mid_band_count']} mid (distance {choice_options['near_band_radius'] + 1}-"
        f"{choice_options['mid_band_radius']}), camouflage run of {choice_options['camouflage_run_length']} "
        f"(distance >= {choice_options['camouflage_min_distance']}), {far_foils} far "
        f"(distance >= {choice_options['far_min_distance']}), no cue ({chance})"
    )


def choose_band_positions(cue_position, target_position, choice_options, rng):
    """Foil positions around the cue on the circle: {position: band}.

    near:       every position within +-near_band_radius of the cue
    mid:        mid_band_count positions at distances near_band_radius+1 ..
                mid_band_radius, split as evenly as possible between the sides
    camouflage: one run of camouflage_run_length consecutive positions, all at
                least camouflage_min_distance from the cue, so a cluster of
                sequence neighbours on screen does not single out the cue's
                neighbourhood
    far:        the rest, at least far_min_distance from the cue

    The bands are centred on the CUE, never on the target: the target always
    sits inside the near band (radius >= 3), so which near images are present
    says nothing about the answer. On the circle the near band is always
    complete; near the 50 -> 1 join it reaches across the join.
    """
    band_by_position = {}
    used_positions = {cue_position, target_position}

    def take(positions, band):
        for position in positions:
            used_positions.add(position)
            band_by_position[position] = band

    near_radius = choice_options["near_band_radius"]
    near_positions = [
        wrap_position(cue_position + lag) for lag in range(-near_radius, near_radius + 1)
        if lag and wrap_position(cue_position + lag) not in used_positions
    ]
    take(near_positions, "near")

    mid_radius = choice_options["mid_band_radius"]
    mid_positions_by_side = {
        side: [wrap_position(cue_position + side * distance) for distance in range(near_radius + 1, mid_radius + 1)]
        for side in (-1, 1)
    }
    mid_count = choice_options["mid_band_count"]
    count_by_side = {-1: mid_count // 2, 1: mid_count // 2}
    if mid_count % 2:
        count_by_side[rng.choice([-1, 1])] += 1
    for side in (-1, 1):
        take(rng.sample(mid_positions_by_side[side], count_by_side[side]), "mid")

    run_length = choice_options["camouflage_run_length"]
    if run_length:
        run_starts = [
            start for start in range(1, SEQUENCE_LENGTH + 1)
            if all(
                wrap_position(start + step) not in used_positions
                and abs(get_circular_lag(wrap_position(start + step), cue_position))
                >= choice_options["camouflage_min_distance"]
                for step in range(run_length)
            )
        ]
        start = rng.choice(run_starts)   # check_choice_options guarantees room for one run
        take([wrap_position(start + step) for step in range(run_length)], "camouflage")

    far_count = choice_options["number_of_choices"] - 1 - len(band_by_position)
    far_candidates = [
        position for position in range(1, SEQUENCE_LENGTH + 1)
        if position not in used_positions
        and abs(get_circular_lag(position, cue_position)) >= choice_options["far_min_distance"]
    ]
    if len(far_candidates) >= far_count:
        take(rng.sample(far_candidates, far_count), "far")
    else:
        take(far_candidates, "far")
        remaining_positions = [position for position in range(1, SEQUENCE_LENGTH + 1) if position not in used_positions]
        take(rng.sample(remaining_positions, far_count - len(far_candidates)), "far_fill")
    return band_by_position


def get_nearest_positions(cue_position, category_code, category_by_position, count_per_side):
    """The count_per_side nearest positions of one category before and after the cue (on the circle)."""
    nearest_positions = []
    for side in (-1, 1):
        same_category = [
            wrap_position(cue_position + side * distance) for distance in range(1, SEQUENCE_LENGTH // 2)
            if category_by_position[wrap_position(cue_position + side * distance)] == category_code
        ]
        nearest_positions += same_category[:count_per_side]
    return nearest_positions


def choose_category_band_positions(cue_position, target_position, sequence_rows, choice_options, rng):
    """Foil positions around the cue with the same number from every category: {position: band}.

    Each category gets number_of_choices / 5 images (target and foils), filled in this order:
        near:       every position within +-near_band_radius of the cue, so the
                    asked-vs-chosen diagonal is complete for offsets +-1..3
        nearest:    each category's nearest_per_category_side nearest images on
                    each side of the cue, if not already in the near band
                    (for the cue's own category these are its neighbouring-cycle
                    exemplars, about +-5: the 'right category, wrong cycle' test)
        camouflage: camouflage_cycles whole five-image cycles, all at least
                    camouflage_min_distance from the cue; a cycle holds one image
                    of every category, so the run keeps the categories balanced
        far:        each category topped up to its share, at least
                    far_min_distance from the cue ('far_fill' if too few)
    The target is always in the near band, and the bands are symmetric about
    the cue, so which images are present gives nothing away. The cue is never
    offered. check_category_band_options makes sure no category's near images
    exceed its share.
    """
    category_by_position = {row["position"]: row["category_code"] for row in sequence_rows}
    images_per_category = choice_options["number_of_choices"] // len(CATEGORY_NAMES)
    band_by_position = {target_position: "target"}

    def category_count(category_code):
        return sum(category_by_position[position] == category_code for position in band_by_position)

    # near band
    near_radius = choice_options["near_band_radius"]
    for lag in range(-near_radius, near_radius + 1):
        position = wrap_position(cue_position + lag)
        if lag and position not in band_by_position:
            band_by_position[position] = "near"

    # nearest images of every category on each side
    for category_code in CATEGORY_NAMES:
        for position in get_nearest_positions(cue_position, category_code, category_by_position,
                                              choice_options["nearest_per_category_side"]):
            if position not in band_by_position:
                band_by_position[position] = "nearest"

    # camouflage: whole far cycles
    eligible_cycles = [
        cycle for cycle in range(1, NUMBER_OF_CYCLES + 1)
        if all(
            position not in band_by_position
            and abs(get_circular_lag(position, cue_position)) >= choice_options["camouflage_min_distance"]
            for position in range((cycle - 1) * CYCLE_LENGTH + 1, cycle * CYCLE_LENGTH + 1)
        )
    ]
    for cycle in rng.sample(eligible_cycles, choice_options["camouflage_cycles"]):
        for position in range((cycle - 1) * CYCLE_LENGTH + 1, cycle * CYCLE_LENGTH + 1):
            band_by_position[position] = "camouflage"

    # far: top every category up to its share
    for category_code in CATEGORY_NAMES:
        images_needed = images_per_category - category_count(category_code)
        if images_needed < 0:
            raise ValueError(f"Category {category_code} has more than {images_per_category} images "
                             f"for cue {cue_position}: check_category_band_options should have caught this")
        unused_positions = [
            position for position in range(1, SEQUENCE_LENGTH + 1)
            if category_by_position[position] == category_code
            and position not in band_by_position and position != cue_position
        ]
        far_positions = [
            position for position in unused_positions
            if abs(get_circular_lag(position, cue_position)) >= choice_options["far_min_distance"]
        ]
        if len(far_positions) >= images_needed:
            for position in rng.sample(far_positions, images_needed):
                band_by_position[position] = "far"
        else:
            for position in far_positions:
                band_by_position[position] = "far"
            nearer_positions = [position for position in unused_positions if position not in far_positions]
            for position in rng.sample(nearer_positions, images_needed - len(far_positions)):
                band_by_position[position] = "far_fill"

    del band_by_position[target_position]   # add_choice_grid places the target itself
    return band_by_position


def get_balanced_target_slots(number_of_choices, number_of_trials, rng):
    """Target grid slot for each trial: every slot used equally often (to within one).

    Slots are dealt in shuffled rounds of all n slots, so with 50 choices and
    50 trials every slot is used exactly once, as before.
    """
    target_slots = []
    while len(target_slots) < number_of_trials:
        target_slots += rng.sample(range(1, number_of_choices + 1), number_of_choices)
    return target_slots[:number_of_trials]


def add_choice_grid(trial, sequence_rows, target_slot, choice_options, rng):
    """The answer grid for one trial (slot 1 = top left, row by row), with the
    target at target_slot and the foils in random order around it.

    Every slot gets choice_NN (image), choice_NN_band ('target', 'cue',
    'random', 'near', 'mid', 'nearest', 'camouflage', 'far' or 'far_fill') and
    choice_NN_lag (position on the circle relative to the cue, -25 ... +24),
    so error analyses can condition on what was on screen.
    """
    number_of_choices = choice_options["number_of_choices"]
    cue_position, target_position = trial["cue_position"], trial["target_position"]

    if number_of_choices == SEQUENCE_LENGTH:
        band_by_position = {
            position: "random" for position in range(1, SEQUENCE_LENGTH + 1) if position != target_position
        }
        band_by_position[cue_position] = "cue"
    elif choice_options["sampling"] == "random":
        candidates = [position for position in range(1, SEQUENCE_LENGTH + 1)
                      if position not in (cue_position, target_position)]
        band_by_position = {position: "random" for position in rng.sample(candidates, number_of_choices - 1)}
    elif choice_options["sampling"] == "bands":
        band_by_position = choose_band_positions(cue_position, target_position, choice_options, rng)
    else:
        band_by_position = choose_category_band_positions(
            cue_position, target_position, sequence_rows, choice_options, rng
        )

    foil_positions = rng.sample(sorted(band_by_position), len(band_by_position))
    grid_positions = foil_positions[:target_slot - 1] + [target_position] + foil_positions[target_slot - 1:]
    band_by_position[target_position] = "target"

    row_by_position = {row["position"]: row for row in sequence_rows}
    # near and nearest-per-category foils that reach across the 50 -> 1 join
    near_band_lags = [
        get_circular_lag(position, cue_position)
        for position, band in band_by_position.items() if band in ("near", "nearest")
    ]
    trial["choice_settings"] = get_choice_settings_label(choice_options)
    trial["number_of_choices"] = number_of_choices
    trial["target_grid_slot"] = target_slot
    trial["cue_grid_slot"] = grid_positions.index(cue_position) + 1 if cue_position in grid_positions else ""
    trial["near_band_crosses_sequence_end"] = int(any(
        not 1 <= cue_position + lag <= SEQUENCE_LENGTH for lag in near_band_lags
    ))
    for slot, position in enumerate(grid_positions, start=1):
        trial[f"choice_{slot:02d}"] = row_by_position[position]["image_file"]
    for slot, position in enumerate(grid_positions, start=1):
        trial[f"choice_{slot:02d}_band"] = band_by_position[position]
    for slot, position in enumerate(grid_positions, start=1):
        trial[f"choice_{slot:02d}_lag"] = get_circular_lag(position, cue_position)


def build_recall_session(participant_id, session, sequence_rows, rng, grid_rng, choice_options):
    """{run number: [trial, ...]} for one session.

    Schedules and trial orders come from rng; the answer grids come from
    grid_rng afterwards, so the grid settings cannot change the questions.
    """
    row_by_position = {row["position"]: row for row in sequence_rows}
    schedule = build_session_schedule(session, rng)

    runs = {}
    for run_number, cue_offset_pairs in schedule.items():
        trials = [
            build_recall_trial(participant_id, session, run_number, cue_position, offset, row_by_position)
            for cue_position, offset in cue_offset_pairs
        ]
        trials = order_trials(trials, rng)
        for trial_number, trial in enumerate(trials, start=1):
            trial["trial"] = trial_number
        runs[run_number] = trials

    for trials in runs.values():
        # every grid slot holds the target equally often within a run
        target_slots = get_balanced_target_slots(choice_options["number_of_choices"], len(trials), grid_rng)
        for trial, target_slot in zip(trials, target_slots):
            add_choice_grid(trial, sequence_rows, target_slot, choice_options, grid_rng)
    return runs


def check_recall_session(runs, session):
    """Raise if any of the boundary-agnostic constraints is broken."""
    all_trials = [trial for trials in runs.values() for trial in trials]
    problems = []
    for run_number, trials in runs.items():
        cues = sorted(trial["cue_position"] for trial in trials)
        if cues != list(range(1, SEQUENCE_LENGTH + 1)):
            problems.append(f"run {run_number} does not probe every cue once")
        offset_counts = Counter(trial["offset"] for trial in trials)
        if set(offset_counts) != set(OFFSETS) or not set(offset_counts.values()) <= {8, 9}:
            problems.append(f"run {run_number} offset counts {dict(offset_counts)}")
        if not order_is_acceptable(trials):
            problems.append(f"run {run_number} breaks the trial-order constraints")
        if run_number < FIRST_RUN_ALLOWING_WRAP[session] and any(trial["wraps_sequence_end"] for trial in trials):
            problems.append(f"run {run_number} asks a wrap-around question before the join has been seen")
        for trial in trials:
            number_of_choices = trial["number_of_choices"]
            grid = [trial[f"choice_{slot:02d}"] for slot in range(1, number_of_choices + 1)]
            if len(set(grid)) != number_of_choices or grid[trial["target_grid_slot"] - 1] != trial["target_image"]:
                problems.append(f"run {run_number} trial {trial['trial']} has a broken choice grid")
            if number_of_choices < SEQUENCE_LENGTH and trial["cue_image"] in grid:
                problems.append(f"run {run_number} trial {trial['trial']} offers the cue")
        slot_counts = Counter(trial["target_grid_slot"] for trial in trials)
        number_of_choices = trials[0]["number_of_choices"]
        if len(slot_counts) != min(number_of_choices, len(trials)) or max(slot_counts.values()) - min(slot_counts.values()) > 1:
            problems.append(f"run {run_number} does not spread the target evenly over the grid slots")

    pair_counts = Counter((trial["cue_position"], trial["offset"]) for trial in all_trials)
    if len(pair_counts) != SEQUENCE_LENGTH * len(OFFSETS) or set(pair_counts.values()) != {1}:
        problems.append("not every cue x offset pair occurs exactly once")
    if set(Counter(trial["offset"] for trial in all_trials).values()) != {SEQUENCE_LENGTH}:
        problems.append("offsets do not occur 50 times each")
    if set(Counter(trial["target_position"] for trial in all_trials).values()) != {len(OFFSETS)}:
        problems.append("positions are not each the target six times")
    if problems:
        raise ValueError("; ".join(problems))


# ------------------------------------------------------------------ main ---

def make_participant_conditions(participant_id, choice_options=None, assignments_file=settings.ASSIGNMENTS_FILE,
                                stimulus_list_file=settings.STIMULUS_LIST_FILE,
                                conditions_folder=settings.CONDITIONS_FOLDER):
    """Write the sequence file and all twelve recall-run files for one participant.

    choice_options: the answer-grid settings (DEFAULT_CHOICE_OPTIONS if None).
    """
    choice_options = choice_options or DEFAULT_CHOICE_OPTIONS
    participant_number = get_participant_number(participant_id)
    participant_id = get_participant_id(participant_number)
    assignments = load_assignments(assignments_file)
    if participant_number not in assignments:
        raise ValueError(f"{participant_id} is not in {assignments_file}")
    assignment = assignments[participant_number]
    check_assignment(assignment)
    problems = check_choice_options(choice_options, assignment["category_sequence_50"])
    if problems:
        raise ValueError("Grid settings do not work: " + " ".join(problems))

    base_orders = get_exemplar_base_orders(load_stimulus_list(stimulus_list_file))
    sequence_rows = build_sequence_rows(participant_number, assignment, base_orders, choice_options)
    write_rows(sequence_rows, get_sequence_file(participant_id, conditions_folder))

    sessions = {}
    for session in SESSIONS:
        rng = random.Random(get_design_seed(participant_number, session))
        grid_rng = random.Random(get_design_seed(participant_number, session) + GRID_SEED_OFFSET)
        runs = build_recall_session(participant_id, session, sequence_rows, rng, grid_rng, choice_options)
        check_recall_session(runs, session)
        for run_number, trials in runs.items():
            write_rows(trials, get_recall_file(participant_id, session, run_number, conditions_folder))
        sessions[session] = runs
    return sequence_rows, sessions


def print_summary(all_sequences, all_sessions):
    """Counterbalancing checks across the participants just generated."""
    exemplar_by_occurrence = Counter(
        (row["image_file"], row["category_occurrence"]) for rows in all_sequences for row in rows
    )
    print(f"exemplar x occurrence counts across {len(all_sequences)} participants: "
          f"{min(exemplar_by_occurrence.values())}-{max(exemplar_by_occurrence.values())}")

    all_trials = [
        trial for sessions in all_sessions for runs in sessions.values()
        for trials in runs.values() for trial in trials
    ]
    category_by_offset = Counter((trial["cue_category"], trial["offset"]) for trial in all_trials)
    print(f"cue category x offset counts: {min(category_by_offset.values())}-{max(category_by_offset.values())}")
    wrapping = sum(trial["wraps_sequence_end"] for trial in all_trials)
    crossing = sum(trial["crosses_five_item_boundary"] for trial in all_trials)
    print(f"trials across the 50 -> 1 join: {wrapping} of {len(all_trials)} ({wrapping / len(all_trials):.1%})")
    print(f"trials crossing a five-item cycle boundary: {crossing} of {len(all_trials)} ({crossing / len(all_trials):.1%})")

    wrap_trials_by_run = Counter(
        (trial["session"], trial["recall_run"]) for trial in all_trials if trial["wraps_sequence_end"]
    )
    for session in SESSIONS:
        counts = [wrap_trials_by_run[(session, run_number)] for run_number in range(1, NUMBER_OF_RUNS + 1)]
        print(f"wrap-around trials per run, {session} (runs 1-6, all participants): {counts}")

    repeated_run_pairs = []
    for sessions in all_sessions:
        evening = {(run, trial["cue_position"], trial["offset"]) for run, trials in sessions["evening"].items() for trial in trials}
        morning = {(run, trial["cue_position"], trial["offset"]) for run, trials in sessions["morning"].items() for trial in trials}
        repeated_run_pairs.append(len(evening & morning))
    print(f"cue x offset pairs in the same run number evening and morning: "
          f"mean {sum(repeated_run_pairs) / len(repeated_run_pairs):.1f} of 300 (50 expected by chance)")


def main(participant_ids):
    if not participant_ids:
        participant_ids = [get_participant_id(number) for number in sorted(load_assignments(settings.ASSIGNMENTS_FILE))]

    all_sequences, all_sessions = [], []
    for participant_id in participant_ids:
        sequence_rows, sessions = make_participant_conditions(participant_id)
        all_sequences.append(sequence_rows)
        all_sessions.append(sessions)
    print(f"wrote conditions for {len(participant_ids)} participant(s) to {settings.CONDITIONS_FOLDER}/")
    print(f"answer grid: {describe_choice_options(DEFAULT_CHOICE_OPTIONS)} "
          f"(other settings are chosen in cued_recall.py's start dialog)")
    print_summary(all_sequences, all_sessions)


if __name__ == "__main__":
    main(sys.argv[1:])
