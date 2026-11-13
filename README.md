# RIPPLAY learning + cued recall (boundary-agnostic)

PsychoPy Coder implementation of the learning and cued-offset recall parts of
*RIPPLAY study design, cued-recall v4*. One launch runs a whole session.
Offsets wrap around the ends of the 50-image sequence, so every starting
position gets three forward and three backward questions per session.

## Files

| file | what it does |
|---|---|
| `cued_recall.py` | run this: start dialog, session flow, data files |
| `make_conditions.py` | design -> conditions CSVs (pure Python, no PsychoPy needed) |
| `localizer.py` | image localizer: every image in `stimuli/` 10 times, 1 s each, random order; called by `cued_recall.py` or run on its own |
| `sequence_reconstruction.py` | full 50-item sequence reconstruction, called by `cued_recall.py`; its own layout, switches and EEG codes |
| `settings.py` | every timing, key, size and switch |
| `triggers.py` | EEG event codes and the events log |
| `design/ripplay_25_participant_assignments_v3.csv` | decoy, core order and 50-position category sequence per participant |
| `conditions/` | generated: `P01_sequence.csv`, `P01_evening_run1.csv` ... `P01_morning_run6.csv` |
| `stimuli/` | your images and `stimulus_list.csv` (not included in the zip) |

## Session flow

**Evening (pre-sleep)**: six cycles of

1. study: the 50 images in the participant's fixed order, 3 s each, jittered
   1-2 s fixation between images
2. shades of grey: 2 minutes, count the times the cross gets lighter, typed
   count at the end
3. recall run: 50 trials

then the full sequence reconstruction. With `localizer` ticked, the image
localizer runs first, before study block 1 (not when resuming at a later cycle).

**Morning (post-sleep)**: full sequence reconstruction, then six recall runs;
no re-study, no shades of grey. With `localizer` ticked, the image localizer
runs last.

## Image localizer (localizer.py)

Every image file in `stimuli/` (50 now, or however many are there) is shown
`LOCALIZER_REPETITIONS` (10) times for `LOCALIZER_IMAGE_DURATION` (1 s), each
followed by a jittered 1-1.5 s fixation. Passive viewing, no responses. The
order is random, shuffled in passes (every image once per pass, a fresh order
each pass), so each image's presentations are spread across the block; the
same image never appears twice in a row. A short break every 100
presentations. About 19 minutes plus breaks.

EEG codes: 21-25 image onset (last digit = category, as elsewhere), 26
fixation. Data rows have `phase = localizer`, `localizer_pass`, `image_id`,
`category`, `image_repetition`, `image_onset` and `fixation_duration`.

`python localizer.py` runs it on its own, with its own start dialog and data
files (`data/<participant>_<session>_localizer_<date>`).

The reconstruction runs when the `reconstruction` box in the start dialog is
ticked (default). Untick it when resuming a morning session in which it was
already done, or to run a session without it; the check screen says whether
it will run.

## Sequence reconstruction (sequence_reconstruction.py)

All 50 studied images in one random grid arrangement, fixed for the whole
reconstruction. The participant clicks them in the order they were shown,
starting with the first. Picked images dim and cannot be picked again; the
last pick is shown at the top; UNDO takes back the last pick; DONE appears
once all 50 are placed. No time limit, no feedback.

Scores (summary row): exact positions, mean absolute displacement, Spearman
rho, Kendall tau, correctly ordered adjacent pairs (of 49), best circular
shift and exact matches after it (catches a correct order started at the
wrong place), core-order transitions with decoys removed (of 39), decoys at
the exact position and decoys in the right within-cycle slot.

Learning is incidental: nothing before the first recall run mentions order.
Rough durations with real response times: about 15 min per evening cycle
(about 90 min for the evening), about 9 min per morning run.

## Recall trial

| phase | duration |
|---|---|
| fixation | 1.0-1.5 s jittered |
| cue image, centred, with the signed offset on a translucent disc at its centre (one event, one marker) | 0.5 s |
| blank screen until R (remembered) / F (forgotten) | up to `MEMORY_RESPONSE_TIMEOUT` (5 s; the retrieval / awake-replay window) |
| answer grid (50 images by default, see *Answer grid*), mouse click | from the next frame after R/F (or its timeout), up to `CHOICE_TIMEOUT` (10 s), on R and F trials |

The grid is positioned while the participant decides, so it appears one to
two frames after the key press; `grid_delay_after_keypress` logs the exact
delay on every trial.

No feedback. Mouse starts at the screen centre every trial; its position is
logged every frame to `<data file>_mouse.csv`.

## Design (make_conditions.py)

- Category sequence: taken from the assignments file and checked on load
  (each cycle = core order with the decoy at the listed slot, each slot twice).
- Exemplars: the k-th occurrence of a category gets
  `base_order[(k + participant - 1) mod 10]`; across 25 participants every
  exemplar occupies every cycle 2-3 times.
- Image IDs: every image gets an `image_id` in the localiser file's format
  (`stimuli/o (07).jpg` -> `O07`), carried as `cue_image_id`,
  `target_image_id` and `chosen_image_id`, so recall and localiser data join
  on the same key.
- Offsets (per session): every run probes all 50 cues once with 8-9 trials per
  offset; across six runs every cue x offset pair occurs once, every offset 50
  times and every position is the target six times. Evening and morning
  schedules are drawn independently.
- Wrap-around: the 12 cue x offset pairs that cross the 50 -> 1 join are only
  asked once the join has been seen (`FIRST_RUN_ALLOWING_WRAP`): never in
  evening run 1 (before study block 2), freely in the morning.
- Trial order within a run: no offset three times in a row, no cue that was
  the previous target, no target twice in a row.
- Choice grid: freshly shuffled for every trial and stored in the conditions
  file (`choice_01` ... `choice_NN`, slot 1 = top left, row by row), with each
  slot's band (`choice_NN_band`) and position relative to the cue on the circle
  (`choice_NN_lag`, -25 ... +24). Within each run the target lands in every
  slot equally often (exactly once each with 50 choices), so target location
  is balanced across centre/edge and left/right. Size and sampling are set in
  the start dialog; see *Answer grid*.
- Grids are drawn from their own random generator after all trial orders are
  fixed, so changing the grid settings never changes the questions or their
  order. The sequence file records the grid settings (`choice_settings`); the
  task regenerates a participant's files when they differ and says so on the
  check screen.
- Flags per trial: `wraps_sequence_end`, `crosses_five_item_boundary`,
  cue/target category and decoy status.

`python make_conditions.py` regenerates all 25 participants and prints the
counterbalancing checks. The task also generates a participant's files on
first use, and regenerates them automatically if they were made by an older
version of `make_conditions.py` (`DESIGN_VERSION`, stored in every sequence
file), so outdated grids can never run by mistake.

## Answer grid

Two dropdowns in the start dialog:

- `answer_choices`: 50 (every studied image, cue included: the original grid)
  down to 10. Below 50 the cue is never offered.
- `choice_sampling`:
  - `random`: the target plus foils drawn uniformly from the other images
  - `bands`: foils built around the cue's position on the circle; a second
    dialog asks for the band sizes (defaults in
    `make_conditions.DEFAULT_CHOICE_OPTIONS`):
    - near: every image within ±`near_band_radius` (3, minimum 3, so the
      target is always inside the band and its presence gives nothing away)
    - mid: `mid_band_count` (3) images at distances 4-`mid_band_radius` (6),
      split across both sides
    - camouflage: one run of `camouflage_run_length` (6) consecutive images at
      least `camouflage_min_distance` (10) from the cue, so a cluster of
      sequence neighbours on screen doesn't single out the cue's (0 = none)
    - far: the rest, at least `far_min_distance` (7) from the cue

  - `category_bands`: the same complete near band plus an equal number of
    images from every category (n / 5 each; n must be a multiple of 5). Its
    own band dialog:
    - near: every image within ±`near_band_radius` (3), so the asked-vs-chosen
      diagonal is complete for ±1-3 on every trial
    - nearest: every category's `nearest_per_category_side` (1) nearest images
      before and after the cue; for the cue's own category these are its
      neighbouring-cycle exemplars (about ±5)
    - camouflage: `camouflage_cycles` (1) whole five-image cycles at least
      `camouflage_min_distance` (10) away; a cycle holds one image of each
      category, so it keeps the categories balanced
    - far: each category topped up to its share, at least `far_min_distance` (7) away

    With 20 choices this gives 4 per category on every trial; lags ±4, ±5 and
    ±6 are on screen about 94%, 71% and 30% of the time (set by each
    participant's sequence, so divide by availability). Fewer choices can
    fail: a category can already have 3 near images, so 15 choices (3 per
    category) is refused for most participants.

Settings that don't fit together (e.g. bands larger than the grid) are
explained and asked for again, checked against this participant's sequence. On the circle the near band is always complete;
for cues at positions 1-3 and 48-50 it reaches across the 50 -> 1 join
(`near_band_crosses_sequence_end`), including in evening run 1, before the
join has been seen. The grid is always a full rectangle in the shape that best
fits the screen (50 -> 10 x 5, 45 -> 9 x 5, 20 -> 5 x 4, 10 -> 5 x 2) and never grows images past
`CHOICE_MAX_IMAGE_SIZE`; with fewer than 50 choices the instructions say "a
set of pictures" instead of "all the pictures".

## Running

```
python cued_recall.py
```

- Choose participant and session; leave `start_run` at 1 and `reconstruction` ticked.
- `start_run` only resumes after a crash: the session restarts at the
  beginning of that cycle (in the evening, its study block, so resuming
  mid-recall gives that cycle's study block a second time; data files are
  tagged `_from<run>`).
- Tick `quick_test` for a short dry run.
- Escape quits and saves. F10 (`SKIP_KEY`; on a Mac hold fn, or turn on
  standard function keys) skips the rest of the current stage (instruction
  screen, study block, shades of grey, recall run or reconstruction) and moves
  on; the skip is logged as a `stage_skipped` row and EEG marker (code 104).
- `OFFSET_ON_CUE = False` in `settings.py` restores the older
  number -> fixation -> image sequence.

## Data

`data/<participant>_<session>_<date>`:

- `.csv`: one row per event of interest, labelled by `phase`, `session`, `cycle`
  and `trial`:
  - `study`: position, category, decoy flag, image, onset, fixation duration
  - `shades_of_grey`: one row per cross change, plus a `count` row with true
    and reported counts
  - `recall`: every conditions column plus `memory_report`, `memory_rt` (from
    onset of the blank after the cue), `chosen_slot`, `chosen_image`,
    `chosen_image_id`, `chosen_position`, `chosen_offset_from_cue` (circular,
    -25 ... +24), `chosen_band`, `exact_correct`, `category_correct`, `chose_cue`,
    `choice_rt`, flip-time onsets and the jittered durations
  - `reconstruction`: one row per pick or undo (reported rank, image, true
    position, grid slot and pixel coordinates, time since the screen appeared
    and since the previous action), plus a `summary` row with the scores, the
    reported order, the grid arrangement, duration and undo count
  - grid coordinates of the target, the cue and the chosen image on every
    trial: `target_grid_x_pixels`, `target_grid_y_pixels`, and the same for
    `cue_grid_` and `chosen_grid_`, each also in height units (`_x_height`,
    `_y_height`)
- `_grid_layout.csv`: one row per grid slot: row, column, centre in height
  units (origin at screen centre, y up) and in pixels (origin at the top-left
  corner, y down, as eye trackers report gaze), image size and window size.
  Written once per session from the actual window, so it matches the screen
  that was used.
- `_events.csv`: every EEG marker with its flip time and code.
- `_mouse.csv`: mouse trajectory on the choice screen, in height units. To
  convert to pixels: `x_pixels = window_width / 2 + x * window_height`,
  `y_pixels = window_height / 2 - y * window_height`.

Pixels are the window's framebuffer pixels. On a high-resolution (Retina)
Mac display these are twice the nominal screen resolution: check that the
eye tracker is calibrated in the same pixel space.

## Not yet implemented

Eye tracking (no tracker configured), baseline shades of grey, localiser and
PVT (which should run immediately before each reconstruction).
