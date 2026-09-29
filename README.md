# VLM MicroDuck

Independent vision and task code for the MicroDuck project. MuJoCo scenes and robot simulation
remain in the `microduck_rlxvla` checkout; this directory contains application code that can be
developed separately from the robot repositories.

## Current milestone: detect and turn toward a red box

`vlm_microduck.detector.find_red_box(frame)` accepts an upright RGB image as a NumPy array and
returns the largest sufficiently large red region, with its bounding box and `left`, `center`, or
`right` position. It returns `None` when it does not find a target. The left/center/right regions
are split at 40% and 60% of the image width.

`vlm_microduck.camera.SimulatorCamera` reads the local MuJoCo frame port, converts the simulator's
UYVY frames to upright RGB, and passes them to the detector. The current camera adapter is for the
local simulator. The optional alignment controller turns the simulated duck until the target is
inside the center region; the preview-only mode remains the default.

## Install for development

```bash
cd /home/mlo/vlm-microduck
uv venv
uv pip install -e .
```

## Run against the local simulator

Start `scripts/duck-sim` with `DUCK_SIM_CAMERAS=a` so `duck-body` opens its camera frame port
(default `127.0.0.1:7901`). With the simulator still running, start the detector in another
terminal:

```bash
uv run vlm-microduck
```

It opens a browser preview at `http://127.0.0.1:8081/` and continuously updates the terminal with
`左侧`, `中间`, `右侧`, or `未找到`. The preview shows the live camera image, detection box, and
left/center/right guides; its status line reports the current detection or a camera connection
problem. Use `Ctrl+C` in the terminal to stop it. Pass `--no-browser` to print the preview URL
without opening a browser automatically. If the simulator uses another host or frame port, pass
`--host`/`--port` or set `DUCK_CAMERA_HOST`/`DUCK_CAMERA_PORT`. To change the preview address, pass
`--preview-host`/`--preview-port` or set `DUCK_PREVIEW_HOST`/`DUCK_PREVIEW_PORT`.

## Automatically turn toward the target

Once the simulator is standing and the red target is detected reliably, stop the existing
detector with Ctrl+C and run:

```bash
cd /home/mlo/vlm-microduck
uv run vlm-microduck --align
```

The controller sends rotation-only `robot.move` intents to
`~/.cache/duck-sim/duck-a.sock` at up to 10 Hz. Its forward and sideways speeds are always zero.
A left-side target requests a positive yaw rate; a right-side target requests a negative one.
A direction needs three consecutive frames before turning starts. Centering also needs three
frames before stopping, so one boundary flash during a footstep does not abort the gait.
A missing target has at most 0.1 seconds of grace, enforced independently by the watchdog.

The default requested turn speed is 1.5 rad/s, the limit also used by the RL repository's CPU
rehearsal script. Accepted values are positive and at most 1.5 rad/s. This is an intent, **not the
measured speed**: receiving the request does not guarantee that the installed gait executes it.
The preview displays the measured IMU yaw rate averaged over roughly one second. A turn with no
sustained response after two seconds is stopped and paused, rather than continuing to claim that
it is turning. If that happens, first check manual movement with alignment paused and report the
observed behavior; do not automatically replace the gait or change the simulator's motor setup.

Use the preview's **切换手动控制** button before driving in the robot console at
`http://127.0.0.1:8080`. Detection and video continue, but the automatic controller no longer
sends movement or idle-zero commands. Click **恢复自动对准** to return control to vision; it sends
one stop and waits for new detection confirmation.

A different movement observed on `robot.subscribe` also automatically pauses alignment, without
writing a stop over that manual command. This is best-effort detection: the API does not identify
the movement's owner, so an identical yaw command or a command overwritten before a state tick
cannot be distinguished. The explicit mode button is the reliable hand-off.

A watchdog stops owned automatic motion after 0.3 seconds without a new frame, and a lost/stale
robot-state stream stops it too. Ctrl+C, SIGTERM, and camera connection errors stop **our own**
rotation; they do not cancel a manual driver after hand-off. robotd also expires stale intents.
Closing the browser alone does not stop the controller.

This application uses the existing simulator and daemon IPC without requiring changes to either
underlying repository. The experimental motor, gait, and observation-noise changes were reverted.
Any future underlying change must be explained and approved before implementation.

To start the existing red-box scene with the simulator's original settings:

```bash
cd /home/mlo/microduckxvla
DUCK_SIM_RL=/home/mlo/Pollen/microduck_rlxvla \
DUCK_SIM_SCENE=/home/mlo/Pollen/microduck_rlxvla/src/mjlab_microduck/robot/microduck/scene_box.xml \
DUCK_SIM_KEYFRAME=SIT \
DUCK_SIM_CAMERAS=a \
scripts/duck-sim
```

Then start `uv run vlm-microduck --align` from this application's directory. The existing
red-box scene is preserved; no scene, training, or hardware configuration changes are required
by this application. If the target is outside the camera view, switch to manual control, bring
it into view, stop, then resume alignment. Automatic target search and forward approach are not
implemented yet. A gait-stall pause also needs an explicit manual check and resume; it does not
automatically retry motion. Earlier physical-motion measurements made with the experimental
simulator settings do not establish performance with the restored settings.

Use `--robot-socket /path/to/duck-a.sock` or `DUCK_ROBOT_SOCKET` to choose a socket. The default
also follows `DUCK_SIM_STATE` and `DUCK_SIM_DUCK`. For another duck, set its matching camera
`--port` as well. Start the simulator first; the controller checks robotd health and the loaded
walking policy without changing robot enable or posture.

Run the controller checks with `uv run python -m unittest discover -s tests -v`.

Example use from Python:

```python
from vlm_microduck import annotate_red_box, find_red_box

detection = find_red_box(rgb_frame)
print(detection.position if detection else "not found")
preview = annotate_red_box(rgb_frame, detection)
```
