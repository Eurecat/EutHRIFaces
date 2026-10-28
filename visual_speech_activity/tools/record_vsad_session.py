#!/usr/bin/env python3
"""
Guided recording of a ground-truth bag for visual speech activity (lip) evaluation.

The script tells the person on camera, on screen and with a beep, when to speak and when to stay
silent, and records the prompts as ground truth next to the stack topics in one rosbag:

    /gt/speech_label  std_msgs/String   SPEAK | SILENT | TRANSITION   (what the person was told)
    /gt/phase         std_msgs/String   e.g. "dist=3.0m"              (scenario block)

TRANSITION covers the first --reaction seconds after every prompt (the person is still reacting),
and the evaluator ignores it. The schedule repeats SPEAK/SILENT blocks at each distance, plus
"hard negatives" a lip detector must not call speech (smile, chew, yawn, turn head).

Run inside the eut_visual_speech_activity container with the camera and the face pipeline already up:

    python3 /workspace/src/visual_speech_activity/tools/record_vsad_session.py \\
        --out /workspace/log/vsad/<name> --distances 0.7 1.5 3 4.5

Add --rate-note "pc 5Hz" etc. to the notes. Metadata goes to <out>/session.json.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import rclpy
from std_msgs.msg import String

TOPICS = [
    "camera/image_raw/compressed",
    "humans/faces/detected",
    "humans/faces/recognized",
    "humans/faces/speaking",
    "audio_and_device_info",       # mic: lets you cross-check labels against audio afterwards
]
HARD_NEGATIVES = [
    ("SILENT", "mouth CLOSED, SMILE and relax (no talking)"),
    ("SILENT", "CHEW as if eating (no sound)"),
    ("SILENT", "turn your HEAD left and right, mouth closed"),
    ("SILENT", "YAWN once, then close the mouth"),
]


def schedule(distances, speak_s, silent_s, repeats, hard_negatives, move_s):
    """List of (phase, label, seconds, instruction). label MOVE is published as TRANSITION."""
    out = []
    for dist in distances:
        phase = f"dist={dist}m"
        out.append(("move", "MOVE", move_s, f"MOVE the camera/yourself to {dist} m, face the camera, mouth closed"))
        out.append((phase, "SILENT", 6.0, f"HOLD still at {dist} m, mouth closed"))
        for r in range(repeats):
            out.append((phase, "SPEAK", speak_s, "SPEAK NOW: talk normally (any text, count, read)"))
            out.append((phase, "SILENT", silent_s, "NO SPEECH: mouth closed, still"))
        if hard_negatives:
            for label, text in HARD_NEGATIVES:
                out.append((phase + ",hard", label, 5.0, text))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="bag directory to create")
    ap.add_argument("--distances", type=float, nargs="+", default=[0.7, 1.5, 3.0, 4.5])
    ap.add_argument("--speak", type=float, default=10.0, help="seconds per SPEAK block")
    ap.add_argument("--silent", type=float, default=8.0, help="seconds per SILENT block")
    ap.add_argument("--repeats", type=int, default=2, help="SPEAK/SILENT pairs per distance")
    ap.add_argument("--reaction", type=float, default=1.5, help="seconds after each prompt labelled TRANSITION")
    ap.add_argument("--move", type=float, default=30.0, help="seconds to reposition before each distance")
    ap.add_argument("--no-hard-negatives", action="store_true")
    ap.add_argument("--topics", nargs="+", default=TOPICS)
    ap.add_argument("--notes", default="", help="free text: lighting, camera, glasses, PC rate...")
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists: choose a new bag directory")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rclpy.init()
    node = rclpy.create_node("vsad_gt_prompter")
    label_pub = node.create_publisher(String, "gt/speech_label", 10)
    phase_pub = node.create_publisher(String, "gt/phase", 10)
    send = lambda pub, text: pub.publish(String(data=text))

    # preflight: the bag is useless without these
    rclpy.spin_once(node, timeout_sec=2.0)
    missing = [t for t in ("camera/image_raw/compressed", "humans/faces/detected", "audio_and_device_info")
               if t in args.topics and node.count_publishers(t) == 0]
    if missing:
        sys.exit(f"no publisher on {missing}: start camera, faces and audio first (and check ROS_DOMAIN_ID)")

    # gt topics must exist when recording starts, so the bag has them from the first message
    bag = subprocess.Popen(["ros2", "bag", "record", "-o", args.out, "--storage", "mcap",
                            "gt/speech_label", "gt/phase", *args.topics],
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    plan = schedule(args.distances, args.speak, args.silent, args.repeats, not args.no_hard_negatives, args.move)
    total = sum(p[2] for p in plan)
    print(f"Recording {total:.0f} s ({total / 60:.1f} min), {len(plan)} blocks. Starting in 5 s...")
    started = time.time()
    log = []
    try:
        time.sleep(5.0)
        for phase, label, seconds, text in plan:
            print(f"\a\n=== [{phase}] {label}: {text}  ({seconds:.0f} s)")
            send(phase_pub, phase)
            t0 = time.time()
            send(label_pub, "TRANSITION")
            time.sleep(min(args.reaction, seconds))
            shown = "TRANSITION" if label == "MOVE" else label
            send(label_pub, shown)
            warned = False
            while time.time() - t0 < seconds:
                if label == "MOVE" and not warned and seconds - (time.time() - t0) < 5.0:
                    print("\a   ...5 s left, get ready")
                    warned = True
                send(label_pub, shown)  # repeated at 2 Hz: a dropped message cannot hide a block
                time.sleep(0.5)
            log.append({"phase": phase, "label": label, "seconds": seconds, "text": text})
        send(label_pub, "TRANSITION")
        time.sleep(1.0)
    except KeyboardInterrupt:
        print("interrupted: the bag keeps what was recorded")
    finally:
        bag.send_signal(signal.SIGINT)
        bag.wait(timeout=60)
        with open(os.path.join(args.out, "session.json"), "w") as f:
            json.dump({"recorded": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
                       "reaction_s": args.reaction, "notes": args.notes, "topics": args.topics,
                       "blocks": log}, f, indent=2)
        node.destroy_node()
        rclpy.shutdown()
    print(f"\nDone: {args.out}")


if __name__ == "__main__":
    main()
