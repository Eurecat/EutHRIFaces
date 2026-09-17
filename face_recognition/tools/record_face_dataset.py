#!/usr/bin/env python3
"""
Record a face-embedding dataset from the live stack for offline identity evaluation.

For every FacialLandmarksArray on /humans/faces/detected it stores, per face:
  - face_id (tracker id), stamp, bbox (normalized), bbox confidence, 70 landmarks
  - embedding from the image whose stamp matches the detection (correct pairing)
  - embedding from the newest image at processing time (what the node did before stamp matching)
  - the embedding crop itself (JPEG), so the session can be re-embedded with any model later
  - video frame index (when --video is given), found by thumbnail matching, so
    detections from different loops of a looping test video can be aligned.

Run inside the eut_face_recognition container:
  source /workspace/install/setup.bash
  python3 /workspace/src/face_recognition/tools/record_face_dataset.py \
      --duration 70 --video /tmp/video_3.mp4 \
      --out /workspace/src/face_recognition/database/eval/video_3.npz
"""

import argparse
import os
import sys
import time
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from hri_msgs.msg import FacialLandmarksArray

from face_recognition.face_embedding_extractor import create_face_embedding_extractor
from face_recognition.face_recognition_node import FaceRecognitionNode

THUMB_SIZE = (64, 36)


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def thumb(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


class Recorder(Node):
    def __init__(self, args):
        super().__init__('face_dataset_recorder')
        self.args = args
        self.images = deque(maxlen=150)  # (stamp_ns, CompressedImage)
        self.pending = deque()
        self.rows = []
        self.video_thumbs = None
        if args.video:
            cap = cv2.VideoCapture(args.video)
            thumbs = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                thumbs.append(thumb(frame))
            self.video_thumbs = np.stack(thumbs)
            self.get_logger().info(f'Loaded {len(thumbs)} video frames for frame matching')

        weights_dir = '/workspace/src/face_recognition/weights'
        self.extractor = create_face_embedding_extractor(
            model_name='vggface2', device=args.device, weights_path=weights_dir,
            face_embedding_weights_path=os.path.join(weights_dir, '20180402-114759-vggface2.pt'))
        # Borrow the node's crop/alignment code with a minimal stand-in for `self`
        self.crop_ctx = SimpleNamespace(
            min_h_size=args.min_h_size, enable_face_alignment=args.align, enable_debug_output=False,
            crop_mode=args.crop_mode, aligned_crop_size=args.aligned_crop_size,
            last_image=None, get_logger=self.get_logger,
            _align_face_crop=lambda *a: FaceRecognitionNode._align_face_crop(self.crop_ctx, *a),
            _extract_face_crop_from_landmarks=lambda m: FaceRecognitionNode._extract_face_crop_from_landmarks(self.crop_ctx, m))

        self.create_subscription(CompressedImage, args.image_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(FacialLandmarksArray, args.faces_topic, self._on_faces, 50)
        self.create_timer(0.01, self._process)
        self.t_end = time.time() + args.duration

    def _on_image(self, msg):
        self.images.append((stamp_ns(msg.header.stamp), msg))

    def _on_faces(self, msg):
        latest = self.images[-1] if self.images else None
        self.pending.append((msg, latest))

    def _decode(self, msg):
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)

    def _crops(self, image, faces):
        self.crop_ctx.last_image = image
        return [FaceRecognitionNode._extract_face_crop(self.crop_ctx, f) for f in faces]

    def _embed(self, crops):
        valid = [c for c in crops if c is not None]
        embs = iter(self.extractor.extract_embeddings_batch(valid)) if valid else iter(())
        return [next(embs) if c is not None else None for c in crops]

    def _process(self):
        while self.pending:
            msg, latest = self.pending[0]
            t = stamp_ns(msg.header.stamp)
            matched = next((m for s, m in self.images if s == t), None)
            if matched is None and (not self.images or self.images[-1][0] < t):
                return  # image not arrived yet
            self.pending.popleft()
            if matched is None or not msg.ids:
                continue
            image = self._decode(matched)
            frame_idx = -1
            if self.video_thumbs is not None:
                frame_idx = int(np.argmin(np.mean((self.video_thumbs - thumb(image)) ** 2, axis=1)))
            crops = self._crops(image, msg.ids)
            emb_matched = self._embed(crops)
            if latest is not None and latest[0] != t:
                emb_latest = self._embed(self._crops(self._decode(latest[1]), msg.ids))
                latest_lag_ms = (latest[0] - t) / 1e6
            else:
                emb_latest, latest_lag_ms = emb_matched, 0.0
            for face, crop, e_m, e_l in zip(msg.ids, crops, emb_matched, emb_latest):
                lms = np.array([[p.x, p.y, p.c] for p in face.landmarks], dtype=np.float32)
                if lms.shape[0] < 70:
                    lms = np.vstack([lms, np.zeros((70 - lms.shape[0], 3), np.float32)])
                b = face.bbox_xyxy
                self.rows.append(dict(
                    stamp_ns=t, frame_idx=frame_idx, face_id=face.face_id,
                    bbox=np.array([b.xmin, b.ymin, b.xmax, b.ymax], np.float32),
                    bbox_conf=float(face.bbox_confidence), width=int(face.width), height=int(face.height),
                    landmarks=lms[:70], latest_lag_ms=latest_lag_ms,
                    emb_matched=e_m, emb_latest=e_l,
                    crop_jpg=cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                    if crop is not None else b''))
        if time.time() > self.t_end:
            raise SystemExit

    def save(self):
        dim = next((r['emb_matched'].shape[0] for r in self.rows if r['emb_matched'] is not None), 512)

        def stack_emb(key):
            return np.stack([r[key] if r[key] is not None else np.full(dim, np.nan, np.float32)
                             for r in self.rows]).astype(np.float32)

        os.makedirs(os.path.dirname(self.args.out), exist_ok=True)
        np.savez_compressed(
            self.args.out,
            stamp_ns=np.array([r['stamp_ns'] for r in self.rows], np.int64),
            frame_idx=np.array([r['frame_idx'] for r in self.rows], np.int32),
            face_id=np.array([r['face_id'] for r in self.rows]),
            bbox=np.stack([r['bbox'] for r in self.rows]),
            bbox_conf=np.array([r['bbox_conf'] for r in self.rows], np.float32),
            width=np.array([r['width'] for r in self.rows], np.int32),
            height=np.array([r['height'] for r in self.rows], np.int32),
            landmarks=np.stack([r['landmarks'] for r in self.rows]),
            latest_lag_ms=np.array([r['latest_lag_ms'] for r in self.rows], np.float32),
            emb_matched=stack_emb('emb_matched'),
            emb_latest=stack_emb('emb_latest'),
            crop_jpg=np.array([r['crop_jpg'] for r in self.rows], dtype=object),
            crop_mode=np.array(self.args.crop_mode),
        )
        self.get_logger().info(f'Saved {len(self.rows)} face rows to {self.args.out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=70.0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--video', default='')
    parser.add_argument('--image-topic', default='/camera/image_raw/compressed')
    parser.add_argument('--faces-topic', default='/humans/faces/detected')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--min-h-size', type=int, default=30)
    parser.add_argument('--no-align', dest='align', action='store_false')
    parser.add_argument('--crop-mode', default='aligned', choices=['aligned', 'bbox'])
    parser.add_argument('--aligned-crop-size', type=int, default=160)
    args = parser.parse_args()

    rclpy.init()
    node = Recorder(args)
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    node.save()
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
