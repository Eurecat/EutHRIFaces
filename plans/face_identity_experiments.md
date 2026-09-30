# Face identity experiments

Running log of what was tried to make `face_recognition` produce one stable identity per person, with measured results. Newest entries at the bottom. Keep adding entries; do not rewrite history.

All offline numbers come from the tools in `face_recognition/tools/` and the data in `face_recognition/database/eval/` (gitignored):

| File | Content |
|---|---|
| `video_3.npz` | 2289 face detections recorded live from `EutPerceptionUtils/eut_utils/samples/video_3.mp4` (5 people, 13.5 s loop, ~5 loops, face detection at ~6.4 Hz). Per detection: stamp, video frame index, tracker id, bbox, 70 landmarks, FaceNet embedding from the correct frame (`emb_matched`) and from the newest frame (`emb_latest`, what the old node did). |
| `video_3_gt.npy` | Person label per detection (0–4, -1 = false positive), built by `build_video_ground_truth.py` from position along the video timeline and checked by eye (`video_3_gt_sheet.jpg`). P0 man at table (frontal), P1 woman at table, P2 glasses man, P3 glasses woman (mostly profile), P4 bald man (always profile), -1 = his hand. |
| `video_3_ab.npz` | `video_3.npz` plus embeddings of other models/crops (see entry 3). |
| `live_2026-09-17_identities.npz` | Stored gallery (20 embeddings + mean, FaceNet tight crops) of identities U6–U13 learned live with the RealSense (see entry 2). |
| `embedding_ab_report.json` | Raw numbers of entry 3. |

Metrics used:
- **own p10**: 10th percentile similarity of a face to its own person's centroid.
- **other p99**: 99th percentile of the best similarity to any *other* person's centroid.
- **margin p10**: 10th percentile of (own − best other). Higher = safer matching.
- **replay**: `evaluate_identity_manager.py` feeds the recorded stream to `FaceIdentityManager` and reports identities created/final, purity, identities per person, coverage and false positives labeled.

---

## Entry 1 — 2026-09-17 — Old IdentityManager on video_3

**Symptom:** live, U1…U390 created for 5 people; MongoDB empty after `compose down`.

**Measured** (replay of the old manager with production params):

| | created | final | ids per person | purity | hand FPs labeled |
|---|---|---|---|---|---|
| old manager, `emb_latest` | 34 | 6 | 6–21 | 0.92 | 49/49 |
| old manager, `emb_matched` | 35 | 7 | 1–15 | 0.97 | 49/49 |

**Root causes found:**
1. Identity lifecycle: one embedding below 0.40 created an identity (same-person single pairs score as low as 0.25); identities with < 100 embeddings were deleted after 3 s unseen, so U numbers churned; mean embeddings were never normalized, so merges were under-scored; track stickiness had no score floor.
2. Crops taken from the newest image, not the detection's frame: 98% of crops 100 ms late (p90 167 ms); 11% of embeddings cos < 0.7 vs the correct one.
3. No quality gate: yaw proxy (inter-ocular / face width) 0.06–0.14 gave 73–96% correct top-1 with negative margins; hand false positives have detection confidence ≤ 0.25 (real faces p5 0.53).
4. MongoDB written only in `destroy_node`; containers are killed (exit 137), so nothing persisted.

**Change:** new `FaceIdentityManager` ported from the diarization identity layer (seeded identities, young/confirmed thresholds + margin, exclusive assignment, merge/absorb, quality gate, throttled continuous Mongo sync) and stamp-matched crops.

**Result:** replay 5 created / 5 final / purity 1.00 / 1 id per person / 0 of 49 FPs; live on video_3 exactly U1–U5 and the same ids after SIGKILL + restart. Mongo ≈ 0.6 writes/s.

## Entry 2 — 2026-09-17 — Live RealSense test (user, other people, photos of his wife)

**Symptom reported:** user stable as U7; chin-only view became U6; photos of the wife became U8 and U9 (never joined); later the user became U13 and never went back to U7.

**Measured** (stored galleries, FaceNet tight crops):

| pair | who | identity-pair score (½ mean·mean + ½ top-5 cross) | single samples → other identity (median) |
|---|---|---|---|
| U7 / U13 | user / user | 0.54 | U13→U7 0.36 |
| U8 / U9 | wife / wife | 0.63 | U9→U8 0.55 |
| U8 / U10 | wife / wife | 0.49 | U10→U8 0.47 |
| U6 / U7 | chin / user | 0.36 | 0.26 |
| worst pair of *different* people in video_3 | | 0.49 | sample p99 0.61 |

**Root causes:**
1. The same person under different conditions (0.35–0.63) overlaps with different people in the same scene (up to 0.61) for FaceNet on tight crops. No threshold can separate them.
2. `merge_threshold` 0.70 is never reached by same-person fragments, so a fragment is permanent.
3. Once a newer fragment (U13, tight cluster of the current look) exists, it wins every match, learns every frame, and the older identity (U7) starves: "forgot me".
4. Each identity kept only its last 100 samples (a few seconds at 30 Hz): its memory drifts to the latest look.
5. Chin-only faces pass the quality gate because YOLO still predicts eye points.

## Entry 3 — 2026-09-17 — Embedding model / crop A/B (video_3, 5-point aligned 112×112 crops)

Preprocessing per model chosen as the best of {RGB, BGR} × {raw 0–255, [-1, 1]} on a 1/4 subset (margin p10), then run on all 2289 detections. Timing: one face per call, `CUDAExecutionProvider`, Jetson Thor.

| Model | Weights license | Dim | Preproc | ms/face | own p10 | other p99 | margin p10 | top-1 | max centroid-pair |
|---|---|---|---|---|---|---|---|---|---|
| FaceNet vggface2, tight bbox crop (current) | MIT (facenet-pytorch / davidsandberg), trained on VGGFace2 | 512 | [-1,1] RGB | – | 0.60 | 0.54 | 0.27 | 0.968 | 0.30 |
| FaceNet vggface2, 5-point aligned 160×160 | same | 512 | same | – | 0.68 | 0.47 | **0.37** | 0.986 | 0.33 |
| OpenCV Zoo SFace 2021dec | **Apache-2.0** | 128 | raw RGB | 3.3 | 0.68 | 0.58 | 0.28 | 0.999 | 0.50 |
| ONNX Model Zoo arcfaceresnet100-8 | Apache-2.0 (ONNX zoo) | 512 | raw RGB | 21.3 | 0.50 | 0.61 | 0.10 | 0.956 | 0.59 |
| yakhyo sphere36 (MS1MV2) | MIT | 512 | [-1,1] RGB | 8.2 | 0.57 | 0.44 | 0.27 | 0.990 | 0.42 |
| yakhyo MobileNetV3-L (MS1MV2) | MIT | 512 | [-1,1] RGB | 4.1 | 0.60 | 0.68 | 0.10 | 0.966 | 0.62 |
| insightface buffalo_l w600k_r50 | **non-commercial research only** — rejected | 512 | [-1,1] RGB | 12.8 | 0.58 | 0.23 | 0.46 | 0.999 | 0.18 |

Gallery score (½ centroid + ½ top-3 gallery samples), different people vs own person:

| Model | own p5 | other p99 | worst person-pair | hand FP p90 |
|---|---|---|---|---|
| FaceNet tight | 0.68 | 0.58 | 0.49 | 0.60 |
| FaceNet aligned | 0.73 | 0.49 | 0.43 | 0.60 |
| SFace | 0.73 | 0.61 | 0.59 | 0.65 |
| sphere36 | 0.61 | 0.45 | 0.47 | 0.50 |

**Conclusions:**
- InsightFace models would solve most of it (other p99 0.23) but their weights are non-commercial; the repository is public and used commercially, so they are rejected.
- No commercially licensed candidate beats **FaceNet with 5-point alignment** on this video. Alignment alone raises margin p10 from 0.27 to 0.37 and lowers the worst person-pair score from 0.49 to 0.43.
- Caveat: video_3 measures "different people in one scene". It does not measure "same person under different conditions" (the live failure of entry 2). That needs a live recording with image crops (not only embeddings) so every model can be re-embedded.
- Licensing note: almost all public face models are trained on datasets with research-only terms (VGGFace2, MS1M, Glint360K…), even when the weights carry MIT/Apache. `dlib_face_recognition_resnet_model_v1` (released to the public domain by its author) is not evaluated yet because dlib is not installed in the image. Get a legal opinion before shipping any of them commercially.

**Next (entry 4):** keep FaceNet, switch to 5-point aligned crops, and fix the identity memory and merge rules that entry 2 showed are fragile under any model:
- diverse gallery per identity (skip near-duplicate frames, evict the most redundant sample instead of the oldest);
- gallery-aware scoring (centroid + closest samples);
- before creating an identity, match the averaged seed against existing identities;
- merge at a calibrated score, vetoed when two identities were ever seen in the same frame on different faces;
- longer tentative period (3 s) so strays are absorbed before being confirmed;
- the recorder also saves aligned crops so live sessions can be replayed with any model.

## Entry 4 — 2026-09-17 — Aligned crops + diverse gallery + seed re-match + co-occurrence veto

**Changes** (`face_alignment.py`, `identity_manager.py`, `face_recognition_node.py`):
- `crop_mode: aligned`: 5-point similarity warp to the standard template at 160×160 (FaceNet input). The box crop (`bbox`) remains available.
- Diverse gallery: a sample with cosine ≥ `redundancy_threshold` (0.92) to a stored one is counted but not stored. When the gallery is full (100), the most redundant sample is evicted instead of the oldest.
- Score = ½ cosine to the gallery mean + ½ mean of the `gallery_top_k` (3) closest stored samples.
- Seed re-match: once a track's 4 seed samples agree, their mean is compared to identities not claimed in this frame. At ≥ `seed_match_threshold` (0.50) with margin, the track joins that identity instead of creating a new one.
- Merge: pair score = ½ mean·mean + ½ mean of the 5 closest cross-gallery pairs ≥ `merge_threshold` (0.50, was 0.70). Checked every `merge_check_interval` (1 s). **Vetoed** when the two identities were ever assigned to different faces in the same frame; that `co_occurring` set is persisted in MongoDB.
- Stray absorption uses the same pair score and veto.
- `min_confirm_seconds` 1 → 3 s, so a stray stays tentative long enough to be absorbed.
- MongoDB `model_key` = `<model>-<crop_mode>` (tight-crop identities are not mixed with aligned ones). New U numbers continue after the highest number stored under any key, so downstream links (PersonManager `face_recognition:U7`) are never reused for another face.
- `record_face_dataset.py` uses the node's crop path and saves every crop as JPEG (`crop_jpg`), so future live sessions can be re-embedded with any model.

**Results:**

| Test | created | final | purity | ids per person | coverage P0–P4 | FPs labeled | store writes |
|---|---|---|---|---|---|---|---|
| video_3 replay, tight crops, 3 loops + restart (entry 1 manager) | 5 | 5 | 1.00 | 1 | 1.00/0.83/0.98/1.00/0.24 | 0/49 | 90 |
| video_3 replay, tight crops, 3 loops + restart (entry 4 manager) | 5 | 5 | 1.00 | 1 | 1.00/0.95/0.99/1.00/0.24 | 0/49 | 28 |
| video_3 replay, aligned crops, 3 loops + restart (entry 4 manager) | 5 | 5 | 1.00 | 1 | 1.00/1.00/1.00/0.99/0.17 | 1/49 | 24 |
| Real node code path (`_process_landmarks_array_batch`), aligned, 2 loops | 5 | 5 | 1:1 mapping | 1 | – | 1/98 | – |

Live-fragment scenes (`tools/evaluate_live_fragments.py`, stored tight-crop galleries of entry 2, which hold only 3–8 distinct samples after de-duplication):

| Scene | Result |
|---|---|
| user_changes_look (U7 → away → U13 → U7) | 2 identities (user look pair score 0.39 < 0.50) |
| wife_photos (U8 → U9 → U10) | 2 identities: U8 + U9 joined, U10 separate |
| user_and_wife_together | user 2, wife 1; no different people merged |

**Conclusions:**
- No regression on video_3; coverage of the woman at the table rose from 0.83 to 0.95–1.00, and MongoDB writes dropped ~3× because static faces no longer rewrite the gallery.
- The stored live galleries are tight-crop embeddings of consecutive frames, so they cannot show whether aligned crops separate "user, different look" from "different people". This needs a new live session recorded with `record_face_dataset.py`, which now also saves the crops.
- Unit tests (`test/test_identity_manager.py`, 20) cover: a returning person with a new look keeps the identity, lookalikes seen together are never merged, a static face does not grow the gallery or trigger writes, eviction keeps distinct looks, 5-point alignment, and U numbers never reused.

**Next:** live RealSense session with a fresh faces DB, recorded, then replayed to calibrate `seed_match_threshold` / `merge_threshold` on aligned crops. Evaluate `dlib_face_recognition_resnet_model_v1` (public domain) on those crops.

