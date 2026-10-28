#!/usr/bin/env python3
"""Replay the labeled 2026-10-21 lab recordings (Pep, Oscar, Juan) through FaceIdentityManager.

Session 1 replays live_2026-10-21_team.npz then live_2026-10-21_team2.npz on an empty gallery;
session 2 restarts from the store and replays team2 again. Only CONFIRMED assignments count,
because only those reach PersonManager. Reports, per person: share of frames on their main id,
the confirmed ids that labeled more than 1% of their frames (1 is ideal), and how many frames got
another person's id. Track labels were checked on the saved crops (see entry 5 of
plans/face_identity_experiments.md). Recordings live in database/eval/ (gitignored).

  python3 tools/evaluate_team_session.py '{"similarity_threshold": 0.55, "learn_threshold": 0.65, ...}'

Face-quality keys (min_learn_face_height_px, profile_eye_ratio, frontal_eye_ratio) are accepted
next to the FaceIdentityManager ones."""
import collections, copy, json, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from face_recognition.identity_manager import FaceIdentityManager
from face_recognition.face_quality import FaceQualityConfig, face_quality, LEFT_EYE_INSIDE, RIGHT_EYE_INSIDE

PEP, OSCAR, JUAN = 0, 1, 2
LABELS = {
 "live_2026-10-21_team.npz": {"face_785": PEP, "face_786": OSCAR, "face_787": JUAN},
 "live_2026-10-21_team2.npz": {
   **{f"face_{n}": PEP for n in (810, 813, 817, 821, 827, 829, 841, 847, 825)},
   **{f"face_{n}": OSCAR for n in (807, 811, 814, 818, 823, 830, 840, 845, 848, 850, 852)},
   **{f"face_{n}": JUAN for n in (799, 812, 815, 819, 822, 831, 844, 849, 826, 828)}},
}
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "eval") + os.sep

def load(name, prefix):
    z = np.load(BASE + name); d = {k: z[k] for k in z.files if k not in ("crop_jpg", "crop_mode")}
    lab = np.array([LABELS[name].get(str(f), -1) for f in d["face_id"]])
    if name == "live_2026-10-21_team2.npz":  # tracker reused face_850 for Juan from 81.5 s (checked on the crops)
        lab[(d["face_id"] == "face_850") & ((d["stamp_ns"] - d["stamp_ns"].min()) / 1e9 > 81.5)] = JUAN
    d["face_id"] = np.array([prefix + str(f) for f in d["face_id"]])
    return d, lab

class Store:
    def __init__(s): s.docs = {}
    def load(s): return [copy.deepcopy(v) for v in s.docs.values()]
    def save(s, i): s.docs[i.unique_id] = copy.deepcopy(i)
    def delete(s, u): s.docs.pop(u, None)
    def mark_merged(s, u, k): s.docs.pop(u, None); return True
    def close(s): pass

class Quiet:
    def __getattr__(s, n): return lambda *a, **k: None

def qual(d, qc):
    lm = d["landmarks"]; out = []
    for i in range(len(lm)):
        hp = (d["bbox"][i, 3] - d["bbox"][i, 1]) * d["height"][i]
        out.append(face_quality(d["bbox"][i], float(d["bbox_conf"][i]),
                   lm[i, LEFT_EYE_INSIDE, 0] if lm[i, LEFT_EYE_INSIDE, 2] > 0 else None,
                   lm[i, RIGHT_EYE_INSIDE, 0] if lm[i, RIGHT_EYE_INSIDE, 2] > 0 else None, qc, hp))
    return np.array(out)

def run(params, min_h=50, verbose=False):
    params = dict(params); qc = FaceQualityConfig(min_learn_face_height_px=params.pop("min_learn_face_height_px", 0.0),
                                                  profile_eye_ratio=params.pop("profile_eye_ratio", 0.10),
                                                  frontal_eye_ratio=params.pop("frontal_eye_ratio", 0.18))
    store = Store(); clock = {"t": 0.0}
    mk = lambda: FaceIdentityManager(logger=Quiet(), store=store, clock=lambda: clock["t"], **params)
    plan = [("live_2026-10-21_team.npz", "a", False), ("live_2026-10-21_team2.npz", "b", False), ("live_2026-10-21_team2.npz", "c", True)]
    m = mk(); res = {}
    for name, prefix, restart in plan:
        if restart: m.flush(force=True); m = mk()
        d, lab = load(name, prefix); q = qual(d, qc)
        h = (d["bbox"][:, 3] - d["bbox"][:, 1]) * d["height"]
        st = d["stamp_ns"]; t0 = st.min(); base = clock["t"] + 5.0
        for s in np.unique(st):
            rows = [r for r in np.where(st == s)[0] if h[r] >= min_h]
            clock["t"] = base + (s - t0) / 1e9
            if not rows: continue
            a = m.process_new_embedding_batch({d["face_id"][r]: d["emb_matched"][r] for r in rows}, {d["face_id"][r]: float(q[r]) for r in rows})
            for r in rows:
                x = a[d["face_id"][r]]
                res[(prefix, r)] = (x.unique_id if x.status == 2 else None, lab[r])  # what PersonManager acts on
        for r in np.where(h < min_h)[0]: res[(prefix, r)] = (None, lab[r])
    # resolve merged ids: map dropped -> survivor through the store's final state is not tracked; use final clusters only for count
    def stats(keys):
        per = collections.defaultdict(collections.Counter)
        for k in keys:
            u, l = res[k]
            if l >= 0: per[l][u] += 1
        out = {}
        for p, c in per.items():
            n = sum(c.values()); lab = {u: v for u, v in c.items() if u}
            main = max(lab, key=lab.get) if lab else None
            out[["Pep", "Oscar", "Juan"][p]] = {"coverage": round(sum(lab.values()) / n, 2), "main": main,
                "main_share": round(lab.get(main, 0) / n, 2), "ids>5%": sorted(u for u, v in lab.items() if v / n > 0.01)}
        mains = [v["main"] for v in out.values()]
        mains_of = {v["main"]: p for p, v in out.items()}
        wrong = sum(1 for k in keys if res[k][1] >= 0 and res[k][0] in mains_of
                    and mains_of[res[k][0]] != ["Pep", "Oscar", "Juan"][res[k][1]])  # labeled as another of the three
        return out, len(set(mains)) == len(mains), wrong
    s1 = stats([k for k in res if k[0] in "ab"]); s2 = stats([k for k in res if k[0] == "c"])
    return {"session1": s1[0], "unique_mains_1": s1[1], "other_id_rows_1": s1[2],
            "after_restart": s2[0], "unique_mains_2": s2[1], "other_id_rows_2": s2[2],
            "created": m.total_identities_created, "final_ids": sorted(m.identity_clusters)}

if __name__ == "__main__":
    print(json.dumps(run(json.loads(sys.argv[1])), indent=1))
