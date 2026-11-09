#!/usr/bin/env python3
"""
migrate_part_ab.py

Rules
-----
* Nothing is ever deleted or edited in place. Legacy outputs are hashed first
  (``snapshot``), then moved (renamed) under <drive_root>/legacy_v1/.
* An array is re-signed (copied byte for byte, new DONE.json with the v2
  digest and a ``migrated_from`` record) only if every input that determines
  its floating-point values is provably unchanged: identity intensity
  (a_alpha == 1.0 and b_alpha == 0.0 exactly), the same tile side T and
  offsets, held-out list, border rule, margin, model checkpoints and frozen
  statistics, batch size, chunk and shard sizes, OpenCV, torch, CUDA and GPU.
  Everything else is RECOMPUTE (in the new grid, numbers change), RETIRED (its
  level is not in the new grid) or REJECT (integrity failure).
* Re-signed conditions are spot-recomputed by run_part_a_internal_simulation.py
  on first use (first chunk of tiles, bitwise comparison).
* New identifiers (T{T}_..., conditions_v2/, shards_v2/) never coincide with
  legacy ones (m{m}_...), so no digest or directory can be silently reused.

Subcommands
-----------
  snapshot   hash every output file; copy small provenance JSONs to migration/legacy_copies/
  plan       classify every legacy Part A condition -> migration/part_a_plan.json (review it)
  apply      --accept <sha256 of the plan>: re-sign, archive, adopt the external calibration cache
  retire     list legacy_v1/ (what could be deleted by the author after the new run passed); deletes nothing

Author: Rodrigo Sá
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import colab_state as cs                      # noqa: E402

TOOL_VERSION = "migrate_part_ab/1.0"
EXCLUDE_TOP = {"inputs", "code", "migration", "legacy_v1", "manuscript_v2"}
PROVENANCE = ("unit_checks/unit_check_report.json", "part_a/part_a_plan_frozen.json", "part_a/part_a_summary.json",
              "part_a/calibration/calibration.json", "part_a/c0_regression_gate.json",
              "part_b/part_b_plan_frozen.json", "part_b/experimental_summary_for_manuscript.json",
              "part_b/inference_record.json")
INVARIANT, SUPPLEMENTARY, RECOMPUTE, RETIRED, REJECT = "INVARIANT", "SUPPLEMENTARY", "RECOMPUTE", "RETIRED", "REJECT"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[migrate] {msg}", flush=True)


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def legacy_artifacts(root: Path, spec_path: Path) -> List[str]:
    """Outputs that were produced under the legacy geometry and still sit where the new run would look."""
    found = []
    spec = load_json(spec_path)
    if spec is not None and spec.get("schema") != "geometry_spec/2.0":
        found.append("geometry_spec (legacy schema)")
    if (root / "part_a" / "conditions").is_dir():
        found.append("part_a/conditions")
    cal = load_json(root / "part_a" / "calibration" / "calibration.json")
    if cal is not None and cal.get("schema") != "calibration/2.0":
        found.append("part_a/calibration (legacy)")
    if any((root / "part_b" / "conditions").glob("E-cal*")):
        found.append("part_b/conditions/E-cal*")
    if (root / "manuscript").is_dir():
        found.append("manuscript")
    return found


def classify_outcome(spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """L: legacy 0.043 fallback (T near 554); M: metadata external + rounded internal (T 1121-1123); E: exact."""
    if not spec:
        return {"outcome": "UNKNOWN", "reason": "no legacy geometry_spec.json"}
    em = spec.get("external_matched", {})
    T = spec.get("external_matched_T_used")
    out = {"T": T, "p_ext_um": em.get("p_ext_um"), "p_ext_source": em.get("p_ext_source"),
           "p_int_source": em.get("p_int_source"), "T_star": em.get("T_star")}
    if T is None:
        out["outcome"] = "UNKNOWN"
    elif abs(int(T) - 554) <= 8:            # legacy T from 0.175 / 0.043 um/px
        out["outcome"] = "L"
    elif 1118 <= int(T) <= 1123:
        out["outcome"] = "M"
    elif int(T) == 1124:
        out["outcome"] = "E"
    else:
        out["outcome"] = "UNKNOWN"
    return out


# -----------------------------------------------------------------------------
# snapshot
# -----------------------------------------------------------------------------

def snapshot(root: Path, spec_path: Path) -> Dict[str, Any]:
    mdir = root / "migration"
    man = mdir / "legacy_manifest.json"
    if man.exists():
        raise RuntimeError(f"{man} exists; a snapshot is taken once")
    files = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or rel.parts[0] in EXCLUDE_TOP or p.name.endswith(".tmp"):
            continue
        files.append({"path": rel.as_posix(), "bytes": p.stat().st_size, "sha256": cs.sha256_file(p)})
    copies = mdir / "legacy_copies"
    copied = {}
    for rel in (*PROVENANCE, spec_path.name if spec_path.parent == root else None):
        if rel is None:
            continue
        src = root / rel
        if src.is_file():
            dst = copies / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied[rel] = cs.sha256_file(dst)
    if spec_path.is_file() and spec_path.parent != root:
        dst = copies / "geometry_spec.json"
        shutil.copy2(spec_path, dst)
        copied["geometry_spec.json"] = cs.sha256_file(dst)
    rec = {"tool": TOOL_VERSION, "utc": utc(), "drive_root": str(root),
           "legacy_geometry": classify_outcome(load_json(spec_path)), "legacy_artifacts": legacy_artifacts(root, spec_path),
           "n_files": len(files), "files": files, "provenance_copies": copied}
    cs.write_bytes_atomic(man, cs.json_bytes(rec))
    log(f"snapshot: {len(files)} files hashed; legacy geometry {rec['legacy_geometry']}")
    return rec


# -----------------------------------------------------------------------------
# plan
# -----------------------------------------------------------------------------

def _current(root: Path, spec_path: Path, unit_dir: Path, split_path: Path) -> Dict[str, Any]:
    import geometry_harmonization as geo
    import run_part_a_internal_simulation as pa
    import verify_unit_checks as vuc

    spec = json.loads(spec_path.read_text())
    levels = geo.levels_from_spec(spec)
    rep = json.loads((unit_dir / "unit_check_report.json").read_text())
    chk = rep["checks"]
    a2 = chk.get("a2_C0_frozen_predictions", {})
    consts = a2.get("constants", {})
    models = {k: {"sha256": consts[k]["sha256"], "mu": consts[k]["mu"], "sigma": consts[k]["sigma"]}
              for k in pa.MODEL_KEYS if k in consts}
    split = json.loads(split_path.read_text())
    bases = sorted(split["test_base_paths"], key=lambda b: (vuc.slide_of(b), b))
    return {"spec": spec, "levels": {lv.T: lv for lv in levels}, "geometry_digest": spec["geometry_digest"],
            "border_rule": chk["d0_border_rule_and_cv2_parity"]["selected_border_rule"],
            "cv2_version": chk["environment"]["opencv"]["cv2_version"], "models": models,
            "torch": {k: (a2.get("torch") or {}).get(k) for k in ("torch", "cuda", "gpu")},
            "bases_sha": pa.digest_of(bases), "batch_size": pa.DEFAULT_BATCH_SIZE,
            "chunk_tiles": pa.DEFAULT_CHUNK_TILES, "shard_tiles": pa.SHARD_TILES, "margin": geo.MARGIN_PX}


def _legacy(root: Path) -> Dict[str, Any]:
    lc = root / "migration" / "legacy_copies"
    plan = load_json(lc / "part_a" / "part_a_plan_frozen.json") or {}
    summ = load_json(lc / "part_a" / "part_a_summary.json") or {}
    pl = plan.get("plan", {})
    args = summ.get("args", {})
    env = (summ.get("environment") or {}).get("torch_config") or {}
    return {"models": {k: {"sha256": v.get("sha256"), "mu": v.get("mu"), "sigma": v.get("sigma")}
                       for k, v in (pl.get("models") or {}).items()},
            "bases_sha": (pl.get("data") or {}).get("heldout_bases_digest"), "border_rule": pl.get("border_rule"),
            "batch_size": pl.get("batch_size"),
            "chunk_tiles": int(args["chunk_tiles"]) if str(args.get("chunk_tiles", "")).isdigit() else None,
            "cv2_version": (summ.get("gate") or {}).get("cv2_version"),
            "torch": {k: env.get(k) for k in ("torch", "cuda", "gpu")},
            "plan_digest": plan.get("plan_digest")}


def classify_condition(cdir: Path, cur: Dict[str, Any], leg: Dict[str, Any]) -> Dict[str, Any]:
    """One legacy condition directory -> class, reasons, and (if re-signable) its v2 identity. Pure apart from reads."""
    import geometry_harmonization as geo
    import run_part_a_internal_simulation as pa

    rec: Dict[str, Any] = {"legacy_cid": cdir.name, "legacy_dir": str(cdir)}
    done = load_json(cdir / "DONE.json")
    if done is None or not cs.verify_files(cdir, done):
        return {**rec, "class": REJECT, "reasons": ["DONE.json missing or files fail their SHA-256"]}
    rec["legacy_digest"] = done.get("digest")
    rec["legacy_done_sha256"] = cs.sha256_file(cdir / "DONE.json")
    rec["files"] = done["files"]
    p = done.get("params", {})
    T, a_al, b_al, alpha = p.get("T"), p.get("a_alpha"), p.get("b_alpha"), p.get("alpha")
    rec.update(T=T, alpha=alpha, a_alpha=a_al, b_alpha=b_al)
    if T is None or a_al is None or b_al is None:
        return {**rec, "class": REJECT, "reasons": ["legacy params lack T / a_alpha / b_alpha"]}
    if not (float(a_al) == 1.0 and float(b_al) == 0.0):
        return {**rec, "class": RECOMPUTE if int(T) in cur["levels"] else RETIRED,
                "reasons": ["intensity map depends on the legacy calibration (a, b)"]}
    lv = cur["levels"].get(int(T))
    if lv is None:
        return {**rec, "class": RETIRED, "reasons": [f"T = {T} is not a level of the new grid "
                                                     "(add it to geometry.supplementary_T to keep it)"]}
    checks = {
        "offsets": (list(p.get("offsets") or []), list(lv.offsets)),
        "bases_sha": (leg["bases_sha"], cur["bases_sha"]),
        "border_rule": (leg["border_rule"], cur["border_rule"]),
        "batch_size": (leg["batch_size"], cur["batch_size"]),
        "chunk_tiles": (leg["chunk_tiles"], cur["chunk_tiles"]),
        "shard_tiles": (done.get("shard_tiles"), cur["shard_tiles"]),
        "cv2_version": (leg["cv2_version"], cur["cv2_version"]),
        "models": (leg["models"], cur["models"]),
        "torch_cuda_gpu": (leg["torch"], cur["torch"]),
    }
    bad = [f"{k}: legacy {a!r} != current {b!r}" for k, (a, b) in checks.items() if a is None or a != b]
    if bad:
        return {**rec, "class": RECOMPUTE, "reasons": bad}
    role = "primary" if lv.T in (geo.N_INT, geo.external_level(list(cur["levels"].values())).T) else "surface"
    if lv.role == geo.ROLE_SUPPLEMENTARY:
        role = "supplementary"
    new_cid = pa.cid_of(lv.T, 0.0, "point")
    new_dg = pa.condition_digest_v2(bases_sha=cur["bases_sha"], T=lv.T, offsets=lv.offsets, margin=cur["margin"],
                                    border_rule=cur["border_rule"], a_alpha=1.0, b_alpha=0.0, models=cur["models"],
                                    batch_size=cur["batch_size"], chunk_tiles=cur["chunk_tiles"],
                                    shard_tiles=cur["shard_tiles"], cv2_version=cur["cv2_version"])
    p_int = float(cur["spec"]["external_matched"]["p_int_um"])
    params = {"cid": new_cid, "T": lv.T, "m": lv.m, "m_exact": f"{lv.m_exact.numerator}/{lv.m_exact.denominator}",
              "m_label": lv.label, "level_role": lv.role, "alpha": 0.0, "ab_source": "point", "role": role,
              "a_source": None, "b_source": None, "a_alpha": 1.0, "b_alpha": 0.0, "identity_intensity": True,
              "n_tiles_per_field": lv.n_tiles, "offsets": list(lv.offsets),
              "frame_edge_share_of_tile_sides": lv.frame_edge_share_of_tile_sides,
              "fov_um": lv.T * p_int, "output_spacing_um": lv.T * p_int / geo.D}
    return {**rec, "class": SUPPLEMENTARY if lv.role == geo.ROLE_SUPPLEMENTARY else INVARIANT,
            "reasons": ["identity intensity; every array-determining input unchanged"],
            "new_cid": new_cid, "new_digest": new_dg, "new_params": params}


def make_plan(root: Path, spec_path: Path, unit_dir: Path, split_path: Path) -> Dict[str, Any]:
    if not (root / "migration" / "legacy_manifest.json").exists():
        raise RuntimeError("run 'snapshot' first")
    cur, leg = _current(root, spec_path, unit_dir, split_path), _legacy(root)
    conds = sorted((root / "part_a" / "conditions").glob("*/")) if (root / "part_a" / "conditions").is_dir() else []
    entries = [classify_condition(c, cur, leg) for c in conds]
    new_cids = [e["new_cid"] for e in entries if e.get("new_cid")]
    if len(new_cids) != len(set(new_cids)):
        raise RuntimeError(f"two legacy conditions map to one new identifier: {new_cids}")
    archive = [r for r in ("part_a/conditions", "part_a/shards", "part_a/calibration", "part_a/part_a_plan_frozen.json",
                           "part_a/part_a_summary.json", "part_a/part_a_slide_level.csv", "part_a/c0_regression_gate.json",
                           "part_b/shards", "part_b/part_b_plan_frozen.json", "part_b/part_b_statistics_full.json",
                           "part_b/experimental_summary_for_manuscript.json", "part_b/prep_equivalence_gate.json",
                           "part_b/inference_record.json", "manuscript") if (root / r).exists()]
    archive += sorted(str(p.relative_to(root)) for p in (root / "part_b" / "conditions").glob("E-cal*"))
    counts: Dict[str, int] = {}
    for e in entries:
        counts[e["class"]] = counts.get(e["class"], 0) + 1
    plan = {"tool": TOOL_VERSION, "utc": utc(), "geometry_digest": cur["geometry_digest"],
            "legacy_manifest_sha256": cs.sha256_file(root / "migration" / "legacy_manifest.json"),
            "legacy_plan_digest": leg["plan_digest"], "counts": counts, "conditions": entries,
            "archive_to_legacy_v1": archive,
            "keep_in_place": ["part_b/conditions/E-locked", "part_b/conditions/E-z (digest verified at apply)",
                              "part_a/calibration external_frame_stats CSV (validated copy)"]}
    out = root / "migration" / "part_a_plan.json"
    cs.write_bytes_atomic(out, cs.json_bytes(plan))
    log(f"plan: {counts}; archive {len(archive)} paths; review {out} (sha256 {cs.sha256_file(out)})")
    return plan


# -----------------------------------------------------------------------------
# apply
# -----------------------------------------------------------------------------

def _adopt_external_cache(root: Path, legacy_cal_dir: Path) -> Dict[str, Any]:
    import photometric_calibration as pc

    params = pc.MaskParams()
    old = legacy_cal_dir / f"external_frame_stats_{params.digest()}.csv"
    if not old.is_file():
        return {"adopted": False, "reason": f"{old.name} not found"}
    cal = load_json(legacy_cal_dir / "calibration.json") or {}
    pin = (cal.get("provenance") or {}).get("pinned_preprocess_sha256")
    if pin != pc.PINNED_PREPROCESS_SHA256:
        return {"adopted": False, "reason": f"legacy pinned sha {pin} != current pin"}
    if cal.get("mask_params_digest") not in (None, params.digest()):
        return {"adopted": False, "reason": "mask parameters differ"}
    with old.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1253 or len({r["slide_id"] for r in rows}) != 24 or len({r["relative_path"] for r in rows}) != 1253:
        return {"adopted": False, "reason": f"{len(rows)} rows: not a complete external cache"}
    new = root / "part_a" / "calibration" / pc.external_cache_name(params)
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old, new)
    if cs.sha256_file(new) != cs.sha256_file(old):
        raise RuntimeError("external cache copy differs from its source")
    return {"adopted": True, "from": str(old), "to": str(new), "sha256": cs.sha256_file(new)}


def _check_part_b_invariant(root: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """E-locked / E-z DONE.json digests must equal the refactored pipe_digest (they carry no geometry)."""
    import run_part_b_and_export as pb

    leg = load_json(root / "migration" / "legacy_copies" / "part_b" / "part_b_plan_frozen.json")
    if leg is None:
        return {"checked": False, "reason": "no legacy Part B plan"}
    model_ids = {k: [v["sha256"], v["mu"], v["sigma"]] for k, v in leg["plan"]["models"].items()}
    with Path(cfg["paths"]["external_manifest"]).open(newline="", encoding="utf-8") as f:
        rel = sorted(r["relative_path"] for r in csv.DictReader(f))
    out = {}
    for name in ("E-locked", "E-z"):
        done = load_json(root / "part_b" / "conditions" / name / "DONE.json")
        if done is None:
            out[name] = "absent (will be computed)"
            continue
        mode, src, label = pb.PIPELINES[name]
        params = {"pipeline": name, "mode": mode, "ab_source": src, "preprocessing_mode": label}
        expect = pb.pipe_digest_from(params, pb.digest_of(rel), model_ids, 32)
        if done.get("digest") != expect:
            raise RuntimeError(f"part_b/{name}: stored digest differs from the refactored pipe_digest; the invariance "
                               "of E-locked/E-z is not established (investigate before recomputing)")
        out[name] = "digest unchanged"
    return {"checked": True, **out}


def apply(root: Path, cfg: Dict[str, Any], accept: str) -> Dict[str, Any]:
    mdir = root / "migration"
    plan_path = mdir / "part_a_plan.json"
    if cs.sha256_file(plan_path) != accept:
        raise RuntimeError(f"--accept {accept[:12]} does not match {plan_path} ({cs.sha256_file(plan_path)[:12]})")
    if (mdir / "applied.json").exists():
        raise RuntimeError("migration already applied")
    plan = json.loads(plan_path.read_text())
    pb_check = _check_part_b_invariant(root, cfg)
    v2 = root / "part_a" / "conditions_v2"
    resigned = []
    for e in plan["conditions"]:
        if e["class"] not in (INVARIANT, SUPPLEMENTARY):
            continue
        src, dst = Path(e["legacy_dir"]), v2 / e["new_cid"]
        payload = {n: (src / n).read_bytes() for n in e["files"]}
        for n, b in payload.items():
            if cs.sha256_bytes(b) != e["files"][n]["sha256"]:
                raise RuntimeError(f"{src / n} changed since the plan")
        if dst.exists():
            old = load_json(dst / "DONE.json") or {}
            if old.get("digest") == e["new_digest"] and cs.verify_files(dst, old) and \
                    all(old["files"][n]["sha256"] == e["files"][n]["sha256"] for n in e["files"]):
                resigned.append(e["new_cid"])
                continue
            raise RuntimeError(f"{dst} exists with different content: refusing to overwrite")
        marker = {"digest": e["new_digest"], "params": e["new_params"], "n_tiles": None,
                  "migrated_from": {"legacy_cid": e["legacy_cid"], "legacy_digest": e["legacy_digest"],
                                    "legacy_done_sha256": e["legacy_done_sha256"],
                                    "legacy_manifest_sha256": plan["legacy_manifest_sha256"],
                                    "migration_plan_sha256": accept, "tool": TOOL_VERSION,
                                    "tool_sha256": cs.sha256_file(Path(__file__)), "utc": utc()}}
        cs.write_verified_files(dst, payload, "DONE.json", marker)
        new_done = json.loads((dst / "DONE.json").read_text())
        if any(new_done["files"][n]["sha256"] != e["files"][n]["sha256"] for n in e["files"]):
            raise RuntimeError(f"{dst}: re-signed arrays differ from the legacy bytes")
        resigned.append(e["new_cid"])
    moved = []
    for rel in plan["archive_to_legacy_v1"]:
        src = root / rel
        if not src.exists():
            continue
        dst = root / "legacy_v1" / rel
        if dst.exists():
            raise RuntimeError(f"{dst} already exists")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        moved.append(rel)
    ext = _adopt_external_cache(root, root / "legacy_v1" / "part_a" / "calibration")
    rec = {"tool": TOOL_VERSION, "utc": utc(), "migration_plan_sha256": accept, "resigned": resigned,
           "moved_to_legacy_v1": moved, "external_calibration_cache": ext, "part_b_invariance": pb_check}
    cs.write_bytes_atomic(mdir / "applied.json", cs.json_bytes(rec))
    log(f"apply: re-signed {len(resigned)} conditions, archived {len(moved)} paths, external cache {ext.get('adopted')}")
    return rec


def retire(root: Path) -> List[Dict[str, Any]]:
    base = root / "legacy_v1"
    items = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size}
             for p in sorted(base.rglob("*")) if p.is_file()] if base.exists() else []
    log(f"retire (dry run, nothing deleted): {len(items)} files, {sum(i['bytes'] for i in items) / 1e6:.1f} MB "
        "under legacy_v1/; delete them yourself only after the new run passed every gate")
    return items


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Migrate legacy (m = 2.165) Part A/B outputs to the optics geometry.")
    p.add_argument("--drive-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("snapshot")
    s1.add_argument("--geometry-spec", type=Path, required=True)
    s2 = sub.add_parser("plan")
    s2.add_argument("--geometry-spec", type=Path, required=True)
    s2.add_argument("--unit-check-dir", type=Path, required=True)
    s2.add_argument("--split", type=Path, required=True)
    s3 = sub.add_parser("apply")
    s3.add_argument("--accept", required=True, help="SHA-256 of migration/part_a_plan.json")
    sub.add_parser("retire")
    a = p.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    if a.cmd == "snapshot":
        snapshot(a.drive_root, a.geometry_spec)
    elif a.cmd == "plan":
        make_plan(a.drive_root, a.geometry_spec, a.unit_check_dir, a.split)
    elif a.cmd == "apply":
        apply(a.drive_root, cfg, a.accept)
    else:
        retire(a.drive_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
