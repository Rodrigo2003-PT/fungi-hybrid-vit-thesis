#!/usr/bin/env python3
"""
run_all_colab.py

Stages
-----------------------------------------------------------------------
  preflight      every required file, folder, pin and archive exists; GPU; DenseNet
                 checkpoint load mode; manifest and split counts. Always runs (seconds). Writes
                 state/preflight_report.json and stops with the full list of problems.
  stage_inputs   unzip the raw TIFF archives from Drive to local disk (once per session). Each zip is copied
                 to local disk in one sequential read, extracted with CRC-32 checks and path-safety checks,
                 then deleted; a single wrapper folder around the class folders is removed. The result is
                 verified by file count, total size and the expected class folders.
  migrate_snapshot  when outputs of the legacy (m = 2.165) geometry exist: hash every file and copy the small
                 provenance JSONs to migration/ (migrate_part_ab.py snapshot). Nothing is modified.
  geometry       checks that every split field and every manifest image resolves to a staged TIFF, then
                 builds geometry_spec.json (schema 2.0) from the exact optics in the config, verifying the
                 TIFF header of one file per slide. An existing spec is reused only when its geometry
                 digest equals the one the config implies; a legacy spec is archived (renamed), never reused.
  unit_checks    verify_unit_checks.py; skipped when a marker records the same arguments, code and geometry
                 digest and the report SHA-256, and the report is PASS (or CONDITIONAL with accept_conditional).
  migrate        migrate_part_ab.py plan / apply: re-signs legacy Part A arrays that the fix leaves numerically
                 unchanged, archives everything that must be recomputed. Stops until the plan's SHA-256 is
                 written to migration.accept_sha256 in colab_config.json.
  calibration    photometric_calibration.py fit with the crash-safe CSV cache; refuses a calibration.json
                 whose geometry digest differs from the spec;
  part_a         run_part_a_internal_simulation.py; resumes per 2,304-tile shard.
  part_b_infer   run_part_b_and_export.py --stage infer; resumes per 256-image shard.
  part_b_stats   run_part_b_and_export.py --stage stats; resumes per statistics task.
  export         export_manuscript_artifacts.py: LaTeX tables, CSV summaries, figures

Other modes
-----------
  --status       print the progress of every stage from the files on Drive and exit (no GPU, no lock)
  --break-lock   delete the driver and stage locks first; use it only when the session that held them is gone
  --stop-after   run the stages up to and including the named one

Author: Rodrigo Sá
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import colab_state as cs  # noqa: E402

STAGES = ("preflight", "stage_inputs", "migrate_snapshot", "geometry", "unit_checks", "migrate", "calibration",
          "part_a", "part_b_infer", "part_b_stats", "export")
DEFAULT_EST = {"preflight": 30, "stage_inputs": 1200, "migrate_snapshot": 300, "geometry": 180, "unit_checks": 900,
               "migrate": 300, "calibration": 600, "part_a": 600, "part_b_infer": 600, "part_b_stats": 300,
               "export": 60}   # seconds needed to make progress
LEGACY_GEOMETRY_KEYS = ("p_int_um", "p_ext_um", "allow_config_spacing", "n_files")
TIFF_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")
CLASS_NAMES = ("Candida albicans", "Candida glabrata")

# SHA-256 pins of the published external pipeline (ext_common.PINNED_SHA256, repository commit 490673e)
PINS = {
    ("pinned_scripts", "preprocess_external.py"): "3df9a093ced600e5407b32bd07f05a77712725e2fc87faa27b63483d3f96e4c0",
    ("pinned_scripts", "evaluate_external_generalization.py"): "dd49c82483448ffd1fae2dd95f1d85251b1aff35effa58af458204e699c4ada2",
    ("pinned_scripts", "compare_external_pipelines.py"): "efae60db1a012bb319cb2f3083ed47b39a13e6f48c5e9f0077cc05244cc1477f",
    ("hybrid_validation_scripts", "load_frozen_hybrid.py"): "9eb98fd9cf292094e3f22655d5693f12c30d88ef07c1a52bafe669f03ff54962",
    ("densenet_scripts", "densenet_model.py"): "8cf07088fac6397187139e31ff5af4920ea26d941e6a7afc43e671ac45688585",
}
CONFIG_YAML_SHA256 = "b00c6bb64f2b3bb5f20e221ed1301912fdde813de6c4f44e62844708af052dde"
# GitHub ViT/HybridViT/ at commit 490673e, mirrored on Drive at colab/lsa_convtok_vit/ (paths relative to that root).
MIRROR_SHA256 = {
    "preprocessed/split_indices.json": "afc1a70fc2cb2081e49835678efac74f1ea9b260b0a5043dc902974483767dd5",
    "validation/models/final_model_lsa_convtok.pth": "ae5968901ccf15073dbe953837c7f0c21033ac3d759af0f59e7dd0e38abf26e9",
    "validation/source/config.py": "1beed83df40a8876af2667a74c65492f6b5df6e2165bb2817b3485887d8930ab",
    "validation/source/training_logic.py": "d9ed3bec22a1f25e572b91febeb28ba0e388779f0c23896c8427a81c5db7664b",
    "validation/source/training_utils.py": "2b0ff78d4d7059d8316c149ac71364648b2c17902f9edc10a4054d32528b49b4",
    "validation/source/lsa_convtok_vit.py": "6ca5c84e2ae7c1960525e8c122ebf1902c8e53522d9240ad6138f7882c93bf0e",
    "validation/source/lsa_attention.py": "409d8243c09886028f4e071c032bb2ee09f3a0cabf37e92b43fc06ffc38aee89",
    "validation/source/lsa_logger.py": "1c780ebb9beeccb97bc4a943422424cec9373f12b26aeb0d61c6f73257601f43",
}
GIT_LFS_POINTER = b"version https://git-lfs"
REQUIRED_IN_DIR = {
    "densenet_scripts": ("ext_common.py", "ext_stats.py", "compare_external_models.py", "load_frozen_model.py",
                         "densenet_model.py"),
    "hybrid_source": ("config.py", "training_logic.py", "training_utils.py", "lsa_convtok_vit.py", "lsa_attention.py",
                      "lsa_logger.py"),
    "hybrid_validation_scripts": ("load_frozen_hybrid.py",),
    "pinned_scripts": ("preprocess_external.py", "evaluate_external_generalization.py", "compare_external_pipelines.py"),
}
CODE_FILES = ("colab_state.py", "pc_resilience.py", "optics.py", "geometry_harmonization.py", "photometric_calibration.py",
              "verify_unit_checks.py", "run_part_a_internal_simulation.py", "run_part_b_and_export.py",
              "export_manuscript_artifacts.py", "profile_colab_t4.py", "colab_setup_env.py", "migrate_part_ab.py",
              "verify_optical_configuration.py")


def content_digest(args: List[str], files: Dict[str, Optional[Path]], extra: Optional[Dict[str, Any]] = None) -> str:
    """Stage digest from arguments AND the bytes of the inputs that determine the stage's output: a changed
    geometry spec, calibration or upstream summary always invalidates the marker (a missing file -> None)."""
    return cs.sha256_bytes(cs.json_bytes({"args": list(args),
                                          "files": {k: (cs.sha256_file(p) if p is not None and Path(p).is_file() else None)
                                                    for k, p in sorted(files.items())},
                                          "extra": extra or {}}))


def geometry_digest_of(spec_path: Path) -> Optional[str]:
    try:
        return json.loads(Path(spec_path).read_text()).get("geometry_digest")
    except (OSError, ValueError):
        return None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- input staging helpers -------------------------------------------------------------------------------

def source_identity(src: Path) -> Dict[str, Any]:
    st = src.stat()
    return {"path": str(src), "bytes": st.st_size if src.is_file() else None, "mtime": int(st.st_mtime)}


def staged_files(dst: Path) -> List[Path]:
    return [p for p in dst.rglob("*") if p.is_file() and p.name != ".staged.json"
            and not (p.name.startswith(".") and p.name.endswith(".part"))]


def check_expect(dst: Path, item: Dict[str, Any]) -> bool:
    return all((dst / e).exists() for e in item.get("expect", []))


def _is_junk(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return name.startswith("__MACOSX/") or base in (".DS_Store", "Thumbs.db") or base.startswith("._")


def extract_zip(src: Path, dst: Path, *, copy_first: bool, scratch: Path) -> None:
    """Safe extraction: no absolute paths, no '..', no symlinks; macOS/Windows metadata skipped."""
    zpath = src
    with zipfile.ZipFile(src) as zf:
        members = [m for m in zf.infolist() if not m.is_dir() and not _is_junk(m.filename)]
        need = sum(m.file_size for m in members) + (src.stat().st_size if copy_first else 0)
    dst.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dst).free
    if need > free * 0.95:
        raise RuntimeError(f"{src.name}: needs {need / 1e9:.1f} GB on local disk, {free / 1e9:.1f} GB free "
                           "(set copy_first to false to extract directly from Drive)")
    if copy_first:
        scratch.mkdir(parents=True, exist_ok=True)
        zpath = scratch / src.name
        if not (zpath.exists() and zpath.stat().st_size == src.stat().st_size):
            shutil.copyfile(src, zpath)
    try:
        with zipfile.ZipFile(zpath) as zf:
            for m in zf.infolist():
                if m.is_dir() or _is_junk(m.filename):
                    continue
                parts = Path(m.filename).parts
                is_link = (m.external_attr >> 16) & 0o170000 == 0o120000
                if m.filename.startswith(("/", "\\")) or ".." in parts or ":" in parts[0] or is_link:
                    raise RuntimeError(f"{src.name}: unsafe member {m.filename!r}")
                out = dst.joinpath(*parts)
                if out.exists() and out.stat().st_size == m.file_size:
                    continue                      # resumed extraction: already complete
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(f".{out.name}.part")
                with zf.open(m) as fin, open(tmp, "wb") as fout:   # CRC-32 checked by zipfile at EOF
                    shutil.copyfileobj(fin, fout, 1 << 22)
                os.replace(tmp, out)
    finally:
        if copy_first and zpath != src:
            zpath.unlink(missing_ok=True)
    for stray in dst.rglob(".*.part"):      # leftovers of an interrupted earlier extraction
        stray.unlink(missing_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    for p in src.rglob("*"):
        if p.is_file() and not _is_junk(p.relative_to(src).as_posix()):
            q = dst / p.relative_to(src)
            if not q.exists() or q.stat().st_size != p.stat().st_size:
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)


def flatten_single_wrapper(dst: Path, item: Dict[str, Any]) -> Optional[str]:
    """If the expected class folders sit inside exactly one wrapper folder, move them up one level."""
    if check_expect(dst, item) or not item.get("expect"):
        return None
    tops = [p for p in dst.iterdir() if p.name != ".staged.json" and not _is_junk(p.name)]
    dirs = [p for p in tops if p.is_dir()]
    if len(tops) != 1 or len(dirs) != 1 or not check_expect(dirs[0], item):
        return None
    wrapper = dirs[0]
    for child in list(wrapper.iterdir()):
        target = dst / child.name
        if target.exists():
            raise RuntimeError(f"cannot flatten {wrapper}: {target} already exists")
        os.replace(child, target)
    wrapper.rmdir()
    return wrapper.name


def resolve_internal(raw_root: Path, base: str) -> Optional[Path]:
    for suf in TIFF_SUFFIXES:
        p = raw_root / f"{base}{suf}"
        if p.exists():
            return p
    return None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nvidia_gpu_name() -> Optional[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30)
        name = out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 and out.stdout.strip() else None
        return name
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


class Driver:
    def __init__(self, cfg: Dict[str, Any], stop_after: Optional[str]) -> None:
        self.cfg, self.stop_after = cfg, stop_after
        self.code = Path(cfg.get("code_dir", HERE))
        self.root = Path(cfg["drive_root"])
        self.local = Path(cfg.get("local_root", "/content/partab_local"))
        self.state = self.root / "state"
        self.state.mkdir(parents=True, exist_ok=True)
        self.budget = cs.SessionBudget(cfg.get("budget_hours", 5.0), cfg.get("margin_minutes", 20.0))
        os.environ[cs.SESSION_ENV] = str(self.budget.start)
        self.timings = cs.Timings(self.state / "stage_timings.json")
        self.lock = cs.RunLock(self.state / "DRIVER_LOCK.json")
        self.P = cfg["paths"]
        self.opt = cfg.get("options", {})

    # ---- helpers ----------------------------------------------------------------------------------------
    def marker(self, name: str) -> Path:
        return self.state / f"{name}.done.json"

    def marker_ok(self, name: str, digest: str) -> bool:
        try:
            rec = json.loads(self.marker(name).read_text())
        except (OSError, ValueError):
            return False
        return rec.get("digest") == digest and all(
            Path(p).exists() and cs.sha256_file(Path(p)) == h for p, h in rec.get("outputs", {}).items())

    def mark(self, name: str, digest: str, outputs: List[Path], extra: Optional[Dict[str, Any]] = None) -> None:
        cs.write_bytes_atomic(self.marker(name), cs.json_bytes(
            {"digest": digest, "outputs": {str(p): cs.sha256_file(p) for p in outputs},
             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {})}))

    def run(self, name: str, cmd: List[str]) -> int:
        need = self.timings.estimate(name, DEFAULT_EST[name]) if name in ("unit_checks",) else DEFAULT_EST[name]
        self.budget.require(need, name)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        log(f"{name}: start ({self.budget.remaining() / 60:.0f} min of budget left)")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=str(self.code), env=env)
        rc: Optional[int] = None
        while rc is None:
            try:
                rc = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.lock.beat()
                if self.budget.remaining() <= 0 and time.time() - t0 > 60:
                    log(f"{name}: budget reached, stopping the child process")
                    proc.terminate()
                    try:
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    rc = cs.EXIT_BUDGET
        if rc == 0:
            self.timings.record(name, time.time() - t0)
        log(f"{name}: exit {rc} after {(time.time() - t0) / 60:.1f} min")
        return rc

    def py(self, script: str, *args: str) -> List[str]:
        return [sys.executable, str(self.code / script), *args]

    def budget_args(self) -> List[str]:
        return ["--budget-hours", str(self.cfg.get("budget_hours", 5.0)),
                "--margin-minutes", str(self.cfg.get("margin_minutes", 20.0))]

    def model_args(self) -> List[str]:
        P = self.P
        out = ["--densenet-scripts", P["densenet_scripts"], "--densenet-ckpt", P["densenet_ckpt"],
               "--hybrid-validation-scripts", P["hybrid_validation_scripts"], "--hybrid-source", P["hybrid_source"],
               "--hybrid-ckpt", P["hybrid_ckpt"], "--hybrid-config", P["hybrid_config"]]
        if self.opt.get("allow_unsafe_densenet_load"):
            out.append("--allow-unsafe-densenet-load")
        return out

    def cond(self) -> List[str]:
        return ["--accept-conditional"] if self.opt.get("accept_conditional") else []

    def npy_root(self) -> Optional[str]:
        p = self.P.get("npy_root")
        if p and Path(p).is_dir() and all((Path(p) / c).is_dir() for c in CLASS_NAMES):
            return p
        return None

    # ---- stages -----------------------------------------------------------------------------------------
    def preflight(self) -> int:
        P, problems, notes = self.P, [], []
        rep: Dict[str, Any] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

        for f in CODE_FILES:
            if not (self.code / f).exists():
                problems.append(f"code file missing: {self.code / f}")
        mirror = self.cfg.get("github_mirror", {})
        mroot = Path(mirror.get("root", "/content/drive/MyDrive/colab/lsa_convtok_vit"))
        mres: Dict[str, Any] = {}
        if not mroot.is_dir():
            problems.append(f"GitHub mirror folder not found: {mroot} (expected the ViT/HybridViT tree)")
        else:
            for rel, sha in MIRROR_SHA256.items():
                p = mroot / rel
                if not p.is_file():
                    mres[rel] = "missing"
                    continue
                with p.open("rb") as fh:
                    if fh.read(len(GIT_LFS_POINTER)) == GIT_LFS_POINTER:
                        problems.append(f"{p} is a Git LFS pointer, not the file: replace it with the real file")
                        mres[rel] = "lfs_pointer"
                        continue
                got = cs.sha256_file(p)
                mres[rel] = "matches GitHub" if got == sha else f"differs ({got[:12]})"
                if got != sha:
                    notes.append(f"{p} differs from GitHub commit {mirror.get('commit', '490673e')[:7]}; "
                                 "the C0 gate against main.tex decides whether it is usable")
        rep["github_mirror"] = {"root": str(mroot), "files": mres}

        legacy = [k for k in LEGACY_GEOMETRY_KEYS if k in (self.cfg.get("geometry") or {})]
        if legacy:
            problems.append(f"colab_config.json geometry has removed keys {legacy}: spacing comes only from the "
                            "'optics' block (no override, no rounded fallback)")
        try:
            import geometry_harmonization as geo
            exp = geo.expected_geometry(self.cfg)
            em = exp["external_matched"]
            rep["optics"] = {"p_int_exact": em["p_int_exact"], "p_ext_exact": em["p_ext_exact"],
                             "fov_um": [em["fov_int_um"], em["fov_ext_um"]], "T": em["T"],
                             "m_external": em["m_external"], "geometry_digest": exp["geometry_digest"]}
            log(f"preflight optics: p_int {em['p_int_um']:.6f}, p_ext {em['p_ext_um']:.6f} um/px; FOV "
                f"{em['fov_int_um']:.2f} / {em['fov_ext_um']:.2f} um; T = {em['T']}; m_ext = {em['m_external']:.5f}")
        except Exception as exc:
            problems.append(f"optics configuration invalid: {type(exc).__name__}: {exc}")
        for key in ("split", "external_manifest", "densenet_ckpt", "hybrid_ckpt", "hybrid_config"):
            if not Path(P[key]).is_file():
                problems.append(f"paths.{key} not found: {P[key]}")
        for key, names in REQUIRED_IN_DIR.items():
            for n in names:
                if not (Path(P[key]) / n).is_file():
                    problems.append(f"paths.{key}: {n} not found in {P[key]}")
        pins = {}
        for (key, name), sha in PINS.items():
            p = Path(P[key]) / name
            if p.is_file():
                got = cs.sha256_file(p)
                pins[str(p)] = got == sha
                if got != sha:
                    problems.append(f"{p}: SHA-256 {got[:12]} differs from the published pin {sha[:12]}")
        if Path(P["hybrid_config"]).is_file():
            got = cs.sha256_file(Path(P["hybrid_config"]))
            pins[P["hybrid_config"]] = got == CONFIG_YAML_SHA256
            if got != CONFIG_YAML_SHA256:
                problems.append(f"{P['hybrid_config']}: SHA-256 differs from the published pin")
        rep["pins_ok"] = pins

        if Path(P["split"]).is_file():
            split = json.loads(Path(P["split"]).read_text())
            n_tr, n_te = len(split.get("train_base_paths", [])), len(split.get("test_base_paths", []))
            rep["split_counts"] = {"train": n_tr, "test": n_te}
            if (n_tr, n_te) != (2910, 821):
                problems.append(f"split has {n_tr} development / {n_te} held-out fields, expected 2910 / 821")
        if Path(P["external_manifest"]).is_file():
            rows = read_csv_rows(Path(P["external_manifest"]))
            need = {"image_path", "relative_path", "filename", "slide_id", "species_name", "label", "hyphae_status"}
            miss = need - set(rows[0]) if rows else need
            rep["manifest_rows"] = len(rows)
            if len(rows) != 1253 or miss:
                problems.append(f"manifest has {len(rows)} rows (expected 1253); missing columns {sorted(miss)}")

        ident = P.get("densenet_checkpoint_identity")
        if ident and Path(ident).is_file():
            mode = str(json.loads(Path(ident).read_text()).get("checkpoint_load_mode", ""))
            rep["densenet_frozen_load_mode"] = mode
            if "weights_only=False" in mode and not self.opt.get("allow_unsafe_densenet_load"):
                problems.append("the frozen DenseNet run loaded its checkpoint with weights_only=False; set "
                                "options.allow_unsafe_densenet_load to true in colab_config.json to match it")
        else:
            notes.append(f"DenseNet checkpoint_identity.json not found at {ident}; load mode not cross-checked")

        for spec in self.cfg.get("archives", []):
            _, path = spec.split("=", 1)
            for f in ("slide_predictions.csv", "image_predictions.csv"):
                if not (Path(path) / f).is_file():
                    msg = f"archive {spec.split('=')[0]}: {Path(path) / f} not found"
                    (notes if self.opt.get("allow_ungated") else problems).append(msg)

        for item in self.cfg.get("stage_inputs", []):
            for s in (item["src"] if isinstance(item["src"], list) else [item["src"]]):
                if not Path(s).exists():
                    (notes if item.get("optional") else problems).append(
                        f"stage_inputs source {'(optional) ' if item.get('optional') else ''}not found: {s}")

        gpu = nvidia_gpu_name()
        rep["gpu"] = gpu
        if gpu is None:
            problems.append("no NVIDIA GPU visible: Runtime > Change runtime type > T4 GPU")
        elif "T4" not in gpu:
            notes.append(f"GPU is {gpu}, not a Tesla T4: numerics are still gated by the C0 checks")
        lock = self.state / "partab_env.lock"
        if not lock.exists():
            problems.append(f"{lock} missing: run notebook cell 2 (colab_setup_env.py) first")
        self.local.mkdir(parents=True, exist_ok=True)
        rep["local_free_gb"] = round(shutil.disk_usage(self.local).free / 1e9, 1)
        if rep["local_free_gb"] < 40:
            notes.append(f"only {rep['local_free_gb']} GB free on local disk; staging needs about 30 GB")
        rep["problems"], rep["notes"] = problems, notes
        rep["status"] = "PASS" if not problems else "FAIL"
        cs.write_bytes_atomic(self.state / "preflight_report.json", cs.json_bytes(rep))
        for n in notes:
            log(f"preflight note: {n}")
        for p in problems:
            log(f"preflight PROBLEM: {p}")
        log(f"preflight: {rep['status']} ({len(problems)} problem(s), {len(notes)} note(s)); GPU {gpu}")
        return 0 if not problems else 1

    def stage_inputs(self) -> int:
        """
        Config entry: {"src": "<zip on Drive>" | ["<zip>", ...] | "<dir>", "dst": "<local dir>",
                       "expect": ["Candida albicans", "Candida glabrata"], "copy_first": true, "optional": false}
        """
        for item in self.cfg.get("stage_inputs", []):
            srcs = [Path(s) for s in (item["src"] if isinstance(item["src"], list) else [item["src"]])]
            dst = Path(item["dst"])
            if item.get("optional") and not all(s.exists() for s in srcs):
                log(f"stage_inputs: optional source {[str(s) for s in srcs]} absent; skipped")
                continue
            ident = [source_identity(s) for s in srcs]
            mk = dst / ".staged.json"
            if mk.exists():
                try:
                    rec = json.loads(mk.read_text())
                except ValueError:
                    rec = {}
                files = staged_files(dst)
                if (rec.get("sources") == ident and rec.get("n_files") == len(files)
                        and rec.get("bytes") == sum(f.stat().st_size for f in files) and check_expect(dst, item)):
                    log(f"stage_inputs: {dst} already staged ({len(files)} files)")
                    continue
            self.budget.require(self.timings.estimate(f"stage:{dst.name}", 1200.0), f"staging {dst}")
            t0 = time.time()
            dst.mkdir(parents=True, exist_ok=True)
            for src in srcs:
                if src.is_file() and zipfile.is_zipfile(src):
                    extract_zip(src, dst, copy_first=item.get("copy_first", True), scratch=self.local / "zip_scratch")
                elif src.is_dir():
                    copy_tree(src, dst)
                else:
                    raise FileNotFoundError(f"stage_inputs: {src} is neither a zip archive nor a directory")
                self.lock.beat()
            wrapper = flatten_single_wrapper(dst, item)
            if wrapper:
                log(f"stage_inputs: removed wrapper folder '{wrapper}' so that {dst} holds the class folders")
            if not check_expect(dst, item):
                tops = sorted(p.name for p in dst.iterdir() if p.name != ".staged.json")[:10]
                raise RuntimeError(f"stage_inputs: {dst} lacks {item.get('expect')}; top-level entries are {tops}. "
                                   "The archive must contain the two class folders (optionally inside one folder).")
            files = staged_files(dst)
            cs.write_bytes_atomic(mk, cs.json_bytes({"sources": ident, "n_files": len(files),
                                                     "bytes": sum(f.stat().st_size for f in files)}))
            self.timings.record(f"stage:{dst.name}", time.time() - t0)
            log(f"stage_inputs: {[s.name for s in srcs]} -> {dst} ({len(files)} files, {(time.time() - t0) / 60:.1f} min)")
        return 0

    def check_inputs_complete(self) -> Dict[str, Any]:
        P = self.P
        split = json.loads(Path(P["split"]).read_text())
        raw_root, ext_root = Path(P["internal_raw_root"]), Path(P["external_root"])
        bases = list(split["train_base_paths"]) + list(split["test_base_paths"])
        miss_i = [b for b in bases if resolve_internal(raw_root, b) is None]
        rows = read_csv_rows(Path(P["external_manifest"]))
        miss_e = [r["relative_path"] for r in rows if not (ext_root / r["relative_path"]).is_file()]
        out: Dict[str, Any] = {"internal_fields": len(bases), "internal_missing": len(miss_i),
                               "external_images": len(rows), "external_missing": len(miss_e)}
        npy = self.npy_root()
        if npy:
            test = list(split["test_base_paths"])
            miss_n = [b for b in test if not (Path(npy) / f"{b}.npy").is_file()]
            out.update(npy_root=npy, npy_missing=len(miss_n))
            if miss_n:
                raise RuntimeError(f"{len(miss_n)} held-out arrays missing under {npy} (first: {miss_n[:3]})")
        if miss_i or miss_e:
            raise RuntimeError(f"inputs incomplete: {len(miss_i)} internal fields missing under {raw_root} "
                               f"(first: {miss_i[:3]}), {len(miss_e)} external images missing under {ext_root} "
                               f"(first: {miss_e[:3]})")
        return out

    def migrate_tool(self, *args: str) -> List[str]:
        return self.py("migrate_part_ab.py", "--drive-root", str(self.root), "--config", str(self.cfg["_config_path"]),
                       *args)

    def migrate_snapshot(self) -> int:
        manifest = self.root / "migration" / "legacy_manifest.json"
        if manifest.exists():
            log("migrate_snapshot: legacy manifest present")
            return 0
        import migrate_part_ab as mig
        found = mig.legacy_artifacts(self.root, Path(self.P["geometry_spec"]))
        if not found:
            log("migrate_snapshot: no legacy outputs; nothing to snapshot")
            return 0
        log(f"migrate_snapshot: legacy outputs found ({found}); hashing before anything changes")
        return self.run("migrate_snapshot", self.migrate_tool("snapshot", "--geometry-spec", self.P["geometry_spec"]))

    def _sample_files(self) -> Dict[str, List[str]]:
        """Header check sample: the first file (sorted) of every slide in each domain."""
        P = self.P
        split = json.loads(Path(P["split"]).read_text())
        by: Dict[str, str] = {}
        for b in sorted(list(split["train_base_paths"]) + list(split["test_base_paths"])):
            by.setdefault(Path(b).name.rsplit("_", 1)[0], b)
        ints = [str(resolve_internal(Path(P["internal_raw_root"]), b)) for b in by.values()]
        ex: Dict[str, str] = {}
        for r in sorted(read_csv_rows(Path(P["external_manifest"])), key=lambda r: r["relative_path"]):
            ex.setdefault(r["slide_id"], r["relative_path"])
        exts = [str(Path(P["external_root"]) / rel) for rel in ex.values()]
        return {"internal": ints, "external": exts}

    def geometry(self) -> int:
        import geometry_harmonization as geo
        P, g = self.P, self.cfg.get("geometry", {})
        spec = Path(P["geometry_spec"])
        inputs = self.check_inputs_complete()
        log(f"geometry: inputs complete ({inputs})")
        exp = geo.expected_geometry(self.cfg)
        if spec.exists():
            try:
                rec = json.loads(spec.read_text())
            except ValueError as exc:
                raise RuntimeError(f"{spec} exists but is unreadable ({exc}); move it aside to rebuild") from exc
            if rec.get("schema") != geo.SPEC_SCHEMA:
                if not (self.root / "migration" / "legacy_manifest.json").exists():
                    raise RuntimeError(f"{spec} is a legacy spec and no migration snapshot exists; run the "
                                       "migrate_snapshot stage first")
                arch = spec.with_name(f"geometry_spec.legacy_{cs.sha256_file(spec)[:8]}.json")
                os.replace(spec, arch)
                log(f"geometry: legacy spec (T = {rec.get('external_matched_T_used')}) archived as {arch.name}")
            elif rec.get("geometry_digest") != exp["geometry_digest"]:
                raise RuntimeError(f"{spec} geometry digest {str(rec.get('geometry_digest'))[:12]} != the digest the "
                                   f"config implies ({exp['geometry_digest'][:12]}): the optics changed. Snapshot "
                                   "(migrate_part_ab.py snapshot), move the spec aside and rerun.")
            else:
                log(f"geometry: existing {spec.name} matches the optics (T = {rec['external_matched_T_used']}, "
                    f"m_ext = {rec['external_matched']['m_external']:.5f}, metadata "
                    f"{rec['external_matched']['metadata_status']})")
                self.mark("geometry", rec["geometry_digest"], [spec], {"inputs": inputs})
                return 0
        sample = self._sample_files()
        args = ["--config", str(self.cfg["_config_path"]), "--internal-files", *sample["internal"],
                "--external-files", *sample["external"], "--out", str(spec)]
        if g.get("expected_opencv"):
            args += ["--expected-opencv", str(g["expected_opencv"])]
        rc = self.run("geometry", self.py("geometry_harmonization.py", *args))
        if rc == 0 and spec.exists():
            self.mark("geometry", exp["geometry_digest"], [spec],
                      {"internal_files": sample["internal"], "external_files": sample["external"], "inputs": inputs})
        return rc

    def unit_checks(self) -> int:
        P, out = self.P, self.root / "unit_checks"
        args = ["--split", P["split"], "--internal-raw-root", P["internal_raw_root"],
                "--external-manifest", P["external_manifest"], "--external-root", P["external_root"],
                "--pinned-scripts", P["pinned_scripts"], "--geometry-spec", P["geometry_spec"],
                *self.model_args(), "--reuse-cache", "--out", str(out)]
        npy = self.npy_root()
        if npy:
            args += ["--npy-root", npy]
        code = {f: cs.sha256_file(self.code / f) for f in
                ("verify_unit_checks.py", "photometric_calibration.py", "geometry_harmonization.py", "optics.py")}
        dg = cs.sha256_bytes(cs.json_bytes({"args": args, "code": code,
                                            "geometry_digest": geometry_digest_of(Path(P["geometry_spec"]))}))
        report = out / "unit_check_report.json"
        if not self.marker_ok("unit_checks", dg):
            rc = self.run("unit_checks", self.py("verify_unit_checks.py", *args))
            if rc != 0 or not report.exists():
                return rc or 1
            self.mark("unit_checks", dg, [report])
        rep = json.loads(report.read_text())
        status = rep.get("overall_status")
        ok = status == "PASS" or (status == "CONDITIONAL" and self.opt.get("accept_conditional"))
        log(f"unit_checks: {status}" + (f"; conditions {rep.get('conditions')}" if status == "CONDITIONAL" else ""))
        if status == "CONDITIONAL" and not ok:
            log("unit_checks: CONDITIONAL is not accepted yet. Read the conditions (notebook cell 7) and set "
                "options.accept_conditional to true in colab_config.json if they are only disclosure items.")
        return 0 if ok else 1

    def migrate(self) -> int:
        mdir = self.root / "migration"
        if not (mdir / "legacy_manifest.json").exists():
            log("migrate: no legacy snapshot; nothing to migrate")
            return 0
        if (mdir / "applied.json").exists():
            log("migrate: already applied")
            return 0
        plan = mdir / "part_a_plan.json"
        if not plan.exists():
            rc = self.run("migrate", self.migrate_tool("plan", "--geometry-spec", self.P["geometry_spec"],
                                                       "--unit-check-dir", str(self.root / "unit_checks"),
                                                       "--split", self.P["split"]))
            if rc != 0:
                return rc
        sha = cs.sha256_file(plan)
        acc = (self.cfg.get("migration") or {}).get("accept_sha256")
        if acc != sha:
            log(f"migrate: review {plan} (INVARIANT / RECOMPUTE / REJECT per condition), then set "
                f"migration.accept_sha256 = \"{sha}\" in colab_config.json and rerun")
            return 1
        return self.run("migrate", self.migrate_tool("apply", "--accept", sha))

    def calibration(self) -> int:
        P, out = self.P, self.root / "part_a" / "calibration"
        cj = out / "calibration.json"
        gd = geometry_digest_of(Path(P["geometry_spec"]))
        if cj.exists():
            try:
                rec = json.loads(cj.read_text())
                if {"point", "bootstrap", "counts"} <= set(rec):
                    have = (rec.get("geometry") or {}).get("geometry_digest")
                    if have != gd:
                        raise RuntimeError(f"{cj} was fitted for geometry {str(have)[:12]}, the spec is {str(gd)[:12]}: "
                                           "run the migrate stage (it archives the legacy calibration)")
                    log("calibration: calibration.json present, complete and for this geometry")
                    return 0
            except ValueError:
                pass
            log("calibration: calibration.json incomplete (interrupted write); refitting from the cached statistics")
            cj.unlink()
        boot = ("import sys; sys.path.insert(0, %r); import pc_resilience; import photometric_calibration as pc; "
                "pc.main(sys.argv[1:])" % str(self.code))
        return self.run("calibration", [sys.executable, "-c", boot, "fit", "--split", P["split"],
                                        "--internal-raw-root", P["internal_raw_root"],
                                        "--external-manifest", P["external_manifest"],
                                        "--external-root", P["external_root"], "--pinned-scripts", P["pinned_scripts"],
                                        "--geometry-spec", P["geometry_spec"], "--out", str(out),
                                        "--workers", str(self.opt.get("calib_workers", 3))])

    def part_a(self) -> int:
        P, out = self.P, self.root / "part_a"
        summ = out / "part_a_summary.json"
        args = ["--split", P["split"], "--internal-raw-root", P["internal_raw_root"],
                "--geometry-spec", P["geometry_spec"], "--unit-check-dir", str(self.root / "unit_checks"),
                "--external-manifest", P["external_manifest"], "--external-root", P["external_root"],
                "--pinned-scripts", P["pinned_scripts"], "--work-dir", str(self.local),
                "--out", str(out), *self.model_args(), *self.budget_args(), *self.cond()]
        mm = self.root / "migration" / "part_a_plan.json"
        if (self.root / "migration" / "applied.json").exists() and mm.exists():
            args += ["--migration-manifest", str(mm)]
        dev = (self.cfg.get("geometry") or {}).get("protocol_deviation")
        if dev:
            args += ["--protocol-deviation", str(dev)]
        dg = content_digest(args, {"geometry_spec": Path(P["geometry_spec"]),
                                   "calibration": out / "calibration" / "calibration.json",
                                   "unit_check_report": self.root / "unit_checks" / "unit_check_report.json",
                                   **{f: self.code / f for f in ("run_part_a_internal_simulation.py", "optics.py",
                                                                 "geometry_harmonization.py", "photometric_calibration.py")}})
        if self.marker_ok("part_a", dg):
            log("part_a: complete (verified marker)")
            return 0
        rc = self.run("part_a", self.py("run_part_a_internal_simulation.py", *args))
        if rc == 0 and summ.exists() and json.loads(summ.read_text()).get("complete"):
            self.mark("part_a", dg, [summ])
        return rc

    def _b_args(self) -> List[str]:
        P = self.P
        args = ["--part-a-dir", str(self.root / "part_a"), "--unit-check-dir", str(self.root / "unit_checks"),
                "--manifest", P["external_manifest"], "--external-root", P["external_root"],
                "--pinned-scripts", P["pinned_scripts"], *self.model_args(), *self.cond(), *self.budget_args(),
                "--holm-scheme", self.opt.get("holm_scheme", "protocol_m12"), "--out", str(self.root / "part_b")]
        for spec in self.cfg.get("archives", []):
            args += ["--archive", spec]
        if self.opt.get("allow_ungated"):
            args.append("--allow-ungated")
        return args

    def _b_files(self) -> Dict[str, Optional[Path]]:
        pa = self.root / "part_a"
        return {"calibration": pa / "calibration" / "calibration.json", "part_a_summary": pa / "part_a_summary.json",
                "part_a_plan": pa / "part_a_plan_frozen.json", "run_part_b": self.code / "run_part_b_and_export.py"}

    def part_b_infer(self) -> int:
        args = self._b_args()
        dg = content_digest(["infer", *args], self._b_files())
        rec = self.root / "part_b" / "inference_record.json"
        if self.marker_ok("part_b_infer", dg):
            log("part_b_infer: complete (verified marker)")
            return 0
        rc = self.run("part_b_infer", self.py("run_part_b_and_export.py", "--stage", "infer", *args))
        if rc == 0 and rec.exists():
            self.mark("part_b_infer", dg, [rec])
        return rc

    def part_b_stats(self) -> int:
        args = self._b_args()
        dg = content_digest(["stats", *args], self._b_files())
        summ = self.root / "part_b" / "experimental_summary_for_manuscript.json"
        if self.marker_ok("part_b_stats", dg):
            log("part_b_stats: complete (verified marker)")
            return 0
        rc = self.run("part_b_stats", self.py("run_part_b_and_export.py", "--stage", "stats", *args))
        if rc == 0 and summ.exists():
            self.mark("part_b_stats", dg, [summ])
        return rc

    def export(self) -> int:   # always rebuilt (seconds); it verifies digest consistency itself
        return self.run("export", self.py("export_manuscript_artifacts.py", "--drive-root", str(self.root),
                                          "--config", str(self.cfg["_config_path"])))

    def go(self) -> int:
        self.lock.acquire()
        try:
            for name in STAGES:
                try:
                    self.lock.beat()
                    rc = getattr(self, name)()
                except cs.BudgetExhausted as exc:
                    log(f"PAUSED before {name}: {exc}")
                    return cs.EXIT_BUDGET
                except zipfile.BadZipFile as exc:
                    log(f"STOPPED in {name}: a zip archive failed its CRC/structure check ({exc}); "
                        "re-upload that archive to Drive and rerun")
                    return 1
                except Exception as exc:
                    log(f"STOPPED in {name}: {type(exc).__name__}: {exc}")
                    return 1
                if rc == cs.EXIT_BUDGET:
                    log(f"PAUSED in {name}: rerun this cell in a new session to resume")
                    return rc
                if rc != 0:
                    log(f"STOPPED: {name} failed with exit code {rc}; see its output above")
                    return rc
                if name == self.stop_after:
                    log(f"stopping after {name} as requested")
                    return 0
            log("ALL STAGES COMPLETE")
            return 0
        finally:
            self.lock.release()
            cs.sync_filesystems()


# ---- status and lock maintenance ---------------------------------------------------------------------------

def _count(paths: List[Path]) -> int:
    return sum(1 for p in paths if p.exists())


def status(cfg: Dict[str, Any]) -> None:
    root = Path(cfg["drive_root"])
    st = root / "state"
    print(f"Drive root: {root}")
    pf = st / "preflight_report.json"
    if pf.exists():
        rec = json.loads(pf.read_text())
        print(f"  preflight      {rec.get('status')} at {rec.get('utc')} ({len(rec.get('problems', []))} problems)")
    for item in cfg.get("stage_inputs", []):
        dst = Path(item["dst"])
        print(f"  stage_inputs   {dst}: {'staged this session' if (dst / '.staged.json').exists() else 'not staged in this session'}")
    gsp = Path(cfg["paths"]["geometry_spec"])
    gs = json.loads(gsp.read_text()) if gsp.exists() else {}
    print(f"  geometry       {'done' if (st / 'geometry.done.json').exists() else 'pending'}"
          f" (spec {gs.get('schema', 'absent')}; T = {gs.get('external_matched_T_used')}; "
          f"m_ext = {(gs.get('external_matched') or {}).get('m_external')})")
    mig = root / "migration"
    print(f"  migration      snapshot {'yes' if (mig / 'legacy_manifest.json').exists() else 'no'}; "
          f"plan {'yes' if (mig / 'part_a_plan.json').exists() else 'no'}; "
          f"applied {'yes' if (mig / 'applied.json').exists() else 'no'}")
    rep = root / "unit_checks" / "unit_check_report.json"
    print(f"  unit_checks    {json.loads(rep.read_text()).get('overall_status') if rep.exists() else 'pending'}")
    cal = root / "part_a" / "calibration"
    csvs = sorted(cal.glob("*.csv")) if cal.exists() else []
    rows = {p.name: max(0, sum(1 for _ in p.open()) - 1) for p in csvs}
    print(f"  calibration    {'done' if (cal / 'calibration.json').exists() else 'pending'}; cached rows {rows}")
    conds = root / "part_a" / "conditions_v2"
    n_done = _count(list(conds.glob("T*/DONE.json"))) if conds.exists() else 0
    n_sh = len(list((root / "part_a" / "shards_v2").rglob("*.done.json"))) if (root / "part_a" / "shards_v2").exists() else 0
    print(f"  part_a         {n_done} of 34 (+ supplementary) conditions assembled, {n_sh} shard markers"
          f"{'; summary written' if (root / 'part_a' / 'part_a_summary.json').exists() else ''}")
    pb = root / "part_b"
    n_pipe = _count(list((pb / "conditions").glob("*/DONE.json"))) if (pb / "conditions").exists() else 0
    n_bsh = len(list((pb / "shards").rglob("*.done.json"))) if (pb / "shards").exists() else 0
    n_memo = len(list((pb / "memo").rglob("*.done.json"))) if (pb / "memo").exists() else 0
    print(f"  part_b_infer   {n_pipe}/5 pipelines, {n_bsh}/5 inference shards")
    print(f"  part_b_stats   {n_memo} statistics tasks memoized"
          f"{'; summary written' if (pb / 'experimental_summary_for_manuscript.json').exists() else ''}")
    bundle = root / "manuscript_v2" / "partAB_manuscript_bundle.zip"
    print(f"  export         {'bundle ready: ' + str(bundle) if bundle.exists() else 'pending'}")
    tm = st / "stage_timings.json"
    if tm.exists():
        print(f"  timings (s)    {json.loads(tm.read_text())}")


def break_locks(cfg: Dict[str, Any]) -> None:
    root = Path(cfg["drive_root"])
    for p in (root / "state" / "DRIVER_LOCK.json", root / "part_a" / "RUN_LOCK.json", root / "part_b" / "RUN_LOCK.json"):
        if p.exists():
            rec = p.read_text()
            p.unlink()
            log(f"removed lock {p} ({rec.strip()[:120]})")


def main() -> None:
    p = argparse.ArgumentParser(description="Resumable Colab driver for Parts A and B.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stop-after", choices=STAGES, default=None)
    p.add_argument("--status", action="store_true", help="Print progress from Drive and exit")
    p.add_argument("--break-lock", action="store_true", help="Delete stale locks before running")
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    cfg["_config_path"] = str(a.config)
    if a.status:
        status(cfg)
        sys.exit(0)
    if a.break_lock:
        break_locks(cfg)
    sys.exit(Driver(cfg, a.stop_after).go())


if __name__ == "__main__":
    main()
