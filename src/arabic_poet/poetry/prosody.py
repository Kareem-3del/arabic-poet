"""Automatic meter checking: CATT diacritization -> pyarud rule-based scansion.

- CATT (github.com/abjadai/catt, Apache-2.0): neural diacritizer, ONNX encoder-decoder,
  run on CPU via onnxruntime. Weights are downloaded once into models/catt/ (pinned URL,
  SHA-256 recorded) instead of site-packages.
- pyarud 1.0.0 (MIT): deterministic 'arud scansion of a diacritized sadr/'ajuz pair; returns
  meter, score, per-foot zihaf/'illa analysis and a qafiya analysis.

The model's own diacritics are stripped first (they are frequently wrong) and the text is
re-diacritized by CATT, so every system is scanned the same way. Diacritization errors
become meter errors; measured accuracy is in docs/EVALUATION.md (meter_gold_v1). This is an
automatic proxy, not expert prosodic judgement.
"""

from __future__ import annotations

import json
import re
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from arabic_poet.config import resolve_path
from arabic_poet.poetry.meter import MeterResult
from arabic_poet.poetry.normalization import split_hemistichs, strip_diacritics
from arabic_poet.utils.logging import get_logger
from arabic_poet.utils.reproducibility import sha256_file, utc_now, write_json

log = get_logger(__name__)

CATT_URL = "https://github.com/abjadai/catt/releases/download/v2/ed_model_onnx.zip"
CATT_DIR = "models/catt/ed_model"

#: pyarud meter keys -> project meter keys (poetry/meter.py)
PYARUD_TO_KEY = {
    "taweel": "tawil",
    "madeed": "madid",
    "baseet": "basit",
    "wafer": "wafir",
    "kamel": "kamil",
    "hazaj": "hazaj",
    "rajaz": "rajaz",
    "ramal": "ramal",
    "saree": "sari",
    "munsareh": "munsarih",
    "khafeef": "khafif",
    "mudhare": "mudari",
    "muqtadheb": "muqtadab",
    "mujtath": "mujtath",
    "mutakareb": "mutaqarib",
    "mutadarak": "mutadarik",
}

KEY_TO_PYARUD = {v: k for k, v in PYARUD_TO_KEY.items()}

#: A bayt counts as metrical only if pyarud marks it valid AND its score reaches this value.
#: Calibrated on meter_gold_v1 (see docs/EVALUATION.md); change only with a new calibration.
DEFAULT_THRESHOLD = 0.9
#: Near-exact scans only. Calibration v2 (scripts/meter_calibration_v2.py): accepts 37% of
#: real abyat and 4-5% of abyat with one word removed or replaced.
STRICT_THRESHOLD = 0.95
MIN_LETTERS = 6  # shortest real hemistichs (majzu' forms) have well over 6 letters
MAX_PLAUSIBLE_SCORE = 1.01  # pyarud rounds to ~1.002 for perfect scans


def ensure_catt() -> Path:
    """Download the CATT encoder-decoder ONNX weights into models/catt/ if missing."""
    target = resolve_path(CATT_DIR)
    if (target / "encoder.onnx").is_file() and (target / "decoder.onnx").is_file():
        return target
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target.parent / "ed_model_onnx.zip"
    log.info("Downloading CATT weights from %s", CATT_URL)
    with urllib.request.urlopen(CATT_URL, timeout=120) as resp, zip_path.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in ("encoder.onnx", "decoder.onnx"):
                with zf.open(member) as src, (target / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    write_json(
        target.parent / "manifest.json",
        {
            "source": CATT_URL,
            "license": "Apache-2.0 (abjadai/catt)",
            "downloaded_at": utc_now(),
            "zip_sha256": sha256_file(zip_path),
            "files": {n: sha256_file(target / n) for n in ("encoder.onnx", "decoder.onnx")},
        },
    )
    zip_path.unlink()
    if not (target / "encoder.onnx").is_file():
        raise RuntimeError(f"CATT archive did not contain the expected ONNX files ({CATT_URL})")
    return target


class CattPyarudValidator:
    """MeterValidator: per-bayt meter via CATT + pyarud."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        from catt_tashkeel.models import CATTEncoderDecoder
        from pyarud import ArudhProcessor

        d = ensure_catt()
        self._catt = CATTEncoderDecoder(
            encoder_path=str(d / "encoder.onnx"), decoder_path=str(d / "decoder.onnx")
        )
        self._arud = ArudhProcessor()
        self.threshold = threshold
        self.name = f"catt-ed-v2+pyarud-1.0.0@{threshold}"

    def diacritize(self, texts: list[str]) -> list[str]:
        clean = [strip_diacritics(t) for t in texts]
        return list(self._catt.do_tashkeel_batch(clean, verbose=False)) if clean else []

    def scan_pair(self, sadr: str, ajuz: str) -> dict:
        # Guard: pyarud can return scores > 1 for near-empty input (e.g. an 'ajuz of "ـ").
        if min(len(re.findall(r"[\u0621-\u064A]", h)) for h in (sadr, ajuz)) < MIN_LETTERS:
            return {"meter": None, "score": 0.0, "valid": False, "error": "hemistich_too_short"}
        try:
            r = self._arud.analyze_verse(sadr, ajuz)
        except Exception as exc:  # pyarud raises on unscannable input
            return {"meter": None, "score": 0.0, "valid": False, "error": type(exc).__name__}
        meter = PYARUD_TO_KEY.get(str(r.meter_key))
        score = float(r.score or 0.0)
        # pyarud's strict is_valid rejects most real abyat after automatic diacritization
        # (8/30 on meter_gold_v1); its best-match score separates verse from non-verse far
        # better, so the decision uses the score. strict_valid is kept for reference.
        if score > MAX_PLAUSIBLE_SCORE:  # anomalous scorer output: never count as metrical
            return {
                "meter": meter,
                "score": round(score, 4),
                "valid": False,
                "error": "score_anomaly",
            }
        valid = meter is not None and score >= self.threshold
        q = r.qafiyah or {}
        rawi = q.get("rawi") if isinstance(q, dict) else getattr(q, "rawi", None)
        return {
            "meter": meter,
            "score": round(score, 4),
            "valid": valid,
            "strict_valid": bool(r.is_valid),
            "rawi": rawi,
            "bahr_type": getattr(r, "bahr_type", None),
        }

    def scan_detail(self, sadr: str, ajuz: str, target_meter: str | None = None) -> dict:
        """Scan one diacritized bayt, optionally against a target meter (pyarud forced_meter),
        with per-hemistich scores and pyarud's qafiya analysis (rawi with wasl/ridf rules)."""
        if min(len(re.findall(r"[\u0621-\u064A]", h)) for h in (sadr, ajuz)) < MIN_LETTERS:
            return {"meter": None, "score": 0.0, "error": "hemistich_too_short"}
        try:
            r = self._arud.analyze_verse(
                sadr, ajuz, forced_meter=KEY_TO_PYARUD.get(target_meter or "")
            )
        except Exception as exc:  # pyarud raises on unscannable input
            return {"meter": None, "score": 0.0, "error": type(exc).__name__}
        score = float(r.score or 0.0)
        if score > MAX_PLAUSIBLE_SCORE:
            return {"meter": None, "score": 0.0, "error": "score_anomaly"}
        q = r.qafiyah
        return {
            "meter": PYARUD_TO_KEY.get(str(r.meter_key)),
            "score": round(score, 4),
            "sadr_score": round(float(r.sadr.score), 4) if r.sadr is not None else None,
            "ajuz_score": round(float(r.ajuz.score), 4) if r.ajuz is not None else None,
            "rawi": (q.rawi or None) if q is not None else None,
            "rawi_haraka": (q.rawi_haraka or None) if q is not None else None,
            "wasl": q.wasl if q is not None else None,
            "qafiyah_text": q.qafiyah_text if q is not None else None,
            "errors": list(r.errors)[:4],
        }

    def validate(self, abyat: list[str], requested_meter: str | None) -> MeterResult:
        pairs = []
        for bayt in abyat:
            parts = split_hemistichs(bayt)
            pairs.append((parts[0], parts[1]) if len(parts) >= 2 and parts[1].strip() else None)
        flat = [h for p in pairs if p for h in p]
        diac = iter(self.diacritize(flat))
        per_line: list[dict] = []
        for p in pairs:
            if p is None:
                per_line.append(
                    {"meter": None, "score": 0.0, "valid": False, "error": "missing_hemistich"}
                )
                continue
            a, b = next(diac), next(diac)
            per_line.append({**self.scan_pair(a, b), "sadr": a, "ajuz": b})
        if not per_line:
            return MeterResult(
                status="not_applicable",
                method=self.name,
                requested_meter=requested_meter,
                note="no abyat parsed",
            )
        valid = [x["meter"] for x in per_line if x["valid"]]
        scannable = [x for x in per_line if not x.get("error")]
        dominant = Counter(valid).most_common(1)[0][0] if valid else None
        n = len(per_line)
        strict = [x for x in per_line if x.get("meter") and x["score"] >= STRICT_THRESHOLD]
        distinct = {strip_diacritics(x["sadr"] + "|" + x["ajuz"]) for x in strict}
        return MeterResult(
            status="ok",
            method=self.name,
            requested_meter=requested_meter,
            detected_meter=dominant,
            consistency=round(sum(1 for m in valid if m == dominant) / n, 4) if dominant else 0.0,
            requested_match=round(sum(1 for m in valid if m == requested_meter) / n, 4)
            if requested_meter
            else None,
            per_line=per_line,
            metrical_rate=round(len(valid) / n, 4),
            # Over scannable abyat only, so layout failures are not counted twice.
            mean_score=round(sum(x["score"] for x in scannable) / len(scannable), 4)
            if scannable
            else None,
            scannable_rate=round(len(scannable) / n, 4),
            strict_rate=round(len(strict) / n, 4),
            strict_requested_match=round(
                sum(1 for x in strict if x["meter"] == requested_meter) / n, 4
            )
            if requested_meter
            else None,
            strict_distinct_rate=round(len(distinct) / n, 4),
        )


def calibrate(gold_path: str | Path, validator: CattPyarudValidator) -> dict:
    """Score the validator on meter_gold_v1 at several thresholds (reported, not tuned silently)."""
    rows = [
        json.loads(x)
        for x in resolve_path(gold_path).read_text(encoding="utf-8").splitlines()[1:]
        if x.strip()
    ]
    diac = validator.diacritize([h for r in rows for h in (r["sadr"], r["ajuz"])])
    scans = []
    for i, r in enumerate(rows):
        validator_threshold = validator.threshold
        validator.threshold = 0.0  # collect raw scores; thresholds applied below
        s = validator.scan_pair(diac[2 * i], diac[2 * i + 1])
        validator.threshold = validator_threshold
        scans.append(
            {
                **r,
                **{f"pred_{k}": v for k, v in s.items()},
                "sadr_diac": diac[2 * i],
                "ajuz_diac": diac[2 * i + 1],
            }
        )
    report: dict = {
        "n": {k: sum(1 for r in rows if r["kind"] == k) for k in ("metrical", "broken", "prose")},
        "thresholds": {},
    }
    for t in (0.7, 0.8, 0.85, 0.87, 0.9, 0.95):

        def ok(s: dict, t: float = t) -> bool:
            return s["pred_meter"] is not None and s["pred_score"] >= t

        met = [s for s in scans if s["kind"] == "metrical"]
        neg = [s for s in scans if s["kind"] != "metrical"]
        report["thresholds"][str(t)] = {
            "metrical_detected": round(sum(ok(s) for s in met) / len(met), 3),
            "meter_correct": round(
                sum(ok(s) and s["pred_meter"] == s["meter"] for s in met) / len(met), 3
            ),
            "false_positive_broken": round(
                sum(ok(s) for s in neg if s["kind"] == "broken")
                / max(1, sum(s["kind"] == "broken" for s in neg)),
                3,
            ),
            "false_positive_prose": round(
                sum(ok(s) for s in neg if s["kind"] == "prose")
                / max(1, sum(s["kind"] == "prose" for s in neg)),
                3,
            ),
        }
    report["items"] = scans
    return report
