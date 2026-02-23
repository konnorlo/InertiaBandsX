from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

from training.core.config import TrainingConfig
from training.core.constants import AUDIO_FILE_EXTENSIONS
from training.core.storage import DatasetStorage

DEFAULT_ALLOW_LICENSES = (
    "cc0",
    "cc-by",
    "cc by",
)


def _normalize_license(text: str) -> str:
    lowered = text.strip().lower()
    lowered = lowered.replace("_", "-")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _license_allowed(license_text: str, allow_patterns: list[str]) -> bool:
    norm = _normalize_license(license_text)
    return any(pattern in norm for pattern in allow_patterns)


def _find_column(example: dict[str, Any], candidates: Iterable[str]) -> str | None:
    lowered_map = {key.lower(): key for key in example.keys()}
    for candidate in candidates:
        if candidate.lower() in lowered_map:
            return lowered_map[candidate.lower()]
    for key in example.keys():
        key_l = key.lower()
        for candidate in candidates:
            if candidate.lower() in key_l:
                return key
    return None


def _select_columns(example: dict[str, Any], explicit_audio_column: str | None) -> dict[str, str | None]:
    audio_column = explicit_audio_column or _find_column(example, ["audio", "sound", "wav", "mp3"])
    license_column = _find_column(example, ["license", "licence"])
    source_column = _find_column(example, ["url", "source_url", "source", "file", "path"])
    creator_column = _find_column(example, ["author", "creator", "artist", "uploader"])
    genre_column = _find_column(example, ["genre", "tag", "tags"])

    return {
        "audio": audio_column,
        "license": license_column,
        "source": source_column,
        "creator": creator_column,
        "genre": genre_column,
    }


def _materialize_audio(
    record: dict[str, Any],
    audio_column: str,
    output_dir: Path,
    dataset_id: str,
    split: str,
    index: int,
) -> Path | None:
    payload = record.get(audio_column)
    if payload is None:
        return None

    if isinstance(payload, str):
        path = Path(payload).expanduser()
        if path.exists() and path.suffix.lower() in AUDIO_FILE_EXTENSIONS:
            return path.resolve()
        return None

    if not isinstance(payload, dict):
        return None

    possible_path = payload.get("path")
    if possible_path is not None:
        p = Path(str(possible_path)).expanduser()
        if p.exists():
            return p.resolve()

    payload_bytes = payload.get("bytes")
    if payload_bytes is not None:
        try:
            decoded, decoded_sr = sf.read(io.BytesIO(payload_bytes), always_2d=True, dtype="float32")
            key_material = f"{dataset_id}|{split}|{index}|bytes|{decoded.shape[0]}|{decoded_sr}"
            digest = hashlib.sha1(key_material.encode("utf-8")).hexdigest()[:16]
            out_path = output_dir / f"{dataset_id.replace('/', '__')}__{split}__{digest}.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), decoded, int(decoded_sr), subtype="PCM_16")
            return out_path.resolve()
        except Exception:
            return None

    array = payload.get("array")
    sample_rate = payload.get("sampling_rate")
    if array is None or sample_rate is None:
        return None

    waveform = np.asarray(array, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform[:, np.newaxis]
    elif waveform.ndim > 2:
        waveform = waveform.reshape(waveform.shape[0], -1)

    key_material = f"{dataset_id}|{split}|{index}|{waveform.shape[0]}|{sample_rate}"
    digest = hashlib.sha1(key_material.encode("utf-8")).hexdigest()[:16]
    out_path = output_dir / f"{dataset_id.replace('/', '__')}__{split}__{digest}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), waveform, int(sample_rate), subtype="PCM_16")
    return out_path.resolve()


def _record_to_jsonable(record: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in record.items():
        try:
            json.dumps(value)
            safe[key] = value
        except TypeError:
            safe[key] = str(value)
    return safe


def fetch_dataset(
    dataset_id: str,
    dataset_config: str | None,
    split: str,
    max_items: int,
    output_audio_dir: Path,
    allow_licenses: list[str],
    audio_column: str | None,
    streaming: bool,
    trust_remote_code: bool,
) -> dict[str, int]:
    config = TrainingConfig.default()
    storage = DatasetStorage(config)

    kwargs: dict[str, Any] = {"split": split}
    if dataset_config:
        kwargs["name"] = dataset_config

    ds = load_dataset(dataset_id, streaming=streaming, trust_remote_code=trust_remote_code, **kwargs)
    if "audio" in ds.features:
        ds = ds.cast_column("audio", Audio(decode=False))

    if streaming:
        iterator_preview = iter(ds)
        first_record = dict(next(iterator_preview))
    else:
        first_record = dict(ds[0])
    selected = _select_columns(first_record, audio_column)
    if selected["audio"] is None:
        raise ValueError("could not detect audio column; pass --audio-column explicitly")

    counters = {
        "seen": 0,
        "saved": 0,
        "skipped_license": 0,
        "skipped_audio": 0,
    }

    iterator = enumerate(ds)
    if max_items > 0:
        iterator = ((idx, row) for idx, row in iterator if idx < max_items)

    pbar = tqdm(iterator, desc=f"fetch {dataset_id}:{split}", unit="item")
    for index, row in pbar:
        counters["seen"] += 1
        record = dict(row)

        license_text = ""
        if selected["license"] is not None:
            license_text = str(record.get(selected["license"], "") or "").strip()
        if allow_licenses and not _license_allowed(license_text, allow_licenses):
            counters["skipped_license"] += 1
            continue

        audio_path = _materialize_audio(
            record=record,
            audio_column=selected["audio"],
            output_dir=output_audio_dir,
            dataset_id=dataset_id,
            split=split,
            index=index,
        )
        if audio_path is None:
            counters["skipped_audio"] += 1
            continue

        genre = ""
        if selected["genre"] is not None:
            genre_val = record.get(selected["genre"], "")
            genre = ", ".join(genre_val) if isinstance(genre_val, list) else str(genre_val)

        audio_id = storage.upsert_audio_file(audio_path=audio_path, source=f"hf:{dataset_id}", genre=genre[:256])
        source_url = str(record.get(selected["source"], "") or "") if selected["source"] else ""
        creator = str(record.get(selected["creator"], "") or "") if selected["creator"] else ""
        tag_values: list[str] = []
        if selected["genre"] is not None:
            raw_tags = record.get(selected["genre"], [])
            if isinstance(raw_tags, list):
                tag_values = [str(item) for item in raw_tags]
            elif raw_tags:
                tag_values = [str(raw_tags)]

        storage.upsert_audio_metadata(
            audio_id=audio_id,
            dataset_id=dataset_id,
            dataset_config=dataset_config or "",
            split=split,
            source_url=source_url[:1024],
            license_text=license_text[:256],
            creator=creator[:256],
            tags=tag_values[:32],
            raw=_record_to_jsonable(record),
        )
        counters["saved"] += 1
        pbar.set_postfix(saved=counters["saved"], skipped_license=counters["skipped_license"])

    return counters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public audio datasets into training DB with license filtering.")
    parser.add_argument("--dataset-id", required=True, help="Hugging Face dataset id, e.g. philgzl/fsd50k")
    parser.add_argument("--dataset-config", default="", help="Optional dataset config/name")
    parser.add_argument("--split", default="train", help="Dataset split to import")
    parser.add_argument("--max-items", type=int, default=200, help="Max items to import; 0 = all")
    parser.add_argument("--audio-column", default="", help="Explicit audio column name if auto-detect fails")
    parser.add_argument(
        "--allow-license",
        action="append",
        default=[],
        help="Allowed license substring (repeatable). Defaults: cc0, cc-by",
    )
    parser.add_argument(
        "--output-audio-dir",
        default="training/data/audio_cache",
        help="Directory where decoded/copied audio files are stored",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use HF streaming mode to avoid downloading entire parquet shards up front",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow dataset repositories with custom loading scripts to execute",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allow = [_normalize_license(item) for item in (args.allow_license or [])]
    if not allow:
        allow = list(DEFAULT_ALLOW_LICENSES)

    config = TrainingConfig.default()
    output_dir = (config.repo_root / args.output_audio_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    counters = fetch_dataset(
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config or None,
        split=args.split,
        max_items=int(args.max_items),
        output_audio_dir=output_dir,
        allow_licenses=allow,
        audio_column=args.audio_column or None,
        streaming=bool(args.streaming),
        trust_remote_code=bool(args.trust_remote_code),
    )
    print(json.dumps(counters, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
