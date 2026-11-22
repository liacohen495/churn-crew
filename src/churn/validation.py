"""Validation gates between the crews. Each check returns a list of error strings;
an empty list means the check passed. `require` turns errors into a ValidationError."""
import json
from pathlib import Path

import pandas as pd

from .logging_utils import get_logger

log = get_logger("validation")


class ValidationError(Exception):
    """Raised when a gate fails. The Flow catches it and stops gracefully."""

    def __init__(self, gate: str, errors: list[str]):
        self.gate = gate
        self.errors = errors
        super().__init__(f"[{gate}] " + "; ".join(errors))


def check_files_exist(paths: list[Path]) -> list[str]:
    return [f"Missing artifact: {p.name}" for p in paths if not Path(p).exists() or Path(p).stat().st_size == 0]


def check_contract(df: pd.DataFrame, contract: dict) -> list[str]:
    """Confirm the cleaned dataset matches dataset_contract.json."""
    errors = []
    cols = contract["columns"]

    missing = [c for c in cols if c not in df.columns]
    extra = [c for c in df.columns if c not in cols]
    if missing:
        errors.append(f"Columns in contract but not in data: {missing}")
    if extra:
        errors.append(f"Columns in data but not in contract: {extra}")

    if len(df) < contract.get("min_rows", 0):
        errors.append(f"Only {len(df)} rows, contract requires >= {contract['min_rows']}")

    pk = contract.get("primary_key")
    if pk in df.columns and df[pk].duplicated().any():
        errors.append(f"Primary key '{pk}' has {int(df[pk].duplicated().sum())} duplicates")

    for c, spec in cols.items():
        if c not in df.columns:
            continue
        s = df[c]
        if not spec.get("nullable", True) and s.isna().any():
            errors.append(f"'{c}' has {int(s.isna().sum())} missing values")
        if spec["dtype"] == "number":
            if not pd.api.types.is_numeric_dtype(s):
                errors.append(f"'{c}' should be numeric but is {s.dtype}")
                continue
            if "min" in spec and (s < spec["min"]).any():
                errors.append(f"'{c}' has values below {spec['min']}")
            if "max" in spec and (s > spec["max"]).any():
                errors.append(f"'{c}' has values above {spec['max']}")
        if "allowed_values" in spec:
            bad = sorted(set(s.dropna().unique()) - set(spec["allowed_values"]))
            if bad:
                errors.append(f"'{c}' has values not allowed by contract: {bad[:5]}")

    tgt = contract["target"]["column"]
    if tgt in df.columns:
        bad = set(df[tgt].unique()) - set(contract["target"]["allowed_values"])
        if bad:
            errors.append(f"Target '{tgt}' has unexpected values: {sorted(bad)}")
    return errors


def check_required_features(features: pd.DataFrame, required: list[str]) -> list[str]:
    """Confirm all features the model needs exist (and are numeric) before modeling."""
    errors = []
    missing = [c for c in required if c not in features.columns]
    if missing:
        errors.append(f"{len(missing)} required features missing: {missing[:8]}")
    present = [c for c in required if c in features.columns]
    non_num = [c for c in present if not pd.api.types.is_numeric_dtype(features[c])]
    if non_num:
        errors.append(f"Non-numeric features: {non_num[:8]}")
    if present and features[present].isna().any().any():
        errors.append("Feature table contains missing values")
    return errors


def require(gate: str, errors: list[str]) -> None:
    if errors:
        for e in errors:
            log.error("[%s] %s", gate, e)
        raise ValidationError(gate, errors)
    log.info("[%s] passed", gate)


def load_contract(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
