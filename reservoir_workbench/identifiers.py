"""Generic normalization; original field-specific aliases are omitted."""
import re

def canonicalize_well_name(value: object) -> str:
    text = "" if value is None else str(value).strip().upper()
    text = re.sub(r"(?<=\d)\.0+$", "", text)
    text = re.sub(r"[^0-9A-Z\u4e00-\u9fff]+", "", text)
    return text
