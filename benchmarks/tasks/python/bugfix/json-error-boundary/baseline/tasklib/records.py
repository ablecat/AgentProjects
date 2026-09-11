import json


def decode_record(payload: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("record is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("record must be an object")
    return value
