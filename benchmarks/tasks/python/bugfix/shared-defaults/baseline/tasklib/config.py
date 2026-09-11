from tasklib.defaults import DEFAULT_LABELS, DEFAULT_RETRIES


def new_config() -> dict[str, object]:
    return {"labels": list(DEFAULT_LABELS), "retries": DEFAULT_RETRIES}
