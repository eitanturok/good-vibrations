"""The universal sample id, "{position_id:06d}-{speaker}" (e.g. "000012-3"): zero-padded so it sorts as a
string. Built from metadata, so old samples (dirs named by a counter, "000123") have one too."""


def sample_name(position_id, speaker) -> str:
    assert 0 <= int(speaker) < 10, speaker  # one digit, or the string sort breaks
    return f"{int(position_id):06d}-{int(speaker)}"


def meta_sample_name(meta: dict) -> str | None:
    # experiment-25 calls the position output_id
    p, s = meta.get("position_id", meta.get("output_id")), meta.get("speaker")
    return None if p is None or s is None else sample_name(p, s)
