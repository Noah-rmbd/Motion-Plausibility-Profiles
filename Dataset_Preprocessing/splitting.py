import hashlib


DEFAULT_CALIBRATION_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 42
RESERVED_CALIBRATION_USERS = {
    ("inat", "98904"),
    ("gowalla", "16936"),
}


def user_split(
    dataset,
    user_id,
    calibration_fraction=DEFAULT_CALIBRATION_FRACTION,
    seed=DEFAULT_SPLIT_SEED,
):
    if (str(dataset), str(user_id)) in RESERVED_CALIBRATION_USERS:
        return "calibration"
    digest = hashlib.sha256(
        f"{seed}:{dataset}:{user_id}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "calibration" if value < calibration_fraction else "fit"
