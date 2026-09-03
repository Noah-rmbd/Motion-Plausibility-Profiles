DEFAULT_MAX_INACTIVITY_SECONDS = 6 * 60 * 60


def is_session_break(elapsed_time_s, max_inactivity_seconds):
    return (
        max_inactivity_seconds is not None
        and max_inactivity_seconds > 0
        and elapsed_time_s > max_inactivity_seconds
    )


def session_ranges(elapsed_times, max_inactivity_seconds):
    ranges = []
    start = 0
    for index, elapsed_time_s in enumerate(elapsed_times):
        if is_session_break(float(elapsed_time_s), max_inactivity_seconds):
            if start < index:
                ranges.append((start, index))
            start = index + 1
    if start < len(elapsed_times):
        ranges.append((start, len(elapsed_times)))
    return ranges


def anomalous_session_range(
    elapsed_times,
    anomaly_start_order,
    anomaly_end_order,
    max_inactivity_seconds,
):
    for start, end in session_ranges(elapsed_times, max_inactivity_seconds):
        if start <= anomaly_start_order and anomaly_end_order < end:
            return start, end
    raise ValueError(
        "The injected anomaly overlaps an inactivity boundary and cannot be "
        "represented as one synthetic trajectory"
    )
