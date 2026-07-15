import numpy as np
import json


def format_value(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


def dict_to_table(title, data):
    key_width = max(len(str(k)) for k in data.keys()) if data else 10
    val_width = max(len(format_value(v)) for v in data.values()) if data else 10
    width = key_width + val_width + 7

    lines = []
    lines.append(f"\n{title}")
    lines.append("-" * width)
    lines.append(f"| {'Field'.ljust(key_width)} | {'Value'.ljust(val_width)} |")
    lines.append("-" * width)

    for k, v in data.items():
        lines.append(
            f"| {str(k).ljust(key_width)} | {format_value(v).ljust(val_width)} |"
        )

    lines.append("-" * width)
    return "\n".join(lines)


def list_of_dicts_to_table(title, rows, columns):
    if not rows:
        return f"\n{title}\n(no rows)"

    widths = {}
    for col in columns:
        widths[col] = max(
            len(col),
            max(len(format_value(row.get(col, ""))) for row in rows),
        )

    total_width = sum(widths.values()) + 3 * len(columns) + 1

    lines = []
    lines.append(f"\n{title}")
    lines.append("-" * total_width)
    header = "| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |"
    lines.append(header)
    lines.append("-" * total_width)

    for row in rows:
        line = (
            "| "
            + " | ".join(
                format_value(row.get(col, "")).ljust(widths[col]) for col in columns
            )
            + " |"
        )
        lines.append(line)

    lines.append("-" * total_width)
    return "\n".join(lines)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
