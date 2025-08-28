import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_keystroke(num_users: int, num_records: int, rng: np.random.Generator) -> pd.DataFrame:
    users = rng.integers(1, num_users + 1, size=num_records)
    base_duration = rng.normal(120, 20, size=num_records)  # ms
    base_inter_delay = rng.normal(150, 30, size=num_records)
    anomaly_flag = rng.binomial(1, 0.1, size=num_records)
    durations = base_duration * (1 + anomaly_flag * rng.normal(0.8, 0.2, size=num_records))
    inter_delays = base_inter_delay * (1 + anomaly_flag * rng.normal(0.7, 0.25, size=num_records))
    return pd.DataFrame({
        "user_id": users,
        "keystroke_duration_ms": np.clip(durations, 20, 800),
        "inter_key_delay_ms": np.clip(inter_delays, 20, 1000),
        "label": 1 - anomaly_flag,  # 1=normal, 0=anomaly
    })


def generate_touch(num_users: int, num_records: int, rng: np.random.Generator) -> pd.DataFrame:
    users = rng.integers(1, num_users + 1, size=num_records)
    swipe_speed = rng.normal(500, 120, size=num_records)  # px/s
    tap_pressure = rng.normal(0.5, 0.1, size=num_records)  # 0..1
    anomaly_flag = rng.binomial(1, 0.1, size=num_records)
    swipe_speed = swipe_speed * (1 + anomaly_flag * rng.normal(0.9, 0.3, size=num_records))
    tap_pressure = tap_pressure * (1 + anomaly_flag * rng.normal(0.6, 0.3, size=num_records))
    return pd.DataFrame({
        "user_id": users,
        "swipe_speed_px_s": np.clip(swipe_speed, 50, 2000),
        "tap_pressure": np.clip(tap_pressure, 0.05, 1.5),
        "label": 1 - anomaly_flag,
    })


def generate_app_usage(num_users: int, num_records: int, rng: np.random.Generator) -> pd.DataFrame:
    users = rng.integers(1, num_users + 1, size=num_records)
    usage_hours = np.clip(rng.normal(2, 1, size=num_records), 0, 12)
    sessions_per_day = np.clip(rng.normal(30, 10, size=num_records), 1, 300)
    night_usage_ratio = np.clip(rng.beta(2, 8, size=num_records), 0, 1)
    anomaly_flag = rng.binomial(1, 0.1, size=num_records)
    usage_hours = usage_hours * (1 + anomaly_flag * rng.normal(1.2, 0.4, size=num_records))
    sessions_per_day = sessions_per_day * (1 + anomaly_flag * rng.normal(1.5, 0.5, size=num_records))
    night_usage_ratio = np.clip(night_usage_ratio + anomaly_flag * rng.normal(0.3, 0.1, size=num_records), 0, 1)
    return pd.DataFrame({
        "user_id": users,
        "app_usage_hours": usage_hours,
        "sessions_per_day": sessions_per_day,
        "night_usage_ratio": night_usage_ratio,
        "label": 1 - anomaly_flag,
    })


def generate_movement(num_users: int, num_records: int, rng: np.random.Generator) -> pd.DataFrame:
    users = rng.integers(1, num_users + 1, size=num_records)
    avg_speed_kmh = np.clip(rng.normal(5, 2, size=num_records), 0, 120)
    radius_of_gyration_km = np.clip(rng.normal(10, 5, size=num_records), 0, 200)
    stop_rate_per_hr = np.clip(rng.normal(8, 3, size=num_records), 0, 60)
    anomaly_flag = rng.binomial(1, 0.1, size=num_records)
    avg_speed_kmh = avg_speed_kmh * (1 + anomaly_flag * rng.normal(1.0, 0.4, size=num_records))
    radius_of_gyration_km = radius_of_gyration_km * (1 + anomaly_flag * rng.normal(1.2, 0.5, size=num_records))
    stop_rate_per_hr = stop_rate_per_hr * (1 + anomaly_flag * rng.normal(0.8, 0.4, size=num_records))
    return pd.DataFrame({
        "user_id": users,
        "avg_speed_kmh": avg_speed_kmh,
        "radius_of_gyration_km": radius_of_gyration_km,
        "stop_rate_per_hr": stop_rate_per_hr,
        "label": 1 - anomaly_flag,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_users", type=int, default=50)
    parser.add_argument("--num_records", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    base = Path(__file__).resolve().parent

    ks = generate_keystroke(args.num_users, args.num_records, rng)
    tg = generate_touch(args.num_users, args.num_records, rng)
    au = generate_app_usage(args.num_users, args.num_records, rng)
    mv = generate_movement(args.num_users, args.num_records, rng)

    ks.to_csv(base / "Keystroke_Dynamics.csv", index=False)
    tg.to_csv(base / "Touch_Gesture.csv", index=False)
    au.to_csv(base / "App_Usage.csv", index=False)
    mv.to_csv(base / "Movement_Patterns.csv", index=False)

    print("Saved CSVs to:", base)


if __name__ == "__main__":
    main()


