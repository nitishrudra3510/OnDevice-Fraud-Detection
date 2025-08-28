import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "2_Datasets"
ARTIFACTS = Path(__file__).resolve().parent / ".artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def extract_keystroke_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["typing_rate_keys_s"] = 1000.0 / (df["inter_key_delay_ms"].replace(0, 1e-3))
    df["hold_to_delay_ratio"] = df["keystroke_duration_ms"] / (df["inter_key_delay_ms"].replace(0, 1e-3))
    return df[[
        "user_id",
        "keystroke_duration_ms",
        "inter_key_delay_ms",
        "typing_rate_keys_s",
        "hold_to_delay_ratio",
        "label",
    ]]


def extract_touch_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["normalized_pressure"] = (df["tap_pressure"] - df["tap_pressure"].mean()) / (df["tap_pressure"].std() + 1e-6)
    df["speed_pressure_product"] = df["swipe_speed_px_s"] * df["tap_pressure"]
    return df[[
        "user_id",
        "swipe_speed_px_s",
        "tap_pressure",
        "normalized_pressure",
        "speed_pressure_product",
        "label",
    ]]


def extract_app_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sessions_per_hour"] = df["sessions_per_day"] / (df["app_usage_hours"].replace(0, 1e-3))
    df["nocturnal_intensity"] = df["night_usage_ratio"] * df["sessions_per_day"]
    return df[[
        "user_id",
        "app_usage_hours",
        "sessions_per_day",
        "night_usage_ratio",
        "sessions_per_hour",
        "nocturnal_intensity",
        "label",
    ]]


def extract_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mobility_index"] = df["avg_speed_kmh"] * np.sqrt(df["radius_of_gyration_km"].clip(lower=1e-6))
    df["stop_to_move_ratio"] = df["stop_rate_per_hr"] / (df["avg_speed_kmh"].replace(0, 1e-3))
    return df[[
        "user_id",
        "avg_speed_kmh",
        "radius_of_gyration_km",
        "stop_rate_per_hr",
        "mobility_index",
        "stop_to_move_ratio",
        "label",
    ]]


def main():
    ks = pd.read_csv(DATA_DIR / "Keystroke_Dynamics.csv")
    tg = pd.read_csv(DATA_DIR / "Touch_Gesture.csv")
    au = pd.read_csv(DATA_DIR / "App_Usage.csv")
    mv = pd.read_csv(DATA_DIR / "Movement_Patterns.csv")

    ks_feat = extract_keystroke_features(ks)
    tg_feat = extract_touch_features(tg)
    au_feat = extract_app_usage_features(au)
    mv_feat = extract_movement_features(mv)

    ks_feat.to_csv(ARTIFACTS / "keystroke_features.csv", index=False)
    tg_feat.to_csv(ARTIFACTS / "touch_features.csv", index=False)
    au_feat.to_csv(ARTIFACTS / "app_usage_features.csv", index=False)
    mv_feat.to_csv(ARTIFACTS / "movement_features.csv", index=False)

    print("Saved feature CSVs to:", ARTIFACTS)


if __name__ == "__main__":
    main()


