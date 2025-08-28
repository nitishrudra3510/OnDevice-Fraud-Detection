# System Design

## Architecture

Sensors → Local Agents → Decision Agent → Alert/Action

```mermaid
flowchart LR
    S1["Sensors: Keystroke, Touch, App Usage, GPS/IMU"] --> A1["Typing Agent (One-Class SVM)"]
    S1 --> A2["Gesture Agent (Isolation Forest)"]
    S1 --> A3["App Usage Agent (Autoencoder)"]
    S1 --> A4["Movement Agent (LSTM)"]
    A1 --> D["Decision Agent (Fusion + Thresholds)"]
    A2 --> D
    A3 --> D
    A4 --> D
    D -->|Normal| U1["UI: Normal User"]
    D -->|Suspicious| U2["UI: Lock Device / Step-up Auth"]
```

## Data Flow

1. Collect raw signals (or load CSVs). 2. Extract features. 3. Preprocess. 4. Per-agent scoring. 5. Fuse scores. 6. Trigger UI/action.

## Decision Fusion

- Normalize agent anomaly scores to [0,1].
- Weighted average or learned logistic regression.
- Thresholds tuned for target precision/recall.

## Security & Privacy

- On-device inference; no raw behavioral data leaves device.
- Model updates via secure channel or federated averaging (future work).

