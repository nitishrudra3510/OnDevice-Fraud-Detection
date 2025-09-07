# Multi-Agent AI System for On-Device Fraud Detection

## Problem Statement

Fraud detection in digital transactions and user behavior has become increasingly critical as cyber threats evolve. Traditional server-side fraud detection systems face significant challenges:

- **Privacy Concerns**: Sensitive user data must be transmitted to external servers
- **Latency Issues**: Real-time detection requires immediate response
- **Scalability**: Centralized systems struggle with high-volume, distributed processing
- **Data Sensitivity**: Behavioral biometrics (keystroke dynamics, touch patterns) are highly personal

**On-device detection** addresses these challenges by processing data locally on user devices, ensuring privacy while enabling real-time fraud detection. This approach is particularly important for:

- Mobile banking applications
- E-commerce platforms
- Digital identity verification
- IoT device security

**Multi-agent AI** is ideally suited for this problem because fraud detection requires analyzing multiple behavioral signals simultaneously. Each agent can specialize in a specific domain (typing patterns, app usage, movement) while a central decision system fuses their outputs for comprehensive risk assessment.

## Why Multi-Agent?

This system employs five specialized agents, each with distinct responsibilities:

### 🤖 DataPreprocessingAgent
- **Role**: Data cleaning, normalization, and feature extraction
- **Responsibilities**: 
  - Handle missing values and outliers
  - Standardize data formats across different sources
  - Extract relevant features from raw behavioral data
  - Prepare data for downstream agents

### 🔍 AnomalyDetectionAgent
- **Role**: Machine learning-based anomaly detection
- **Responsibilities**:
  - Apply One-Class SVM for keystroke dynamics
  - Use Isolation Forest for touch gesture patterns
  - Implement Autoencoder for app usage behavior
  - Deploy CNN/LSTM for movement pattern analysis

### 📊 RiskScoringAgent
- **Role**: Risk assessment and probability scoring
- **Responsibilities**:
  - Convert anomaly scores to risk probabilities
  - Normalize scores across different agents
  - Apply domain-specific risk weighting
  - Generate confidence intervals

### ⚖️ DecisionAgent
- **Role**: Final fraud classification and decision making
- **Responsibilities**:
  - Fuse multi-agent scores using weighted combination
  - Apply threshold-based classification
  - Classify transactions as Safe, Suspicious, or Fraudulent
  - Handle edge cases and uncertainty

### 📝 ReportingAgent
- **Role**: Human-readable output and logging
- **Responsibilities**:
  - Generate natural language explanations
  - Create audit logs and reports
  - Provide interpretable risk assessments
  - Interface with external systems

## Project Description

### Workflow Overview

```
Raw Data → Preprocessing → Detection → Scoring → Decision → Reporting
    ↓           ↓            ↓         ↓         ↓         ↓
Behavioral   DataPreprocessing  AnomalyDetection  RiskScoring  DecisionAgent  ReportingAgent
Biometrics      Agent             Agent           Agent         Agent          Agent
```

### Data Flow

1. **Input**: Raw behavioral data (keystroke dynamics, touch gestures, app usage, movement patterns)
2. **Preprocessing**: DataPreprocessingAgent cleans and extracts features
3. **Detection**: AnomalyDetectionAgent applies ML models to detect anomalies
4. **Scoring**: RiskScoringAgent converts detections to risk scores
5. **Decision**: DecisionAgent fuses scores and makes final classification
6. **Reporting**: ReportingAgent generates human-readable output and logs

### Key Features

- **Privacy-Preserving**: All processing happens on-device
- **Real-Time**: Sub-second response times for fraud detection
- **Multi-Modal**: Combines multiple behavioral biometrics
- **Adaptive**: Configurable weights and thresholds
- **Interpretable**: Natural language explanations for decisions

## Tools, Libraries & Frameworks

### Core Technologies
- **Python 3.8+**: Primary programming language
- **scikit-learn**: Machine learning algorithms (One-Class SVM, Isolation Forest)
- **TensorFlow/Keras**: Deep learning models (Autoencoder, CNN, LSTM)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **Streamlit**: Web-based user interface

### Multi-Agent Frameworks (Optional)
- **LangChain**: For LLM integration and agent orchestration
- **CrewAI**: Multi-agent collaboration framework
- **AutoGen**: Conversational AI agent framework

### On-Device Optimization
- **TensorFlow Lite**: Model quantization and mobile deployment
- **ONNX**: Cross-platform model optimization
- **Core ML**: iOS deployment (optional)

## LLM Selection

### Ideal Choice: GPT-4
- **Reasoning**: Superior reasoning capabilities for complex fraud pattern analysis
- **Context Understanding**: Better comprehension of multi-modal behavioral data
- **Explanation Quality**: More coherent and accurate natural language explanations
- **Use Cases**: 
  - Generating detailed fraud risk explanations
  - Interpreting complex multi-agent decision patterns
  - Creating human-readable audit reports

### Free Alternative: GPT-3.5-turbo
- **Cost-Effective**: Significantly lower API costs
- **Good Performance**: Adequate for most fraud detection explanations
- **Wide Availability**: Easy integration with OpenAI API
- **Use Cases**: Basic risk explanations and standard reporting

### Open Source Option: Mistral-7B
- **Privacy**: Can be run locally without external API calls
- **Cost**: No ongoing API costs
- **Customization**: Can be fine-tuned for specific fraud detection domains
- **Use Cases**: Offline fraud detection systems

### Justification for LLM Usage

LLMs are essential for this project because:

1. **Interpretability**: Fraud detection decisions must be explainable to users and regulators
2. **Natural Language**: Converts technical ML outputs into human-readable reports
3. **Context Awareness**: Understands relationships between different behavioral signals
4. **Adaptive Communication**: Tailors explanations based on user expertise level
5. **Audit Trail**: Generates comprehensive logs for compliance and debugging

## Code & Deployment

### Setup Instructions

1. **Clone Repository**:
```bash
   git clone <repository-url>
   cd OnDevice-Fraud-Detection
```

2. **Create Virtual Environment**:
```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
pip install -r requirements.txt
```

4. **Generate Synthetic Data**:
   ```bash
python 2_Datasets/Synthetic_Data_Generator.py --num_users 100 --num_records 2000
```

5. **Run Feature Engineering**:
   ```bash
python 3_Feature_Engineering/feature_extraction.py
python 3_Feature_Engineering/preprocessing.py
```

6. **Train Models**:
   ```bash
python 4_Models/one_class_svm.py
python 4_Models/isolation_forest_model.py
python 4_Models/autoencoder_model.py
   python 4_Models/movement_cnn.py
   ```

7. **Start Multi-Agent System**:
   ```bash
python 4_Models/multi_agent_system.py
```

8. **Launch Streamlit App**:
   ```bash
   streamlit run 8_Streamlit_App/app.py
   ```

### Repository Structure

```
OnDevice-Fraud-Detection/
├── 1_Documentation/           # Project documentation
├── 2_Datasets/               # Synthetic data generation
├── 3_Feature_Engineering/    # Data preprocessing
├── 4_Models/                 # Multi-agent ML models
├── 5_OnDevice_Optimization/  # TensorFlow Lite conversion
├── 6_Evaluation/             # Performance metrics
├── 7_Demo_App/              # Mobile integration
├── 8_Streamlit_App/         # Web interface
├── agents/                   # Multi-agent system (NEW)
│   ├── data_preprocessing_agent.py
│   ├── anomaly_detection_agent.py
│   ├── risk_scoring_agent.py
│   ├── decision_agent.py
│   └── reporting_agent.py
├── deploy/                   # Deployment scripts
└── requirements.txt          # Dependencies
```

### Demo Deployment

#### Streamlit Cloud
1. Fork this repository
2. Connect to Streamlit Cloud
3. Deploy from GitHub repository
4. Access at: `https://your-app.streamlit.app`

#### Hugging Face Spaces
1. Create new Space on Hugging Face
2. Upload repository files
3. Configure `requirements.txt`
4. Access at: `https://huggingface.co/spaces/your-username/your-space`

## Future Work

### Real-Time Streaming Support
- **WebSocket Integration**: Real-time data streaming from mobile devices
- **Kafka/Redis**: Message queuing for high-throughput scenarios
- **Edge Computing**: Deploy agents on edge servers for reduced latency

### Blockchain Transaction Logs
- **Immutable Audit Trail**: Store fraud decisions on blockchain
- **Smart Contracts**: Automated fraud response mechanisms
- **Cross-Chain Analysis**: Detect patterns across multiple blockchains

### Reinforcement Learning Agents
- **Adaptive Thresholds**: RL agents that learn optimal decision thresholds
- **Dynamic Weighting**: Agents that adjust their influence based on performance
- **Adversarial Training**: RL agents trained against sophisticated fraud attempts

### Advanced Features
- **Federated Learning**: Train models across multiple devices without sharing data
- **Explainable AI**: Enhanced interpretability using SHAP and LIME
- **Multi-Modal Fusion**: Integration of additional data sources (camera, microphone)
- **Privacy-Preserving ML**: Homomorphic encryption for secure computation

## Contributing

This project is designed for educational and research purposes. Contributions are welcome in the following areas:

- New anomaly detection algorithms
- Enhanced multi-agent coordination
- Improved on-device optimization
- Additional behavioral biometrics
- Better LLM integration

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Samsung EnnovateX 2025 AI Challenge
- Open source machine learning community
- Behavioral biometrics research community# Coding-Ninjas-FraudShield-Multi-Agent-On-Device-Fraud-Detection-System
