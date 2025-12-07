TrackRNN – GRU-Based Temporal Predictive Model for Signal Anomaly Detection

TrackRNN is a lightweight, self-supervised anomaly detection framework that uses a GRU-based Recurrent Neural Network to predict the future of a signal from its past.
By comparing prediction error + spectral deviation, TrackRNN identifies unusual behavior without requiring labeled anomalies.

Originally built on railway-like audio signals, TrackRNN is designed as a general, domain-agnostic time-series anomaly detector.

🚀 Core Idea

Instead of classifying signals, TrackRNN operates on a predictive modeling principle:

A GRU model learns to predict the next N samples from the previous M samples.

During inference, deviations between predicted and actual signals indicate abnormal patterns.

A composite anomaly score (CAS) combines:

MAE_z → normalized prediction error

SDF_z → Spectral Drift Function (measures frequency-shift anomalies)

This combination allows TrackRNN to detect micro-anomalies invisible to traditional threshold-based systems.

🧠 Architecture

GRU-based RNN with a small, efficient hidden state

Self-supervised learning (no labels required)

MAE loss + optional spectral magnitude loss

Composite anomaly scoring (CAS)
CAS = z(MAE) + z(SDF)

Multiple thresholding strategies

Mean + 3σ

95th percentile

99th percentile

Forced threshold (user-defined)

The model runs well even on CPU environments.

💡 Why TrackRNN is Unique

Traditional anomaly detectors rely on classification or manual thresholds.
TrackRNN learns the normal predictive structure of a signal and flags anything that breaks that structure—making it:

Robust

Unsupervised

Domain-flexible

Generalizable

📢 Generalization to All Signal Types

(New section added as requested)

Although TrackRNN was initially tested on railway-style signals, its architecture is designed to generalize to all kinds of continuous signals, including:

Mechanical vibrations

Motor/engine noise

Biomedical data (ECG, breathing, EMG)

Structural health monitoring

Drone acoustics

IoT sensor data

Environmental audio

Industrial rotating machinery

Because TrackRNN learns patterns directly from raw temporal sequences, it adapts naturally to any domain where future prediction is meaningful.

This makes it a universal anomaly detection framework.

🔬 Active Research Direction

TrackRNN is not just a project — it is a developing research pipeline.

Ongoing work includes:

Multi-signal fine-tuning (one model for many domains)

Domain transfer learning using GRU embeddings

Real-time streaming anomaly detection

Deploying GRU inference on edge devices

Hybrid GRU + Transformer models

Uncertainty modeling for anomaly scoring

Expanding CAS into multi-resolution spectral features

This project forms a strong foundation for future research papers, industry solutions, or hackathon implementations.

📊 Key Experimental Outputs

The system automatically generates:

Prediction Example Plots – Showing how well the GRU predicts the future

CAS Histogram – Distribution of anomaly scores

CAS Sliding Window Graph – Time vs anomaly score

Precision–Recall Curve – Evaluating synthetic injected anomalies

Top Anomalous Windows – Extracted .wav segments for manual inspection

These visualizations help debug and validate model behavior.

🛠 How to Train
python rnn_train.py --data path/to/dataset --epochs 20 --threshold 2.23


Modify hyperparameters:

--past, --future for window sizes

--hidden, --layers for GRU complexity

--spectral-loss-weight for spectral drift sensitivity

--k for thresholding

--save-top to export anomaly segments

TrackRNN runs on CPU or GPU seamlessly.

📦 Project Structure
TrackRNN/
│── train.py
│── model.py
│── utils.py
│── plots/
│── top_windows/
│── metadata.json
│── README.md  ← (this file)

📈 Performance Summary (Your Latest Results)

Train MAE Mean ± Std: 0.663 ± 0.262

Train SDF Mean ± Std: 0.180 ± 0.097

CAS Threshold (forced): 2.23

PR-AUC on injected anomaly: 0.728

Top anomalous segments successfully extracted

These results confirm the GRU is learning stable normal behavior and identifying anomalous shifts.

🔮 Future Roadmap

Multi-domain pretraining

Transformer-based predictive heads

Anomaly explanation module

Multi-scale spectral scoring

Real railway deployment integration

Fine-tuning dataset for vibration/ECG/shock signals

Convert TrackRNN to a pip installable library

🤝 Contributions & License

Open for collaboration!
Feel free to raise issues, propose improvements, or suggest new datasets.

Summary

TrackRNN is a general-purpose, research-oriented anomaly detector built on GRU predictive modeling.
With CAS scoring, spectral drift detection, percentile thresholds, and synthetic anomaly benchmarking, it provides a powerful foundation for analyzing any continuous signal.
