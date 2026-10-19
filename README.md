<div align="center">

# Enterprise Machine Learning Systems
### Dual-Domain AI Architectures: Credit Scoring & Diagnostic Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-11B38A?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![OOP Architecture](https://img.shields.io/badge/Design-Clean_OOP-6C5CE7?style=for-the-badge)](#system-architecture)
[![Hardware Resilience](https://img.shields.io/badge/Hardware-Fault--Tolerant-00B894?style=for-the-badge)](#hardware--runtime-fault-tolerance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p align="center">
  A production-ready repository delivering dual enterprise ML engines built from the ground up on <b>Object-Oriented Design (SOLID)</b>, featuring <b>zero data leakage pipelines</b>, automated multi-model benchmarking, and a <b>Hardware/Device Error-Resilience Layer</b> (CUDA out-of-memory auto-fallbacks, headless display mitigations, multi-core dynamic allocation).
</p>

[Explore Credit Scoring](#1-credit-scoring-model) • [Explore Disease Prediction](#2-disease-prediction-from-medical-data) • [Device Recovery Matrix](#hardware--runtime-fault-tolerance) • [Quick Start](#quick-start)

</div>

---

## Table of Contents
1. [Core Design Philosophy](#-core-design-philosophy)
2. [Hardware & Runtime Fault-Tolerance](#-hardware--runtime-fault-tolerance)
3. [Project 1: Credit Scoring Model](#1-credit-scoring-model)
4. [Project 2: Disease Prediction Engine](#2-disease-prediction-from-medical-data)
5. [Comparative Benchmark Dashboard](#-benchmark-dashboard)
6. [Repository Architecture](#-repository-architecture)
7. [Quick Start & Setup](#-quick-start)

---

## Core Design Philosophy

Both engines adhere to strict production standards:
* **Strict OOP Scaffolding:** Inheritance via `abc.ABC`, runtime strategy polymorphism across classifiers, and tight data encapsulation to prevent pre-split leakage.
* **Redundant Data Networking:** Remote UCI dataset endpoints with automatic failovers to geographically distributed CDN mirrors.
* **Input Immunity:** Comprehensive CLI and inference guardrails blocking non-numeric data, unhandled missing parameters, and mathematically invalid values.

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                     HARDWARE RESILIENCE LAYER                          │
 │  (Checks CUDA Availability → Traps OOM → Allocates CPU Affinity)       │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
┌───────────────────────────────┐   ┌───────────────────────────────────┐
│     CREDIT SCORING ENGINE     │   │     DISEASE PREDICTION ENGINE     │
│  - Financial Imbalance Tuning │   │  - Multi-Path UCI Data Ingestion  │
│  - WOE & Information Value    │   │  - Strict Feature Imputation      │
│  - Calibrated Risk Deciles    │   │  - Dual-Class Diagnostic Scoring  │
└──────────────┬────────────────┘   └─────────────────┬─────────────────┘
               │                                      │
               └──────────────────┬───────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              UNIFIED OOP MODEL EVALUATOR & INFERENCE CLI                │
│             (Accuracy, Precision, Recall, F1, ROC-AUC)                  │
└─────────────────────────────────────────────────────────────────────────┘
