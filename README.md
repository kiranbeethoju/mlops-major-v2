# MLOps Linear Regression Pipeline

This repository contains a complete MLOps pipeline for Linear Regression using the California Housing dataset from sklearn.

## Project Overview

This project demonstrates a full MLOps workflow including:
- Model training with scikit-learn LinearRegression
- Unit testing
- Manual quantization of model parameters
- Docker containerization
- CI/CD pipeline with GitHub Actions

## Dataset & Model
- **Dataset**: California Housing dataset from sklearn.datasets
- **Model**: scikit-learn LinearRegression

## Project Structure
```
MLOPS_Major/
├── src/
│   ├── train.py          # Model training script
│   ├── quantize.py       # Manual quantization script
│   ├── predict.py        # Prediction script
│   └── utils.py          # Utility functions
├── tests/
│   └── test_train.py     # Unit tests
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
├── .github/
│   └── workflows/
│       └── ci.yml        # CI/CD workflow
└── README.md            # This file
```

## Comparison Table

| Component | Description | Status |
|-----------|-------------|--------|
| Repository Setup | README.md, .gitignore, requirements.txt | Complete |
| Model Training | LinearRegression training with R² score | Complete |
| Testing Pipeline | Unit tests for dataset, model, and performance | Complete |
| Manual Quantization | 8-bit quantization of model parameters | Complete |
| Dockerization | Dockerfile with predict.py verification | Complete |
| CI/CD Workflow | GitHub Actions with 3 jobs (updated to latest versions) | Complete |

## Usage

### Local Development
1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run training:
   ```bash
   python src/train.py
   ```

4. Run tests:
   ```bash
   pytest tests/
   ```

5. Run quantization:
   ```bash
   python src/quantize.py
   ```

### Docker
```bash
docker build -t mlops-linear-regression .
docker run mlops-linear-regression
```

## CI/CD Pipeline

The GitHub Actions workflow includes three jobs:
1. **test suite**: Runs pytest to validate code
2. **train and quantize**: Trains model and performs quantization
3. **build and test container**: Builds Docker image and tests prediction

## Author
- Email: G24Ai1115@iitj.ac.in
- Course: MLOps Major Assignment 