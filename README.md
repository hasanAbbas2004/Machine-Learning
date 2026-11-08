# Loan Approval Prediction: Hybrid Parallel ML Pipeline in C++

A from-scratch C++ machine learning pipeline that predicts loan approvals from applicant data. It trains three models (logistic regression, random forest and a multilayer perceptron) **at the same time** using **MPI** (one model per process) and **OpenMP** (multithreading inside each model), then serves predictions and a risk score through an interactive CLI.

No ML libraries are used: the preprocessing and all three models are implemented by hand.

## How it works

```
loan_data.csv ──► loan_preprocessor ──► processed_data.csv ──► hybrid_ml_trainer ──► *.bin models ──► ml_predictor
  (24k rows)       clean, impute,                              3 MPI ranks: data is                  interactive
                   encode                                      scattered, each rank                  decision +
                                                               trains one model with                 risk score
                                                               OpenMP threads
```

| Stage | Program | Source |
|---|---|---|
| Preprocess | `loan_preprocessor` | `main.cpp`, `loan_data_preprocessor.cpp` |
| Train | `hybrid_ml_trainer` | `main_model.cpp`, `random_forest.cpp`, `mlp.cpp`, `logistic_regression.cpp` |
| Predict | `ml_predictor` | `prediction.cpp` + the three model files |

### Preprocessing
- Mean imputation for missing numeric values, defaults for missing categorical ones
- Correction of invalid (negative or zero) values
- Encoding: `employed` → 1 / `unemployed` → 0, `Approved` → 1 / `Rejected` → 0
- Optional `--sample <n>` preview and `--profile <file>` timing export

### Parallel training
Rank 0 loads the data and splits it across the three ranks with `MPI_Scatterv`. Each rank trains a different model on its share, using OpenMP threads inside the model:

| Rank | Model | Output |
|---|---|---|
| 0 | Random forest (100 trees, max depth 10) | `random_forest_model.bin` |
| 1 | MLP (5 → 10 → 5 → 1, 100 epochs) | `mlp_model.bin` |
| 2 | Logistic regression | `logistic_regression_model.bin` |

Training times are gathered back to rank 0 with `MPI_Gather` and printed as a summary.

### Prediction
`ml_predictor` asks for income, credit score, loan amount, debt-to-income ratio and employment status. It loads the saved models, prints an approval decision, and combines logistic regression and MLP outputs into a 0–100 risk score.

## Dataset

`loan_data.csv`: 24,000 loan applications.

| Income | Credit_Score | Loan_Amount | DTI_Ratio | Employment_Status | Approval |
|---|---|---|---|---|---|
| 26556 | 581 | 8314 | 79.26 | employed | Rejected |

## Build and run

Requirements: `g++` with OpenMP, plus an MPI implementation (Open MPI or MPICH) for the trainer. The included `.devcontainer` sets up an Ubuntu C++ environment.

```bash
# 1. Preprocess
g++ -O2 -fopenmp main.cpp loan_data_preprocessor.cpp -o loan_preprocessor
./loan_preprocessor loan_data.csv processed_data.csv --sample 5

# 2. Train (needs exactly 3 MPI processes)
mpic++ -O2 -fopenmp main_model.cpp random_forest.cpp mlp.cpp logistic_regression.cpp -o hybrid_ml_trainer
mpirun -np 3 ./hybrid_ml_trainer processed_data.csv

# 3. Predict with the saved models
g++ -O2 -fopenmp prediction.cpp random_forest.cpp mlp.cpp logistic_regression.cpp -o ml_predictor
./ml_predictor
```

OpenMP uses 5 threads per process by default. To change that, compile with `-DOMP_NUM_THREADS=<n>`.

## Project structure

```
include/                     headers: models, preprocessing, CSV parser, OpenMP config
main.cpp                     preprocessor CLI
loan_data_preprocessor.cpp   cleaning, imputation, encoding
main_model.cpp               MPI training driver
random_forest.cpp            random forest
mlp.cpp                      multilayer perceptron
logistic_regression.cpp      logistic regression
prediction.cpp               interactive predictor and risk score
loan_data.csv                raw dataset
```

## Roadmap
- Held-out test split with accuracy, precision, recall, F1 and confusion matrix (interface declared in `include/evaluate.h`)
- Replace the outdated `Makefile` with targets for the three programs above

## Tech
C++ · MPI · OpenMP · hand-written ML, no external ML libraries · [fast-cpp-csv-parser](https://github.com/ben-strasser/fast-cpp-csv-parser) for CSV loading
