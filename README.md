# Bike Rental Prediction — Neural Network from First Principles

## What it does

This project predicts hourly bike rental demand using a PyTorch neural network.

The project demonstrates a reproducible machine learning workflow, including data splitting, preprocessing, neural network development, model training, validation, baseline comparison, experimentation, and automated testing.

The dataset contains 6,000 hourly bike rental observations.

## How to run it

### 1. Clone the repository

```powershell
git clone <your-github-repository-url>
cd lab07-first-principles
```

### 2. Create the Conda environment

```powershell
conda env create -f environment.yml
conda activate lab07-first-principles
```

### 3. Run the training pipeline

```powershell
python train.py
```

The script trains the neural network, evaluates the model against baseline approaches, runs additional experiments, and saves reports in the `reports/` directory.

### 4. Run automated tests

```powershell
python -m pytest -v
```

### 5. Check code quality

```powershell
ruff check .
ruff format --check model.py prepare.py train.py tests
```

## Data and split

The project uses the instructor-provided `bikes_data.py` file to generate the bike rental dataset.

The dataset contains the following input features:

- Hour
- Temperature in Celsius
- Rainfall in millimetres
- Wind speed in kilometres per hour
- Weekend indicator
- Station type (categorical)

The target variable is `rentals`, representing the number of bike rentals.

The data is divided into three subsets:

| Dataset | Percentage | Records |
|---|---:|---:|
| Training | 60% | 3,600 |
| Validation | 20% | 1,200 |
| Testing | 20% | 1,200 |
| Total | 100% | 6,000 |

A fixed random seed of 2026 is used for reproducibility.

Numerical features are standardized using `StandardScaler`, while categorical features are encoded using `OneHotEncoder`.

The preprocessing transformations are fitted only on the training dataset to prevent data leakage. Validation and test datasets use the transformations learned from training data.

## Model and training

The project uses PyTorch to build a custom neural network.

### Architecture

- Input layer: 8 features
- First hidden layer: 64 neurons with ReLU activation
- Second hidden layer: 32 neurons with ReLU activation
- Output layer: 1 neuron for rental prediction

### Training configuration

| Parameter | Configuration |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Mean Squared Error (MSE) |
| Batch size | 64 |
| Epochs | 100 |
| Random seed | 2026 |

The training loop is implemented manually using PyTorch operations:

- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`

Training loss, validation loss, and validation MAE are recorded after each epoch.

The model weights corresponding to the lowest validation loss are restored before final evaluation.

## Results

The final neural network was evaluated against two baseline approaches using the held-out test dataset.

### Test-set performance

| Model | MAE | RMSE |
|---|---:|---:|
| Neural Network | 2.7992 | 4.1057 |
| Training Mean Baseline | 9.8484 | 13.7946 |
| Linear Regression | 8.7630 | 12.6408 |

The neural network achieved the lowest MAE and RMSE among the three approaches.

This indicates that the neural network captured relationships in the dataset that the simpler baseline models did not capture as effectively.

### Generated reports

The following outputs are saved in the `reports/` directory:

- `history.csv` — training and validation metrics for each epoch.
- `learning_curve.png` — visualization of training and validation loss.
- `metrics.json` — test-set performance metrics.
- `experiments.csv` — comparison of additional model experiments.

## Experiments

Two additional experiments were conducted by changing one parameter at a time.

| Experiment | Learning Rate | Hidden Size | Best Epoch | Validation MAE |
|---|---:|---:|---:|---:|
| Baseline | 0.001 | 64 | 100 | 2.9015 |
| Lower Learning Rate | 0.0005 | 64 | 100 | 5.6355 |
| Larger Hidden Size | 0.001 | 128 | 99 | 2.6398 |

### Observations

**1. Lower learning rate**

Reducing the learning rate from 0.001 to 0.0005 increased validation MAE within the same 100-epoch training period. The model may require more training time to converge at the lower learning rate.

**2. Larger hidden size**

Increasing the first hidden layer size from 64 to 128 improved validation MAE from approximately 2.9015 to 2.6398.

The larger model performed better on the validation dataset in these experiments.

The test dataset was not used to select the model configuration.

## Limitations

- The dataset is a generated instructional dataset and may not represent real-world bike rental demand.
- The model uses a limited number of weather, time, and station features.
- The experiments cover only a small number of hyperparameter configurations.
- The model has not been evaluated on external real-world bike rental data.
- Additional feature engineering and hyperparameter tuning may improve performance.

## Project structure

```text
lab07-first-principles/
│
├── bikes_data.py
├── prepare.py
├── model.py
├── train.py
│
├── tests/
│   └── test_pipeline.py
│
├── reports/
│   ├── history.csv
│   ├── learning_curve.png
│   ├── metrics.json
│   └── experiments.csv
│
├── requirements.txt
├── requirements-dev.txt
├── environment.yml
└── README.md
```

## Testing

The project includes automated tests covering:

- 60/20/20 data splitting and no overlap.
- Training-only scaler fitting.
- Dataset length, tensor shapes, and data types.
- Reproducible shuffled DataLoader behavior.
- Neural network output dimensions.
- Optimizer weight updates.
- Model memorization of a small dataset.
- Evaluation without modifying model weights.
- Restoration of the best validation model weights.

All 12 pytest test cases passed during development.