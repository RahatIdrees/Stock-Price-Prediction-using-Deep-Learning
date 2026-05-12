# Stock-Price-Prediction-using-Deep-Learning


This project focuses on predicting stock closing prices using deep learning techniques. Three sequential models were implemented and evaluated:

- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

The models were trained on historical stock market data containing:

- Open Price
- High Price
- Low Price
- Close Price
- Volume

The project demonstrates how deep learning can capture temporal dependencies in financial time series data.

---

#  Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab

---

#  Dataset Information

The dataset is a multivariate time series stock market dataset containing:

- Open
- High
- Low
- Close
- Volume

The **Close Price** was selected as the target variable for prediction.

---

#  Data Preprocessing

The following preprocessing steps were applied:

- Date conversion into datetime format
- Sorting data chronologically
- Feature selection
- Data normalization using MinMaxScaler
- Sliding window sequence generation
- Train/Validation/Test split

##  Dataset Split

- 70% Training Data
- 15% Validation Data
- 15% Testing Data

---

#  Sequence Creation

A sliding window approach was used where:

- Previous 50 time steps were used
- To predict the next closing price

---

#  Model Architecture

##  RNN Model
- SimpleRNN Layer (50 Units)
- Dense Output Layer

##  LSTM Model
- LSTM Layer (50 Units)
- Dense Output Layer

##  GRU Model
- GRU Layer (50 Units)
- Dense Output Layer

---

#  Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 20 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | Mean Squared Error |
| Activation Function | tanh |

---

#  Performance Evaluation

The models were evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

---

#  Results

| Model | MSE ↓ | MAE ↓ | RMSE ↓ | MAPE ↓ |
|---|---|---|---|---|
| RNN | 342.96 | 12.90 | 18.52 | 2.35% |
| LSTM | 455.80 | 14.49 | 21.35 | 2.70% |
| GRU | **284.49** | **11.05** | **16.87** | **2.02%** |

---

#  Result Analysis

##  GRU Model
- Best overall performance
- Lowest prediction error
- Better generalization
- Efficient in capturing temporal dependencies

##  LSTM Model
- Good at learning long-term dependencies
- Slightly higher prediction error than GRU

##  RNN Model
- Performs reasonably well
- Struggles with long-term dependencies
- Less effective during volatile stock movements

---
