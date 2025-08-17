# Pricing Asian Options DNN

This project implements a **Deep Neural Network** model to price **Asian options** using simulated market data.  
It is part of my work on combining **quantitative finance** and **machine learning** for option pricing.

## Project overview
Asian options are path-dependent derivatives whose payoff depends on the average price of the underlying asset over a certain period.  
Traditional pricing methods like **Monte Carlo simulation** can be computationally expensive.  
This project uses a **Deep Neural Network** to learn the mapping between market parameters and option prices, enabling faster pricing.

## Main features
- Dataset generation with market simulation parameters (spot price, volatility, interest rate, maturity, etc.)
- Normalization and preprocessing of input features
- Model architecture design and training (PyTorch)
- Evaluation against Monte Carlo benchmark

## Repository structure
- `data/` : contains datasets used for training and testing
- `models/` : contains trained model weights and preprocessing scalers
- `dataset_generator.py` : generates synthetic pricing datasets
- `ml_model.py` : defines and trains the DNN
- `evaluation.py` : evaluates the trained model against benchmarks

## Requirements
- Python 3.x
- PyTorch
- NumPy / Pandas
- Scikit-learn


