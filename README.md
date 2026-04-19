# Student Performance Factors

This project analyzes the factors influencing student performance (Exam Scores) using the [Student Performance Factors dataset from Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors).

## Project Structure & Machine Learning Pipeline

The entire machine learning workflow is executed in `project.ipynb`. The pipeline consists of the following phases:

1. **Data Ingestion and Cleaning**
   - Automatically downloads the dataset using `kagglehub`.
   - Maps categorical columns to optimal numerical (ordinal) formats and maps boolean features (0/1).
   - Handles missing and duplicate values via Median / Mode imputations.

2. **Exploratory Machine Learning (Baselines)**
   - Prepares an 80/20 train/test unseen split.
   - Evaluates baseline models: Decision Tree, Linear Regression, XGBoost, and CatBoost using 5-Fold Cross Validation.
   - Measures Root Mean Squared Error (RMSE), MAE, MAPE, and Adjusted R².

3. **Feature Engineering & Selection**
   - Calculates **Feature Importance** utilizing an out-of-the-box XGBoost model to find the primary drivers of exam scores.
   - Compares subsets (Top 5 features vs. Bottom 5 features) to gauge predictive signal density.
   - **Composite Feature Engineering**: Uses `itertools` to generate 2-way and 3-way mathematical interaction features (combinations of top features mapping complex relations), inserting the most impactful composite columns into the dataset.

4. **Robust Preprocessing**
   - **Outlier Handling**: Applies Sklearn's `IsolationForest` to strictly remove statistical anomalies (~3% contamination) from the training pool.
   - **Feature Scaling**: Implements `StandardScaler` to ensure distance and linear algorithms (like Ridge/Linear Regression) are balanced for the final ensembling phase.

5. **Advanced Model Tuning & Ensembling**
   - **Hyperparameter Optimization**: Tunes XGBoost using `RandomizedSearchCV` to prevent overfitting (constrained by `n_jobs=1` to prevent macOS/Jupyter `BrokenProcessPool` serialization errors).
   - **Ultimate Voting Ensemble**: Chains the best tuned XGBoost estimator, a robust CatBoost model, and a mathematically distinct Linear `RidgeCV` within a weighted `VotingRegressor`.
   - **Target Transformation**: Wraps the entire ensemble in a `TransformedTargetRegressor` to standardize the target outputs dynamically.

6. **Explainability & Diagnostic Analytics**
   - **SHAP Analysis**: Deploys `shap.TreeExplainer` to interpret individual feature impacts globally visually.
   - **Residual Analysis**: Plots actual vs. predicted exam scores to catch model biases (e.g., tail undercutting at extreme high/low scales).

7. **Advanced Exploration: Two-Stage Hybrid Model (Oracle Segmentation)**
   - After identifying that a typical global Linear model vastly underestimates "high-achievers" (Scores > 76), the notebook constructs a **Two-Stage Piecewise Architecture**.
   - **Normal Students (Exam Score < 76):** Routed to Linear Regression (which models typical pacing perfectly).
   - **High Achievers (Exam Score $\ge$ 76):** Routed to the advanced XGBoost regressor to map entirely distinct feature behaviors.
   - Produces a unified segmented accuracy result.

## Repository Contents
- `project.ipynb`: The main structured Jupyter Notebook that contains the entire analytical and modeling workflow.
- `requirements.txt`: Python dependencies required to run the notebook locally.
- `.gitignore`: Files ignored by Git (including the `.venv` virtual environment folder and Kaggle config files).

## Local Setup Instructions

1. **Create and activate a Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Kaggle Authentication:**
   - Ensure you have generated a `kaggle.json` file from your Kaggle account (*Settings -> Create New API Token*).
   - Place this file in your user directory: `~/.kaggle/kaggle.json`.
   - Update permissions for security: `chmod 600 ~/.kaggle/kaggle.json`.

4. **Run the Notebook:**
   Open `project.ipynb` in VS Code or Jupyter Lab and execute the cells sequentially.
