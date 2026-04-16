# Student Performance Factors

This project analyzes the factors influencing student performance (Exam Scores) using the [Student Performance Factors dataset from Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors).

## Project Structure

- `project.ipynb`: The main Jupyter Notebook that contains the entire workflow.
  - Dataset download using `kagglehub`.
  - Missing data / Duplicates / Data checks.
  - Data Cleaning (Feature Engineering, Ordinal and Binary Encoding for categorical variables).
  - Train/Test Split & K-Fold Cross Validation.
  - Machine Learning Models (Linear Regression, Decision Tree, XGBoost, CatBoost).
  - Benchmarking (MAE, MSE, RMSE, MAPE, Adjusted R^2) and visualizations.
  - Feature Importance.
- `requirements.txt`: Python dependencies required to run the notebook.
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
