# online-customer-segmentation
A complete TFX pipeline from data ingestion to pushing the model to serving.

## README for Online Shopper Intention Classification Pipeline
This repository contains a machine learning pipeline built with TFX to predict online shopper purchasing intention using the publicly available "Online Shoppers Purchasing Intention Dataset".

## Pipeline Overview
The pipeline performs the following steps:

1. **Data Ingestion**: Reads the CSV data from the data directory.
2. **Data Validation and Preprocessing**:
   - Splits the data into training and evaluation sets.
   - Generates statistics for data exploration.
   - Creates a schema based on the data statistics.
   - Validates the data against the schema.
   - Undersamples the training data to address class imbalance (optional).
3. **Transformation**: Applies feature engineering and data cleaning using a user-defined Python function in the `transform.py` script.
4. **Hyperparameter Tuning**: Tunes hyperparameters for the model using a custom tuner defined in `tuner.py`.
5. **Training**: Trains a model (likely a Wide & Deep Network) using the transformed data and tuned hyperparameters (defined in `trainer.py`).
6. **Evaluation**: Evaluates the trained model against the baseline model and compares various metrics like accuracy, AUC, and recall using TFMA. Slices the evaluation results by features like Region, VisitorType, and Weekend.
7. **Model Pushing**: Pushes the model to a designated location (`serving_model_dir`) if it passes the evaluation criteria.

## Running the Pipeline Locally

Install the required packages listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

Run the pipeline:
    ```bash
    python run_pipeline.py
    ```
