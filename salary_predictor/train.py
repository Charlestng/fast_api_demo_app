import pandas as pd
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":

    ### MLFLOW Experiment setup
    experiment_name="salary_estimator"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)

    print("training model...")
    
    # Time execution
    start_time = time.time()

    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=False) # We won't log models right away

    # Import dataset
    df = pd.read_csv("data/Salary_Data.csv")

    # X, y split 
    X = df.loc[:, ["YearsExperience"]]
    y = df.loc[:, ["Salary"]]

    # Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Log experiment to MLFlow
    with mlflow.start_run(run_id = run.info.run_id) as run:
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_train)

        # Log model seperately to have more flexibility on setup 
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="salary_estimator",
            registered_model_name="salary_estimator_LR",
            signature=infer_signature(X_train, predictions)
        )
        
    print("...Done!")
    print(f"---Total training time: {time.time()-start_time}")