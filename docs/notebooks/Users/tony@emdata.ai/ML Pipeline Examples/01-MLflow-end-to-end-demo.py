# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # MLflow: An end-to-end Demo
# MAGIC 
# MAGIC ## What is MLflow?
# MAGIC MLflow is an [open source](https://github.com/mlflow) project designed to unify the machine learning lifecycle.<br> 
# MAGIC 
# MAGIC <img src="https://i.imgur.com/J46hHPb.png" style="width:500px"/> <br>
# MAGIC 
# MAGIC Simply stated:
# MAGIC > It is a combination of _conventions_, _specifications_ and _tools_, that allows you to manage the end-to-end ML lifecycle with full reproducibility, and ability to promote to a DevOps enabled Production Pipeline.
# MAGIC 
# MAGIC We can call MLflow with 
# MAGIC - Command Line Interface (`CLI`)
# MAGIC - `Python`, `R`, `Java` libraries
# MAGIC - `REST` API endpoint
# MAGIC - Graphics User Interface (`GUI`)
# MAGIC 
# MAGIC With a strong community backing, the software is released under an Apache license.
# MAGIC 
# MAGIC ## Installing MLflow
# MAGIC 
# MAGIC <img src="https://i.imgur.com/JJQg3J0.png" style="width:500px"/> <br>
# MAGIC 
# MAGIC The latest version of MLflow can be easily installed from the [PyPi repo](https://pypi.org/project/mlflow/) with a `pip install mlflow` and hosted on:
# MAGIC - a local machine (your laptop)
# MAGIC - a Virtual machine/VM cluster
# MAGIC 
# MAGIC From there, it can be accessed from a locally hosted endpoint e.g. `http://localhost:5000` - more information here on the official [MLflow quickstart page](https://www.mlflow.org/docs/latest/quickstart.html).
# MAGIC 
# MAGIC ## Using MLflow on Databricks
# MAGIC 
# MAGIC MLflow comes pre-integrated in the Databricks GUI, and is provided as a managed service in Databricks (so you don't have to worry about server maintenance, version updates, user identity tracking etc). <br>
# MAGIC 
# MAGIC > For our purposes today, we'll demonstrate MLflow's capabilities within Azure Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC # Primary components
# MAGIC We're going to demo the three **core** components that make up mlflow: <br>
# MAGIC ![MLflow](https://i.imgur.com/vulSrq4.png).<br>
# MAGIC   
# MAGIC **Note**: you don't have to use all three, each feature can be used independently. <br>
# MAGIC   
# MAGIC #### Tracking
# MAGIC This allows us to **log all aspects of the ML process** - like _different hyperparameters_ we tried, _evaluation metrics_, as well as the code we ran - alongside other arbitrary artifacts such as _test data_. <br>
# MAGIC 
# MAGIC This also provides a _leaderboard-style UI_ that makes it easy to see which model performed the best.
# MAGIC 
# MAGIC #### Projects
# MAGIC These are all about **reproducibility and sharing**. They combine _GIT_, the environment/model framework, either _conda_ or _docker_ and the specification that makes the code re-runnable. 
# MAGIC 
# MAGIC #### Models
# MAGIC An abstraction that allows us to **create/export models** from any open source framework via the _Tracking_ and _Projects_ abstractions. We can also export them to a standard format that can be deployed to any number of systems. Since most deployment systems use some sort of container based solution (e.g. _AzureML_ or _Sagemaker_), models make easy deployments to these systems - or we can deploy directly to _Kubernetes_ or _Azure Container Registry_.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Agenda
# MAGIC 
# MAGIC **In this notebook** we will demonstrate the following topics:<br>
# MAGIC 
# MAGIC <img src="https://i.imgur.com/7yofoyJ.png" style="width:1800px"/> <br>
# MAGIC 
# MAGIC #### Step 1: Load our exploration dataset into a DataFrame
# MAGIC In this case, we'll be using the "[Inside Airbnb](http://insideairbnb.com/get-the-data.html)" dataset, and loading it from a csv from an Azure Storage Container.
# MAGIC 
# MAGIC #### Step 2: Perform basic exploratory analysis
# MAGIC Like plotting on a heatmap to get a better sense of the data.
# MAGIC 
# MAGIC #### Step 3: `Tracking` Demo: Random Forest Experiment
# MAGIC We perform multiple experiments using scikit-learn's Random Forest Regressor and log the models on MLflow to demonstrate the tracking capabilities.
# MAGIC 
# MAGIC #### Step 4: `Projects` Demo: Package up a Random Forest model as a Project
# MAGIC We will define these components that makes up an MLflow Project.:
# MAGIC - **MLProject** file
# MAGIC - **Conda** file
# MAGIC - **Run** script <br>
# MAGIC 
# MAGIC We will also load and run a Project straight from git to demonstrate git integration capabilities.
# MAGIC 
# MAGIC #### Step 5: `Model Management` Demo: Explore model flavors and framework abstraction capabilities
# MAGIC We explore the power of model flavors and framework abstraction capabilities available with MLflow models.
# MAGIC 
# MAGIC #### Step 6: `Production Serving` Demo: Containerize the trained model and deploy to Azure Container Instances
# MAGIC We will build a _Docker Container Image_ for a trained model and deploy to _Azure Container Instance_ (can easily swap to Kubernetes as well).
# MAGIC 
# MAGIC #### Step 7: `Live scoring` Demo: Make a prediction against the live API endpoint
# MAGIC We use an _HTTP call_ and _Postman_ to make a prediction against a test payload.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 1: Load our exploration dataset into a DataFrame

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### "[Inside Airbnb](http://insideairbnb.com/get-the-data.html)": About the Dataset
# MAGIC The data behind the Inside Airbnb site is sourced from publicly available information from the Airbnb site.<br>
# MAGIC 
# MAGIC <img src="http://insideairbnb.com/images/insideairbnb_graphic_site_1200px.png" style="width:800px"/> <br>
# MAGIC 
# MAGIC We will load our file from a dataset made available (courtesy of Databricks) in an Azure Storage Account:

# COMMAND ----------

import pandas as pd

# Specify SAS
spark.conf.set(
  "fs.azure.sas.training.dbtraincanadacentral.blob.core.windows.net",
  "?ss=b&sp=rl&sv=2018-03-28&st=2018-04-01T00%3A00%3A00Z&sig=dwAT0CusWjvkzcKIukVnmFPTmi4JKlHuGh9GEx3OmXI%3D&srt=sco&se=2023-04-01T00%3A00%3A00Z")

# Load CSV to Spark DataFrame
df = spark.read.format("csv").load("wasbs://training@dbtraincanadacentral.blob.core.windows.net/airbnb/sf-listings/airbnb-cleaned-mlflow.csv", inferSchema = True, header = True)

# Conver to Pandas for downstream use
pandas_df = df.toPandas()

display(df)

# COMMAND ----------

display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Perform basic exploratory analysis

# COMMAND ----------

# MAGIC %md
# MAGIC Let's generate a heatmap of \[**latitude**, **longitude**\] _VS_ **price** against a map of San Francisco.

# COMMAND ----------

# MAGIC %python
# MAGIC from pyspark.sql.functions import col
# MAGIC 
# MAGIC try:
# MAGIC   df
# MAGIC except NameError: # Looks for local table if df not defined
# MAGIC   airbnbDF = spark.table("airbnb")
# MAGIC 
# MAGIC v = ",\n".join(map(lambda row: "[{}, {}, {}]".format(row[0], row[1], row[2]), df.select(col("latitude"),col("longitude"),col("price")/600).collect()))
# MAGIC displayHTML("""
# MAGIC <html>
# MAGIC <head>
# MAGIC  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.3.1/dist/leaflet.css"
# MAGIC    integrity="sha512-Rksm5RenBEKSKFjgI3a41vrjkw4EVPlJ3+OiI65vTjIdo9brlAacEuKOiQ5OFh7cOI1bkDwLqdLw3Zg0cRJAAQ=="
# MAGIC    crossorigin=""/>
# MAGIC  <script src="https://unpkg.com/leaflet@1.3.1/dist/leaflet.js"
# MAGIC    integrity="sha512-/Nsx9X4HebavoBvEBuyp3I7od5tA0UzAxs+j83KgC8PU0kgB4XiK4Lfe4y4cgBtaRJQEIFCW+oC506aPT2L1zw=="
# MAGIC    crossorigin=""></script>
# MAGIC  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.heat/0.2.0/leaflet-heat.js"></script>
# MAGIC </head>
# MAGIC <body>
# MAGIC     <div id="mapid" style="width:700px; height:500px"></div>
# MAGIC   <script>
# MAGIC   var mymap = L.map('mapid').setView([37.7587,-122.4486], 12);
# MAGIC   var tiles = L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
# MAGIC     attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors',
# MAGIC }).addTo(mymap);
# MAGIC   var heat = L.heatLayer([""" + v + """], {radius: 25}).addTo(mymap);
# MAGIC   </script>
# MAGIC   </body>
# MAGIC   </html>
# MAGIC """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: `Tracking` Demo: Grid Search on Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Question**: say you trained a Random Forest Regressor with a certain set of hyperparameters, and you get a certain result, how do you track it to compared with other algorithms? <br>
# MAGIC **Answer**: many people use spreadsheets, and sometimes pen and paper. <br>
# MAGIC 
# MAGIC Many others also have _"hyper-parameter déjà vu"_ - try one algorithm with one hyperparameter, then go on to try another. Then try another that seems eerily familiar - then we realize we’ve already tried that experiment. <br>
# MAGIC 
# MAGIC Let's see how MLflow `Tracking` solves this problem.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### MLflow `Tracking`: Brief Summary
# MAGIC 
# MAGIC **MLflow `Tracking`** is a logging API organized around the concept of **runs**, which are executions of data science code.  Runs are aggregated into **experiments** where many runs can be a part of a given experiment and an MLflow server can host many experiments.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" style="height: 200px; margin: 20px"/>
# MAGIC   
# MAGIC Each run can record the following information:<br><br>
# MAGIC 
# MAGIC - **Parameters:** Key-value pairs of input parameters such as the number of trees in a random forest model
# MAGIC - **Metrics:** Evaluation metrics such as RMSE or Area Under the ROC Curve
# MAGIC - **Artifacts:** Arbitrary output files in any format.  This can include images, pickled models, and even data files
# MAGIC - **Source:** The code that originally ran the experiment
# MAGIC 
# MAGIC MLflow tracking also serves as a **model registry** so tracked models can easily be stored and, as necessary, deployed into production. <br>
# MAGIC   
# MAGIC **Note**: In Databricks, each notebook acts as it's own "experiment", whereas in a local machine or vm MLflow setup, we would have to create a new experiment manually.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Setup MLflow
# MAGIC 
# MAGIC This script sets up MLflow for use in this notebook.

# COMMAND ----------

try:
  import os
  import mlflow
  from mlflow.tracking import MlflowClient
  from databricks_cli.configure.provider import get_config
  from mlflow.exceptions import RestException
  
  os.environ['DATABRICKS_HOST'] = get_config().host
  os.environ['DATABRICKS_TOKEN'] = get_config().token
  tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(dbutils.entry_point.getDbutils().notebook().getContext().tags())
  
  if tags.get("notebookId"):
    os.environ["MLFLOW_AUTODETECT_EXPERIMENT_ID"] = 'true'
  else:
    # Handles notebooks run by test server (executed as run)
    _name = "/Shared/BuildExperiment"
    
    try:
      client = MlflowClient()
      experiment = client.get_experiment_by_name("/Shared/BuildExperiment")

      if experiment:
        client.delete_experiment(experiment.experiment_id) # Delete past runs if possible
        
      mlflow.create_experiment(_name)
    except RestException: # experiment already exists
      pass
    
    os.environ['MLFLOW_EXPERIMENT_NAME'] = _name
  
  # Silence YAML deprecation issue https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
  os.environ["PYTHONWARNINGS"] = 'ignore::yaml.YAMLLoadWarning' 

  displayHTML("Initialized environment variables for MLflow server.")
except ImportError:
  raise ImportError ("Attach the MLflow library to your cluster through the clusters tab.  You cannot continue without this...")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's execute **3 experiments**  on our Airbnb dataset using scikit-learn's Random Forest Regressor.<br>
# MAGIC 
# MAGIC We want to know which combination of hyperparameter values: [**`n_estimators`**, **`max_depth`**, **`random_state`**] is the most effective.

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Test/Train Split on our Pandas DataFrame
X_train, X_test, Y_train, Y_test = train_test_split(pandas_df.drop(["price"], axis=1), pandas_df[["price"]].values.ravel(), random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Function Definition for experiment runs
# MAGIC 
# MAGIC To make our life easier, we define a parameterized function that will:
# MAGIC - Start an MLflow run based on the `run_name` provided into the function call (note that the experiment is this notebook)
# MAGIC - Create and train a model based on hyperparameters provided
# MAGIC - Log the model as an artifact
# MAGIC - Log hyperparameters and model performance metrics
# MAGIC - Create and log feature importance
# MAGIC - Create and log residual plot

# COMMAND ----------

def log_rf(run_name, params, X_train, X_test, y_train, y_test):
  import os
  import matplotlib.pyplot as plt
  import mlflow.sklearn
  import seaborn as sns
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import tempfile

  with mlflow.start_run(run_name=run_name) as run:
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")

    # Log params
    [mlflow.log_param(param, value) for param, value in params.items()]

    # Create metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("  mse: {}".format(mse))
    print("  mae: {}".format(mae))
    print("  R2: {}".format(r2))

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)  
    mlflow.log_metric("r2", r2)  
    
    # Create feature importance
    importance = pd.DataFrame(list(zip(df.columns, rf.feature_importances_)), 
                                columns=["Feature", "Importance"]
                              ).sort_values("Importance", ascending=False)
    
    # Log importances using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
    temp_name = temp.name
    try:
      importance.to_csv(temp_name, index=False)
      mlflow.log_artifact(temp_name, "feature-importance.csv")
    finally:
      temp.close() # Delete the temp file
    
    # Create plot
    fig, ax = plt.subplots()

    sns.residplot(predictions, y_test, lowess=True)
    plt.xlabel("Predicted values for Price ($)")
    plt.ylabel("Residual")
    plt.title("Residual Plot")

    # Log residuals using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="residuals-", suffix=".png")
    temp_name = temp.name
    try:
      fig.savefig(temp_name)
      mlflow.log_artifact(temp_name, "residuals.png")
    finally:
      temp.close() # Delete the temp file
      
    display(fig)
    # Return the experimentID and runID
    return [run.info.experiment_id, run.info.run_uuid]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment 1: [`n_estimators` = 10, `max_depth` = 5, `random_state` = 42]

# COMMAND ----------

params = {
  "n_estimators": 10,
  "max_depth": 5,
  "random_state": 42
}

[experimentID, run1ID] = log_rf("Experiment 1", params, X_train, X_test, Y_train, Y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment 2: [`n_estimators` = 100, `max_depth` = 10, `random_state` = 42]

# COMMAND ----------

params = {
  "n_estimators": 100,
  "max_depth": 10,
  "random_state": 42
}

[experimentID, run2ID] = log_rf("Experiment 2", params, X_train, X_test, Y_train, Y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment 3: [`n_estimators` = 1000, `max_depth` = 100, `random_state` = 42]

# COMMAND ----------

params = {
  "n_estimators": 1000,
  "max_depth": 100,
  "random_state": 42
}

[experimentID, run3ID] = log_rf("Experiment 3", params, X_train, X_test, Y_train, Y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Results in UI
# MAGIC 
# MAGIC #### Runs Summary
# MAGIC For each experiment, we see in the top right the **Runs Summary**:<br>
# MAGIC ![Run summary](https://i.imgur.com/z21Oaab.png)
# MAGIC 
# MAGIC #### Leaderboard
# MAGIC A **Leaderboard style** view:<br>
# MAGIC ![Leaderboard](https://i.imgur.com/VEEFwgX.png)
# MAGIC 
# MAGIC #### Detailed artifacts
# MAGIC And the corresponding **artifacts per experiment** (Experiment 3 here):<br>
# MAGIC ![Experiment 3](https://i.imgur.com/Seoxlgq.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Results in MlflowClient()
# MAGIC 
# MAGIC We can query past runs programatically in order to use this data back in Python. The pathway to doing this is an `MlflowClient` object that we initialized above.

# COMMAND ----------

client = MlflowClient()

runs = pd.DataFrame([(run.run_uuid, run.start_time, run.end_time, run.artifact_uri) for run in client.list_run_infos(experimentID)])
runs.columns = ["run_uuid", "start_time", "end_time", "artifact_uri"]

display(runs)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment 3: Results

# COMMAND ----------

# Sort DataFrame to extract Experiment 3 run information
run3_DF = runs.where(runs["run_uuid"] == run3ID).sort_values("start_time", ascending=False).iloc[0]

# COMMAND ----------

# List associated artifacts
display(dbutils.fs.ls(run3_DF["artifact_uri"]))

# COMMAND ----------

# Return the evaluation metrics for Experiment 3 run
client.get_run(run3_DF.run_uuid).data.metrics

# COMMAND ----------

# Reload the model from MLflow Registry and take a look at the feature importance
model3 = mlflow.sklearn.load_model(run3_DF.artifact_uri + "/random-forest-model/")
model3.feature_importances_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: `Projects` Demo: Package up a Random Forest model as a Project

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### MLflow `Projects`: Brief Summary
# MAGIC 
# MAGIC **ML Projects is a specification for how to organize code in a project.**  The heart of this is an **MLproject file,** a YAML specification for the components of an ML project.  This allows for more complex workflows since a project can execute another project, allowing for encapsulation of each stage of a more complex machine learning architecture.  This means that teams can collaborate more easily using MLflow projects.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-project.png" style="height: 300px; margin: 20px"/></div>
# MAGIC 
# MAGIC We will package up our Random Forest Regressor Model above and load it as a standalone entity.
# MAGIC 
# MAGIC #### Git integration
# MAGIC We will also trigger this scikit-learn ElasticNet [example MLflow project backed by GitHub](https://github.com/mlflow/mlflow-example) for an example of how to integrate MLflow with DevOps source control pipelines.
# MAGIC 
# MAGIC <div><img src="https://i.imgur.com/782nMYH.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Packaging up an MLflow Project
# MAGIC 
# MAGIC First we're going to create a MLflow project consisting of the following elements:<br><br>
# MAGIC 
# MAGIC 1. **MLProject file**:`MLProject`
# MAGIC 2. **Conda environment**: `conda.yaml`
# MAGIC 3. **Runtime script**: `train.py`
# MAGIC 
# MAGIC We're going to pass parameters into `train.py` so that we can try different hyperparameter options as well.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC #### Aside: using the `click` library
# MAGIC 
# MAGIC Our runtime script will be similar to what we ran with our Experiment Tracking function above, with the addition of decorators from the `click` library.<br>
# MAGIC 
# MAGIC [Click](https://click.palletsprojects.com/en/7.x/) allows us to parameterize our code:
# MAGIC 
# MAGIC <div><img src="https://i.imgur.com/h0yrchh.png" style="height: 700px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Save our Airbnb DataFrame as a csv
# MAGIC 
# MAGIC To be modular, we will be passing the location of our data file as a parameter into `train.py` (note, this can be in a Storage Account as well).

# COMMAND ----------

# Clean up directory
data_path = "dbfs:/user/raki.rahman@slalom.com/ml-production/airbnb/"

dbutils.fs.rm(data_path, True) # Clears the directory if it already exists
dbutils.fs.mkdirs(data_path)

# Save DataFrame to CSV
pandas_df.to_csv(data_path.replace("dbfs:","/dbfs") + "airbnb-cleaned-mlflow.csv", index=False)

display(dbutils.fs.ls(data_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Project Directory
# MAGIC We first create a directory to hold our project files.

# COMMAND ----------

train_path = "dbfs:/user/raki.rahman@slalom.com/ml-production/mlflow-model-training/"

dbutils.fs.rm(train_path, True) # Clears the directory if it already exists
dbutils.fs.mkdirs(train_path)

print("Created directory `{}` to house the project files.".format(train_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **MLProject file**:`MLProject`
# MAGIC 
# MAGIC This is the heart of an MLflow project. It includes pointers to the conda environment and a `main` entry point, which is backed by the file `train.py`.

# COMMAND ----------

dbutils.fs.put(train_path + "MLproject", 
'''
name: MLflow-Projects-Demo

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data_path: {type: str, default: "/dbfs/user/raki.rahman@slalom.com/ml-production/airbnb/airbnb-cleaned-mlflow.csv"}
      n_estimators: {type: int, default: 10}
      max_depth: {type: int, default: 20}
      max_features: {type: str, default: "auto"}
    command: "python train.py --data_path {data_path} --n_estimators {n_estimators} --max_depth {max_depth} --max_features {max_features}"
'''.strip())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **Conda environment**: `conda.yaml`

# COMMAND ----------

dbutils.fs.put(train_path + "conda.yaml", 
'''
name: MLflow-Projects-Demo
channels:
  - defaults
dependencies:
  - cloudpickle=1.2.2
  - numpy=1.14.3
  - pandas=0.23.0
  - scikit-learn=0.19.1
  - pip:
    - mlflow==1.0.0
'''.strip())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### **Runtime script**: `train.py` 
# MAGIC 
# MAGIC Identical in content to the `log_rf` function we explored above for our three experiments with Random Forest (except it has a `click` wrapper).

# COMMAND ----------

dbutils.fs.put(train_path + "train.py", 
'''
import click
import os
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

@click.command()
@click.option("--data_path", default="/dbfs/user/raki.rahman@slalom.com/ml-production/airbnb/airbnb-cleaned-mlflow.csv", type=str)
@click.option("--n_estimators", default=10, type=int)
@click.option("--max_depth", default=20, type=int)
@click.option("--max_features", default="auto", type=str)
def mlflow_rf(data_path, n_estimators, max_depth, max_features):

  with mlflow.start_run() as run:
    # Import the data
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
    
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")
    
    # Log params
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)

    # Log metrics
    mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
    mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))  
    mlflow.log_metric("r2", r2_score(y_test, predictions))  

if __name__ == "__main__":
  mlflow_rf() # Note that this does not need arguments thanks to click
'''.strip())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's view the files we created defining our MLflow project at `train_path`:

# COMMAND ----------

display(dbutils.fs.ls(train_path))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Execute locally packaged MLflow Project as a standalone entity

# COMMAND ----------

mlflow.projects.run(uri=train_path.replace("dbfs:","/dbfs"),
  parameters={
    "data_path": data_path.replace("dbfs:","/dbfs") + "airbnb-cleaned-mlflow.csv",
    "n_estimators": 10,
    "max_depth": 20,
    "max_features": "auto"
})

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC #### Results in UI
# MAGIC 
# MAGIC ##### Detailed artifacts
# MAGIC And the corresponding **artifacts** from this standalone Packaged experiment:
# MAGIC 
# MAGIC <div><img src="https://i.imgur.com/80iDZ8Z.png" style="height: 500px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC #### Execute GitHub MLflow Project as a standalone entity
# MAGIC 
# MAGIC We trigger this scikit-learn ElasticNet [example MLflow project backed by GitHub](https://github.com/mlflow/mlflow-example) by loading from GitHub directly.
# MAGIC 
# MAGIC Note below the **`MLProject`** file:
# MAGIC 
# MAGIC <div><img src="https://i.imgur.com/PvZnzMa.png" style="height: 350px; margin: 20px"/></div>

# COMMAND ----------

mlflow.run(
  uri="https://github.com/mlflow/mlflow-example",
  parameters={'alpha':0.4}
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC #### Results in UI
# MAGIC 
# MAGIC ##### Detailed artifacts
# MAGIC And the corresponding **artifacts** from this standalone Packaged experiment from GitHub:
# MAGIC 
# MAGIC <div><img src="https://i.imgur.com/Rx2mhcn.png" style="height: 500px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: `Model Management` Demo: Explore model flavors and framework abstraction capabilities

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### MLflow `Models`: Brief Summary
# MAGIC 
# MAGIC **MLflow models is a tool for deploying models that's agnostic to both the framework the model was trained in and the environment it's being deployed to (such as Azure ML or Kubernetes)**.  It's convention for packaging machine learning models that offers self-contained code, environments, and models.<br>
# MAGIC 
# MAGIC The main abstraction in the models package is the concept of **flavors**, which are different ways the model can be used. For instance, a TensorFlow model can be loaded as a _TensorFlow DAG_ (if supported by the serving layer) or as a _Python function_ (if not supported by serving layer): 
# MAGIC > Using the MLflow model convention allows for the model to be used regardless of the library that was used to train it originally. <br>
# MAGIC 
# MAGIC The primary difference between MLflow `projects` and `models` is that models are geared more towards inference and serving.  The `python_function` flavor of models gives a generic way of bundling models regardless of whether it was `sklearn`, `keras`, or any other machine learning library that trained the model. We can thereby deploy a python function without worrying about the underlying format of the model.<br>
# MAGIC 
# MAGIC **MLflow therefore maps any training framework to any deployment environment**, massively reducing the complexity of inference.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-models-enviornments.png" style="height: 400px; margin: 20px"/></div>
# MAGIC 
# MAGIC ### Model Flavors
# MAGIC 
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#module-mlflow.pyfunc" target="_blank">mlflow.pyfunc</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.keras.html#module-mlflow.keras" target="_blank">mlflow.keras</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#module-mlflow.pytorch" target="_blank">mlflow.pytorch</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#module-mlflow.sklearn" target="_blank">mlflow.sklearn</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.spark.html#module-mlflow.spark" target="_blank">mlflow.spark</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.tensorflow.html#module-mlflow.tensorflow" target="_blank">mlflow.tensorflow</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/index.html" target="_blank">You can see all of the flavors and modules here.</a>
# MAGIC 
# MAGIC Models also offer reproducibility since the run ID and the timestamp of the run are preserved as well.  
# MAGIC 
# MAGIC <div><img src="https://i.imgur.com/K74aGqv.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Explore Model `flavors`
# MAGIC 
# MAGIC To demonstrate the power of model flavors, let's first create two models using different frameworks:
# MAGIC - Random Forest on **scikit-learn**
# MAGIC - Neural Network on **Keras**

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Train and log a scikit-learn `Random Forest` model on MLflow

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow.sklearn

with mlflow.start_run(run_name="RF Model") as run:
  rf = RandomForestRegressor(n_estimators=100, max_depth=5)
  rf.fit(X_train, Y_train)

  rf_mse = mean_squared_error(Y_test, rf.predict(X_test)) # Calculate MSE
  mlflow.sklearn.log_model(rf, "model") # Log model
  mlflow.log_metric("mse", rf_mse) # Log MSE

  sklearnRunID = run.info.run_uuid
  sklearnURI = run.info.artifact_uri
  
  experimentID = run.info.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Train and log a Keras `Neural Network` on MLflow

# COMMAND ----------

import tensorflow as tf
tf.set_random_seed(42) # For reproducibility

from keras.models import Sequential
from keras.layers import Dense
import mlflow.keras

with mlflow.start_run(run_name="NN Model") as run:
  nn = Sequential([
    Dense(40, input_dim=21, activation='relu'),
    Dense(20, activation='relu'),
    Dense(1, activation='linear')
  ])

  nn.compile(optimizer="adam", loss="mse")
  nn.fit(X_train, Y_train, validation_split=.2, epochs=40, verbose=0)

  nn_mse = mean_squared_error(Y_test, nn.predict(X_test)) # Calculate MSE
  mlflow.keras.log_model(nn, "model") # Log model
  mlflow.log_metric("mse", nn_mse) # Log MSE

  kerasRunID = run.info.run_uuid
  kerasURI = run.info.artifact_uri

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Model flavor: `Random Forest` vs `Neural Network`

# COMMAND ----------

print(dbutils.fs.head(sklearnURI+"/model/MLmodel"))

# COMMAND ----------

print(dbutils.fs.head(kerasURI+"/model/MLmodel"))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Interchangibility: `Random Forest` vs `Neural Network`
# MAGIC 
# MAGIC Thanks to MLflow, we can now use both of these models in the **same way** and the **same syntax**, even though they were trained by different packages.

# COMMAND ----------

import mlflow.pyfunc

# COMMAND ----------

# Random Forest on scikit learn
rf_pyfunc_model = mlflow.pyfunc.load_model(model_uri=(sklearnURI+"/model").replace("dbfs:", "/dbfs"))
type(rf_pyfunc_model)

# COMMAND ----------

# Neural Network on Keras
nn_pyfunc_model = mlflow.pyfunc.load_model(model_uri=(kerasURI+"/model").replace("dbfs:", "/dbfs"))
type(nn_pyfunc_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### `.Predict` method: `Random Forest` vs `Neural Network`
# MAGIC 
# MAGIC Both will implement a `.predict` method.  The `sklearn` model is still of type `sklearn` because this **particular** package natively implements this method (but the functionality is identical).

# COMMAND ----------

rf_pyfunc_model.predict(X_test)

# COMMAND ----------

nn_pyfunc_model.predict(X_test).head(n = 5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: `Production Serving` Demo: Containerize the trained model and deploy to Azure Container Instances

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Model Serving in Azure: Brief Summary
# MAGIC 
# MAGIC Operationalizing machine learning models in a consistent and reliable manner is one of the most relevant technical challenges in the industry today.
# MAGIC 
# MAGIC [Docker](https://opensource.com/resources/what-docker), a tool designed to make it easier to package, deploy and run applications via containers is almost always involved in the Operationalization/Model serving process. Containers essentially abstract away the underlying Operating System and machine specific dependencies, allowing a developer to package an application with all of it's dependency libraries, and ship it out as one self-contained package. 
# MAGIC 
# MAGIC By using Dockerized Containers, and a Container hosting tool - such as **Kubernetes** or **Azure Container Instances**, our focus shifts to connecting the operationalized ML Pipeline (using MLflow) with a robust Model Serving tool to manage and (re)deploy our Models as it matures.
# MAGIC 
# MAGIC # Azure Machine Learning Services
# MAGIC We'll be using Azure Machine Learning Service to track experiments and deploy our model as a REST API via [Azure Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/).
# MAGIC 
# MAGIC ![](https://github.com/Microsoft/Azure-Databricks-NYC-Taxi-Workshop/raw/master/images/8-machine-learning/1-aml-overview.png)<BR>

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create an Azure ML Workspace
# MAGIC 
# MAGIC Before models can be deployed to Azure ML, an Azure ML Workspace must be created or obtained.<br>
# MAGIC 
# MAGIC The `azureml.core.Workspace.create()` function will load a workspace of a specified name or create one if it does not already exist.<br>
# MAGIC 
# MAGIC For more information about creating an Azure ML Workspace, see the [Azure ML Workspace management documentation](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace).

# COMMAND ----------

import azureml
from azureml.core import Workspace

workspace_name = "mlflowdemoamlworkspace"
workspace_location = "canadacentral"
resource_group = "MLflow_Demo"
subscription_id = "a14b17d4-c95b-4a54-b730-907766b4a71a"

workspace = Workspace.create(name = workspace_name,
                             subscription_id = subscription_id,
                             resource_group = resource_group,
                             location = workspace_location,
                             exist_ok=True)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Result**:
# MAGIC <div><img src="https://i.imgur.com/Sh479p0.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Use MLflow to build a Container Image for the trained Random Forest Regressor model `rf_pyfunc_model` from earlier
# MAGIC We will use the `mlflow.azuereml.build_image` function to build an Azure Container Image for the trained MLflow model.<br>
# MAGIC 
# MAGIC This function also registers the MLflow model with a specified Azure ML workspace.

# COMMAND ----------

import mlflow.azureml

model_image, azure_model = mlflow.azureml.build_image(model_uri=sklearnURI+"/model", 
                                                      workspace=workspace, 
                                                      model_name="Airbnb-randomforest",
                                                      image_name="airbnb-rf-container-image",
                                                      description="mlflow wrapped Random Forest model for predicting price of Airbnb properties in San Francisco",
                                                      synchronous=False)
model_image.wait_for_creation(show_output=True)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Result**:
# MAGIC <div><img src="https://i.imgur.com/3yKByBi.png" style="height: 200px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create an Azure Container Instance (ACI) webservice deployment using the model's Container Image
# MAGIC 
# MAGIC Using the Azure ML SDK, we will deploy the Container Image that we built for the trained MLflow model to ACI.

# COMMAND ----------

from azureml.core.webservice import AciWebservice, Webservice

aci_webservice_name = "airbnb-rf-aci"
aci_webservice_deployment_config = AciWebservice.deploy_configuration()
aci_webservice = Webservice.deploy_from_image(name=aci_webservice_name, image=model_image, deployment_config=aci_webservice_deployment_config, workspace=workspace)
aci_webservice.wait_for_deployment(show_output=True)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Result**:
# MAGIC <div><img src="https://i.imgur.com/Ct08tUo.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

aci_scoring_uri = aci_webservice.scoring_uri
print(aci_scoring_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: `Live scoring` Demo: Make a prediction against the live API endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load a sample input vector from the dataset
# MAGIC 
# MAGIC Lets load a single record from our test validation dataset for scoring.

# COMMAND ----------

sample = X_test.head(1)
sample_json = sample.to_json(orient="split")
query_input = list(sample.as_matrix().flatten())
print(sample_json)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Evaluate the sample input vector by sending an HTTP request
# MAGIC We will query the ACI webservice's scoring endpoint by sending an `HTTP POST` request that contains the input vector.
# MAGIC 
# MAGIC Let's define a function that takes in the 'scoring_uri' and `inputs` JSON, and returns the prediction `preds`.

# COMMAND ----------

import requests
import json

def query_endpoint_example(scoring_uri, inputs, service_key=None):
  headers = {
    "Content-Type": "application/json",
  }
  if service_key is not None:
    headers["Authorization"] = "Bearer {service_key}".format(service_key=service_key)
    
  print("Sending batch prediction request with inputs: {}".format(inputs))
  response = requests.post(scoring_uri, data=inputs, headers=headers)
  print("Response: {}".format(response.text))
  preds = json.loads(response.text)
  print("Received response: {}".format(preds))
  return preds

# COMMAND ----------

prediction = query_endpoint_example(scoring_uri=aci_scoring_uri, inputs=sample_json)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **We can also call the above uri with a tool like `Postman` to call the containerized model as well**:
# MAGIC <div><img src="https://i.imgur.com/mr00sW0.png" style="height: 800px; margin: 20px"/></div>