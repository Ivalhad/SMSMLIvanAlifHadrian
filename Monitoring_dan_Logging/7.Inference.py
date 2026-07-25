import requests
import json
import pandas as pd

MODEL_ENDPOINT = "http://127.0.0.1:8080/invocations"

# Load test data
test_df = pd.read_csv("../Membangun_model/heart_disease_preprocessing/test.csv")
sample = test_df.drop(columns=["target"]).head(5)

payload = {
    "dataframe_records": sample.to_dict(orient="records")
}

response = requests.post(
    MODEL_ENDPOINT,
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

print(f"Status Code : {response.status_code}")
print(f"Predictions : {response.json()}")