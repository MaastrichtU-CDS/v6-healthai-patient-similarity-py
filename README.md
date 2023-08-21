# Vantage6 algorithm for patient similarity for the HealthAI PoC

This algorithm was designed for the [vantage6](https://vantage6.ai/) 
architecture. 

## Input data

The algorithm expects each data node to hold a `csv` with the following data
and adhering to the following standard:

``` json
{
  "id": {
    "description": "patient identifier",
    "type": "string"
  },
  "t": {
    "description": "patient t stage",
    "type": "categorical",
    "values": [
      "T0", "T1", "T1a", "T1b", "T1c", "T2", "T2a", "T2b", "T3", "T3a", "T4"
    ]
  },
  "n": {
    "description": "patient n stage",
    "type": "categorical",
    "values": ["N0", "N1", "N2", "N2b", "N2c", "N3"]
  },
  "m": {
    "description": "patient m stage",
    "type": "categorical",
    "values": ["M0", "M1", "M1a", "M1b", "M1c"]
  }
}
```

## Using the algorithm

Below you can see an example of how to run the algorithm:

``` python
import time
from vantage6.client import Client

# Initialise the client
client = Client('http://127.0.0.1', 5000, '/api')
client.authenticate('username', 'password')
client.setup_encryption(None)

# Define algorithm input
input_ = {
    'method': 'master',
    'master': True,
    'kwargs': {
        'org_ids': [2, 3],          # organisations to run kmeans
        'k': 3,                     # number of clusters to compute
        'epsilon': 0.05,            # threshold for convergence criterion
        'max_iter': 300,            # maximum number of iterations to perform
        'columns': ['t', 'n', 'm']  # columns to be used for clustering
    }
}

# Send the task to the central server
task = client.task.create(
    collaboration=1,
    organizations=[2, 3],
    name='v6-healthai-patient-similarity-py',
    image='aiaragomes/v6-healthai-patient-similarity-py:latest',
    description='run tnm patient similarity',
    input=input_,
    data_format='json'
)

# Retrieve the results
task_info = client.task.get(task['id'], include_results=True)
while not task_info.get('complete'):
    task_info = client.task.get(task['id'], include_results=True)
    time.sleep(1)
result_info = client.result.list(task=task_info['id'])
results = result_info['data'][0]['result']
```

## Testing locally

If you wish to test the algorithm locally, you can create a Python virtual 
environment, using your favourite method, and do the following:

``` bash
source .venv/bin/activate
pip install -e .
python v6_kmeans_py/example.py
```

The algorithm was developed and tested with Python 3.7.

## Acknowledgments

This project was financially supported by the
[AiNed foundation](https://ained.nl/over-ained/).