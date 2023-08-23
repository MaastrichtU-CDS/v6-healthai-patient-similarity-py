# -*- coding: utf-8 -*-

""" Sample code to test the federated algorithm with a mock client
"""
import os
from vantage6.tools.mock_client import ClientMockProtocol


# Start mock client
data_dir = os.path.join(
    os.getcwd(), 'v6_healthai_patient_similarity_py', 'local'
)
client = ClientMockProtocol(
    datasets=[
        os.path.join(data_dir, 'data1.csv'),
        os.path.join(data_dir, 'data2.csv')
    ],
    module='v6_healthai_patient_similarity_py'
)

# Get mock organisations
organizations = client.get_organizations_in_my_collaboration()
print(organizations)
ids = [organization['id'] for organization in organizations]

# Check master method
master_task = client.create_new_task(
    input_={
        'master': True,
        'method': 'master',
        'kwargs': {
            'org_ids': [0, 1],
            'k': 4,
            'epsilon': 0.01,
            'max_iter': 50,
            'columns': ['t_num', 'n_num', 'm_num']
        }
    },
    organization_ids=[0, 1]
)
results = client.get_results(master_task.get('id'))
profiles = results[0]['profiles']

for profile in profiles:
    print(profile[-1])
