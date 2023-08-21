# -*- coding: utf-8 -*-

""" Helper functions for running TNM patient similarity
"""
import time
import pandas as pd
from vantage6.tools.util import info


def coordinate_task(client, input: dict, ids: list) -> list:
    """ Coordinate tasks to be sent to data nodes, which includes dispatching
    the task, waiting for results to return and collect completed results

    Parameters
    ----------
    client
        Vantage6 user or mock client
    input
        Input parameters for the task, such as the method and its arguments
    ids
        List with organisation ids that will receive the task

    Returns
    -------
    results
        Collected partial results from all the nodes
    """

    # Create a new task for the desired organizations
    info('Dispatching node tasks')
    task = client.create_new_task(
        input_=input,
        organization_ids=ids
    )

    # Wait for nodes to return results
    info('Waiting for results')
    task_id = task.get('id')
    task = client.get_task(task_id)
    while not task.get('complete'):
        task = client.get_task(task_id)
        info('Waiting for results')
        time.sleep(1)

    # Collecting results
    info('Obtaining results')
    results = client.get_results(task_id=task.get('id'))

    return results


def survival_rate(df: pd.DataFrame, cutoff: int, delta: int) -> list:
    """ Compute survival rate at certain time points after diagnosis

    Parameters
    ----------
    df
        DataFrame with TNM data
    cutoff
        Maximum number of days for the survival rate profile
    delta
        Number of days between the time points in the profile

    Returns
    -------
    survival_rates
        Survival rate profile
    """

    # Get survival days, here we assume the date of last follow-up as death date
    df['date_of_diagnosis'] = pd.to_datetime(df['date_of_diagnosis'])
    df['date_of_fu'] = pd.to_datetime(df['date_of_fu'])
    df['survival_days'] = df.apply(
        lambda x: (x['date_of_fu'] - x['date_of_diagnosis']).days, axis=1
    )

    # Get survival rate after a certain number of days
    times = list(range(0, cutoff, delta))
    all_alive = len(df[df['vital_status'] == 'alive'])
    all_dead = len(df[df['vital_status'] == 'dead'])
    survival_rates = []
    for time in times:
        dead = len(df[df['survival_days'] <= time])
        alive = (all_dead - dead) + all_alive
        survival_rates.append(alive)

    return survival_rates
