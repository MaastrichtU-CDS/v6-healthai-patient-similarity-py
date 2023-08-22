# -*- coding: utf-8 -*-

""" Federated algorithm for patient similarity for TNM data of NSCLC patients
"""
import numpy as np
import pandas as pd

from scipy.spatial import distance
from sklearn.cluster import KMeans
from vantage6.tools.util import info
from v6_healthai_patient_similarity_py.helper import coordinate_task
from v6_healthai_patient_similarity_py.helper import survival_rate


def master(
        client, data: pd.DataFrame, k: int, epsilon: int = 0.05,
        max_iter: int = 300, columns: list = None, org_ids: list = None
) -> dict:
    """ Master algorithm that coordinates the tasks and performs averaging

    Parameters
    ----------
    client
        Vantage6 user or mock client
    data
        DataFrame with the input data
    k
        Number of clusters to be computed
    epsilon
        Threshold for convergence criterion
    max_iter
        Maximum number of iterations to perform
    columns
        Columns to be used for clustering
    org_ids
        List with organisation ids to be used

    Returns
    -------
    results
        Dictionary with the final averaged result
    """

    # Get all organization ids that are within the collaboration or
    # use the provided ones
    info('Collecting participating organizations')
    organizations = client.get_organizations_in_my_collaboration()
    ids = [organization.get('id') for organization in organizations
           if not org_ids or organization.get('id') in org_ids]

    # Initialise k global cluster centroids, for now start with k random points
    # drawn from the first data node
    info('Initializing k global cluster centres')
    input_ = {
        'method': 'initialize_centroids_partial',
        'kwargs': {'k': k, 'columns': columns}
    }
    results = coordinate_task(client, input_, ids[:1])
    centroids = results[0]
    info(f'Initial: {centroids}')

    # The next steps are run until convergence is achieved or the maximum
    # number of iterations reached. In order to evaluate convergence,
    # we compute the difference of the centroids between two steps. We
    # initialise the `change` variable to something higher than the threshold
    # epsilon.
    iteration = 1
    change = 2*epsilon
    while (change > epsilon) and (iteration < max_iter):
        # The input for the partial algorithm
        info('Defining input parameters')
        input_ = {
            'method': 'kmeans_partial',
            'kwargs': {'k': k, 'centroids': centroids, 'columns': columns}
        }

        # Send partial task and collect results
        results = coordinate_task(client, input_, ids)

        # Organise local centroids into a matrix
        local_centroids = []
        for result in results:
            for local_centroid in result:
                local_centroids.append(local_centroid)
        X = np.array(local_centroids)

        # Average centroids by running kmeans on local results
        # TODO: add other averaging options
        info('Run global averaging for centroids')
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        info(f'Kmeans result {kmeans}')
        new_centroids = kmeans.cluster_centers_

        # Compute the sum of the magnitudes of the centroids differences
        # between steps. This change in centroids between steps will be used
        # to evaluate convergence.
        info('Compute change in cluster centroids')
        change = 0
        for i in range(k):
            diff = new_centroids[i] - np.array(centroids[i])
            change += np.linalg.norm(diff)
        info(f'Iteration: {iteration}, change in centroids: {change}')

        # Re-define the centroids and update iterations counter
        centroids = list(list(centre) for centre in new_centroids)
        iteration += 1

    # Averaged centroids
    info(f'Result for centroids: {centroids}')

    # Get survival profiles for the clusters per node
    info('Getting survival profiles per node')
    input_ = {
        'method': 'survival_profiles_partial',
        'kwargs': {'kmeans': kmeans, 'columns': columns}
    }
    results = coordinate_task(client, input_, ids)

    # Averaging survival profiles
    info('Averaging survival profiles')
    profiles = []
    for i in range(k):
        profile = np.zeros(len(results[0][0][0]))
        patients = 0
        for result in results:
            profile += np.array(result[0][i])
            patients += result[1][i]
        profiles.append(list(profile/patients))

    return {
        'centroids': centroids,
        'profiles': profiles
    }


def RPC_initialize_centroids_partial(
        data: pd.DataFrame, k: int, columns: list = None
) -> list:
    """ Initialise global centroids for kmeans

    Parameters
    ----------
    data
        Dataframe with input data
    k
        Number of clusters
    columns
        Columns to be used for clustering

    Returns
    -------
    centroids
        Initial guess for global centroids
    """
    # Select columns and drop rows with NaNs
    data = data[columns].dropna(how='any')

    # Remove duplicates
    data = data.drop_duplicates()

    # TODO: use a better method to initialize centroids
    info(f'Randomly sample {k} data points to use as initial centroids')
    df = data.sample(k)

    # Organise initial guess for centroids as a list
    centroids = []
    for index, row in df.iterrows():
        centroids.append(row.values.tolist())

    return centroids


def RPC_kmeans_partial(
        df: pd.DataFrame, k: int, centroids: list, columns: list = None
) -> list:
    """ Partial method for federated kmeans

    Parameters
    ----------
    df
        DataFrame with input data
    k
        Number of clusters to be computed
    centroids
        Initial cluster centroids
    columns
        List with columns to be used for kmeans, if none is given use everything

    Returns
    -------
    centroids
        List with the partial result for centroids
    """
    # Drop rows with NaNs
    df = df.dropna(how='any')

    info('Selecting columns')
    if columns:
        df = df[columns]

    info('Calculating distance matrix')
    distances = np.zeros([len(df), k])
    for i in range(len(df)):
        for j in range(k):
            xi = list(df.iloc[i].values)
            xj = centroids[j]
            distances[i, j] = distance.euclidean(xi, xj)

    info('Calculating local membership matrix')
    membership = np.zeros([len(df), k])
    for i in range(len(df)):
        j = np.argmin(distances[i])
        membership[i, j] = 1

    info('Generating local cluster centroids')
    centroids = []
    for i in range(k):
        members = membership[:, i]
        dfc = df.iloc[members == 1]
        centroid = []
        for column in columns:
            centroid.append(dfc[column].mean())
        centroids.append(centroid)

    return centroids


def RPC_survival_profiles_partial(
        df: pd.DataFrame, kmeans, columns: list
) -> list:
    """ Partial method for survival profiles

    Parameters
    ----------
    df
        DataFrame with input data
    kmeans
        Result of kmeans
    columns
        List with columns to be used for getting cluster membership

    Returns
    -------
    profiles
        List with the partial result for survival profiles
    """
    # Drop rows with NaNs
    df = df.dropna(how='any')

    info('Getting memberships')
    X = df[columns].values
    df['cluster'] = kmeans.predict(X)

    info('Getting survival rates')
    profiles = []
    patients = []
    for i in range(len(kmeans.cluster_centers_)):
        df_tmp = df[df['cluster'] == i]
        profiles.append(survival_rate(df_tmp, cutoff=730, delta=30))
        patients.append(len(df_tmp))

    return profiles, patients
