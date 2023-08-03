#!/usr/bin/env python3
# authors: Jakub Barylski, Sophia Bałdysz
# coding: utf-8

"""
Draws a map for similar clusters generated by two clustering methods.
Requires two csv files with exported Cytoscape node tables after clustering (e.g. MCL). 
"""

from collections import defaultdict
from itertools import count
from pathlib import Path
from typing import Dict, Set, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_table(table_path: Path,
               prefix: str = '',
               cluster_column: str = '__mclCluster',
               sort: bool = True) -> Dict[str, Set[str]]:
    """
    Read csv-formatted Cytoscape node table
    and pick human-readable names for MCL clusters
    based on HMM members
    :param table_path: path to csv file with clustered models
    :param sort: sorts clusters by size (number of members in cluster), true if from largest to smallest
    :param prefix: prefix to add to the cluster name (e.g. name of clustering method, optional)
    :param cluster_column: column with cluster designations
    :return: {'clu_1': {'hmm_name_a', 'hmm_name_b', (...)}, (...), 'clu_N': {'hmm_name_y', 'hmm_name_z', (...)} }
    """
    raw_cluster_dict = defaultdict(set)
    cluster_count = count(1)
    table = pd.read_csv(table_path.as_posix(), usecols=['name', cluster_column, 'DEFINITION', 'TYPE'])
    records = table.to_dict(orient='records')
    for record in records:
        raw_cluster_dict[record[cluster_column]].add((record['name'], record['DEFINITION'].split('; ')[0], record['TYPE']))
    raw_cluster_dict = dict(raw_cluster_dict)
    print(f'Read {len(raw_cluster_dict)} clusters from {table_path.name}')

    # sort cluster from smallest to largest
    if sort:
        raw_cluster_dict = {k: v for k, v in sorted(raw_cluster_dict.items(), key=lambda x: len(x[1]), reverse=True)}

    # pick meaningful names for cluster

    uninformative_names = ('No_description_provided',
                           'Domain of unknown function',
                           'hypothetical protein',
                           'Uncharacterized',
                           'gp')  # typical domain names that don't confer any human-readable information

    renamed_cluster_dict = {}
    node_ctable_records = []
    for cluster, models in raw_cluster_dict.items():
        categories = iter(('MIN_REP', 'POSITIVE', 'none'))
        replacement_names = []
        while not replacement_names:
            current_category = next(categories)
            replacement_names = [(n, d) for n, d, t in models if t == current_category]
        preferred_names = [d for n, d in replacement_names if n.startswith('PFAM')
                           if not any([d.lower().startswith(x.lower()) for x in uninformative_names])]
        less_preferred_names = [d for n, d in replacement_names
                                if d not in uninformative_names
                                if not any([d.lower().startswith(x.lower()) for x in uninformative_names])]
        if preferred_names:
            replacement_names = preferred_names
        elif less_preferred_names:
            replacement_names = less_preferred_names
        else:
            replacement_names = [f'X']
        name_stem = sorted(replacement_names, key=lambda name: len(name))[0].split(' [')[0].replace(' ', '_')
        ordinal_string = f'{next(cluster_count)}'.zfill(2)
        selected_name = f'{prefix}{ordinal_string}_{name_stem}'
        if 'Domain_of_unknown_function' in selected_name:
            raise ValueError(str(less_preferred_names))
        renamed_cluster_dict[selected_name] = set([n for n, d, t in models])
        for model in models:
            node_ctable_records.append({'key': model[0], 'cluster': selected_name})

    return renamed_cluster_dict


def filter_dict(cluster_dict: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    Remove clusters with only one member from comparisons 
    :param cluster_dict: {cluster_name: (members)} dictionary, from which clusters with only one member will be filtered out
    :return: {'filtered_clu_1': {'hmm_a', 'hmm_b', (...)}, (...), 'filtered_clu_N': {'hmm_y', 'hmm_z', (...)} }
    """
    return {cluster: hmms for cluster, hmms in cluster_dict.items() if len(hmms) > 1}


def read_filter(table_path: Path,
                prefix: str = '') -> Dict[str, Set[str]]:
    """
    Read csv file and remove clusters with only one member before comparisons
    :param table_path: path to csv file with clustered models
    :param prefix: prefix to add to the cluster name (e.g. name of clustering method, optional)
    :return: {'filtered_clu_1': {'hmm_a', 'hmm_b', (...)}, (...), 'filtered_clu_N': {'hmm_y', 'hmm_z', (...)} }
    """
    return filter_dict(read_table(table_path=table_path,
                                  prefix=prefix))


def jaccard_index(reference_set: Set[Any],
                  compared_set: Set[Any]) -> float:
    """
    Calculate Jaccard similarity coefficient for two sets of elements, here: 2 sets of hmm models 
    from 2 different clustering methods 
    for details see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6439793/
    :param reference_set: elements of the reference set
    :param compared_set: elements of the compared set
    :return: Jaccard index value
    """
    intersection = reference_set.intersection(compared_set)
    union = reference_set | compared_set
    return len(intersection) / len(union)


def containment(reference_set: Set[Any],
                compared_set: Set[Any]) -> float:
    """
    Calculate Jaccard containment of the compared element set in the reference set
    for details see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6439793/
    :param reference_set: elements of the reference set
    :param compared_set: elements of the compared set
    :return: Jaccard containment value
    """
    intersection = reference_set.intersection(compared_set)
    return len(intersection) / len(reference_set)


def compare(reference_cluster_dict: Dict[str, Set[str]],
            compared_cluster_dict: Dict[str, Set[str]],
            method: callable = jaccard_index) -> pd.DataFrame:
    """
    Compares two dictionaries containing cluster names (keys)
    and lists of members (values) using specified comparison function.
    :param reference_cluster_dict: {'R_clu_1': {'hmm_a', 'hmm_b', (...)}, (...), 'R_clu_N': {'hmm_y', 'hmm_z', (...)} }
    :param compared_cluster_dict: {'C_clu_1': {'hmm_a', 'hmm_c', (...)}, (...), 'C_clu_N': {'hmm_w', 'hmm_z', (...)} }
    :param method: comparison function e.g. jaccard_index or containment
    :return: dataframe with similarity metrics used to construct the cluster-map
    """
    results_dataframe = pd.DataFrame()
    for reference_cluster, reference_set in reference_cluster_dict.items():
        for compared_cluster, compared_set in compared_cluster_dict.items():
            results_dataframe.loc[reference_cluster, compared_cluster] = method(reference_set, compared_set)
    results_dataframe.fillna(0, inplace=True)
    return results_dataframe


def make_cluster_map(reference_cluster_dict: Dict[str, Set[str]],
                     compared_cluster_dict: Dict[str, Set[str]],
                     output_path: Path,
                     method: callable = jaccard_index):
    """
    Generate cluster-map figure based on two dictionaries with cluster names and sets of elements
    :param reference_cluster_dict: {'R_clu_1': {'hmm_a', 'hmm_b', (...)}, (...), 'R_clu_N': {'hmm_y', 'hmm_z', (...)} }
    :param compared_cluster_dict: {'C_clu_1': {'hmm_a', 'hmm_c', (...)}, (...), 'C_clu_N': {'hmm_w', 'hmm_z', (...)} }
    :param output_path: path to output file (should include pyplot-compatible image extension)
    :param method: comparison function e.g. jaccard_index or containment
    """
    comparison_frame = compare(reference_cluster_dict=reference_cluster_dict,
                               compared_cluster_dict=compared_cluster_dict,
                               method=method)

    norm = plt.Normalize(0, 1)
    cluster_map = sns.clustermap(comparison_frame,
                                 annot=False,
                                 figsize=(20, 20),
                                 cmap='rocket_r',
                                 linewidths=0.2,
                                 norm=norm)
    cluster_map.ax_row_dendrogram.set_visible(False)
    cluster_map.ax_col_dendrogram.set_visible(False)
    cluster_map.savefig(output_path.as_posix(), format='svg', dpi=1200)


if __name__ == '__main__':

    # input and output files
    reference_file = Path(' ... /example_concurrence_clustered.csv')
    compared_file = Path(' ... /example_hhsuite_clustered.csv')
    refc, compc = read_filter(reference_file, 'Co'), read_filter(compared_file, 'Hh')

    out_dir = Path(" ... /ClusterComparison/")

    # run the comparison and generate figure for different comparison methods
    # "reverse" is used to generate reverse containment (reference in compared)
    for comparison_method, reverse in ((jaccard_index, ''),
                                       (containment, ''),
                                       (containment, '_reversed')):
        reference_clusters, compared_clusters = (compc, refc) if reverse else (refc, compc)
        out_path = out_dir.parent.joinpath(f'{comparison_method.__name__}{reverse}.svg')
        make_cluster_map(reference_clusters,
                         compared_clusters,
                         out_path,
                         comparison_method)
