#!/usr/bin/env python3
# authors: Jakub Barylski
# coding: utf-8

"""
Extract minimal set of representative models that cover specified fraction of the provided protein set.
"""

import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd
import tqdm

from utils import checkpoint, Parallel, BatchParallel, count_fasta


@checkpoint
def read_domtblout(hmmer_domtblout: Path,
                   score_cutoff: float = 20,
                   pickle_path: Path = None) -> Dict[str, Set[str]]:
    """
    Read HMMER output file and return dictionary of hmms and hit proteins
    :param hmmer_domtblout: HMMER3 domain table file
    :param score_cutoff: minimal HMMER3 score of the domain required to include it into analysis
    :param pickle_path: path to a temporary file storing results in case of crash or re-run with modified parameters
    :return: {hmm_0: {protein.a, protein.c, (...) protein.z},
              (...),
              hmm_N: {protein.b, protein.f, (...) protein.x}}
    """
    hmm_to_protein = defaultdict(set)
    with tqdm.tqdm() as progress:
        with hmmer_domtblout.open() as domtblout:
            for n, line in enumerate(domtblout, 1):
                fields = str(line).split()
                hmm, protein, score = fields[3], fields[0], float(fields[13])
                if score >= score_cutoff:
                    hmm_to_protein[hmm].add(protein)
                    progress.set_description(f'Reading {hmmer_domtblout.name} {n}')

    return hmm_to_protein


def shrink_randomly(model_hits_in_positive: Dict[str, Set[str]],
                    untested_models: Set[str],
                    find_at_least: int,
                    previous_found: int = 0) -> Tuple[Dict[str, Set[str]], int]:
    """
    Recursively try to delete random model as long as number of hot protein is above the limit
    :param model_hits_in_positive: dictionary with sets of proteins from positive (e.g. enzymatically active) dataset that are recognized by different HMMs
                                   {hmm_0: {protein.a, protein.c, (...) protein.z}, hmm_N: {protein.b, protein.f, (...) protein.x}}
    :param untested_models: set of models that were never dropped from the parent search
                     (In firs iteration it corresponds to all models and gradually shrinks to the empty set when subsequent HMMs are discarded)
    :param find_at_least: minimal number of proteins that align to set of models
                          (sets of HMMs that don't match this number are discarded)
    :param previous_found: optional value used during recursive model drop to keep track of the number of matched proteins
    :return: reduced model dictionary and number of aligned proteins form positive dataset
             {hmm_2: {protein.a, protein.c, (...) protein.z},
              (...),
              hmm_N: {protein.b, protein.f, (...) protein.x}},
             432
    """
    if not untested_models:
        return model_hits_in_positive, previous_found
    else:
        dropped_model = random.choice(tuple(untested_models))
        shrunken_model_dict = dict(model_hits_in_positive)
        del shrunken_model_dict[dropped_model]
        untested_models.remove(dropped_model)
        proteins_found = len(set([protein for model in shrunken_model_dict.values() for protein in model]))
        if proteins_found < find_at_least:
            return shrink_randomly(model_hits_in_positive,
                                   untested_models=untested_models,
                                   find_at_least=find_at_least,
                                   previous_found=proteins_found)
        else:
            return shrink_randomly(shrunken_model_dict,
                                   untested_models=untested_models,
                                   find_at_least=find_at_least,
                                   previous_found=proteins_found)


@checkpoint
def stochastic_search(model_hits_in_positive: Dict[str, Set[str]],
                      model_hits_in_background: Dict[str, Set[str]],
                      find_at_least: int,
                      replicates=int(1e6),
                      pickle_path: Path = None):
    """
    Select partially minimized set of representative models using a random depletion approach.
    :param model_hits_in_positive: dictionary with proteins from positive dataset (e.g. database of enzymatically active proteins) that are aligned to different HMMs
                                   {hmm_0: {protein.a, protein.c, (...) protein.z}, hmm_N: {protein.b, protein.f, (...) protein.x}}
    :param model_hits_in_background: dictionary with proteins from background dataset (e.g. UniRef90) that are aligned to different HMMs
                                   {hmm_0: {protein.x1, protein.x2, (...) protein.xZ}, hmm_N: {protein.x3, protein.x12, (...) protein.xX}}
    :param find_at_least: minimal number of proteins that align to set of models
                          (sets of HMMs that don't match this number are discarded)
    :param replicates: number of random searches to initialize
    :param pickle_path: path to a temporary file storing results in case of crash or re-run with modified parameters
    :return: {model: positive hits} dictionary for all models present in reduced sets,
             {model: background hits} dictionary for all models present in reduced sets,
             smallest observed size of representative model set
             and the largest observed number of aligned proteins form positive dataset
    """

    jobs = Parallel(shrink_randomly,
                    [None for _ in range(replicates)],
                    kwargs={'positive': model_hits_in_positive,
                            'find_at_least': find_at_least,
                            'untested': set(model_hits_in_positive)},
                    description=f'Random selection',
                    backend='loky')

    preselected_models = set()
    hit_proteins = set()
    all_results = [(set(r.keys()), len(r), n_proteins) for r, n_proteins in jobs.result]
    min_size = min([r_size for r, r_size, n_proteins in all_results])
    minimal_results = [(r, n_proteins) for r, r_size, n_proteins in all_results if r_size == min_size]
    print(f'{len(minimal_results)} {min_size}-sized combinations found within {len(all_results)} options')
    for model_set, proteins_found in minimal_results:
        preselected_models.update(model_set)
        hit_proteins.add(proteins_found)
    max_protein_covered = max(hit_proteins)

    print(f'Stochastic search found {len(preselected_models)} models in {min_size}-sized combination ({max_protein_covered} proteins covered)')
    preselected_hits_in_positive = {model: model_hits_in_positive[model] for model in preselected_models}
    preselected_hits_in_background = {model: model_hits_in_background[model] for model in preselected_models}

    return preselected_hits_in_positive, preselected_hits_in_background, min_size, max_protein_covered


def shrink_exhaustively(preselected_models: Tuple = None,
                        model_hits_in_positive: Dict[str, Set[str]] = None,
                        find_at_least: int = None) -> Set[Tuple[str]]:
    """
    Brute-force through all possible subsets of the model set to find the smallest arrangements that meet the "find_at_least" criterion.
    :param preselected_models: tuple with combination of preliminary selected models to refine
    :param model_hits_in_positive: dictionary with sets of proteins from positive (e.g. enzymatically active) dataset that are recognized by different HMMs
                                   {hmm_0: {protein.a, protein.c, (...) protein.z}, hmm_N: {protein.b, protein.f, (...) protein.x}}
    :param find_at_least: minimal number of proteins that align to set of models
                          (sets of HMMs that don't match this number are discarded)
    :return: all sub-combinations of initial model set that meet "find_at_least" criterion
             {(hmm_3, hmm_12, hmm_34, ... ),
              (...),
              (hmm_3, hmm_8, hmm_44, ... )}
    """
    if preselected_models is not None:
        model_hits_in_positive = {model: proteins for model, proteins in model_hits_in_positive.items() if model in preselected_models}
    proteins_found = len(set([protein for model in model_hits_in_positive.values() for protein in model]))
    if proteins_found < find_at_least:
        return set()
    else:
        result = {tuple(sorted(model_hits_in_positive.keys()))}
        for model_to_drop in model_hits_in_positive:
            result.update(shrink_exhaustively(model_hits_in_positive={model: proteins for model, proteins in model_hits_in_positive.items() if model != model_to_drop},
                                              find_at_least=find_at_least))
        return result


@checkpoint
def exhaustive_search(model_hits_in_positive: Dict[str, Set[str]],
                      model_hits_in_background: Dict[str, Set[str]],
                      expected_size: int,
                      find_at_least: int,
                      pickle_path: Path = None):
    """
    Re-shuffle all pre-filtered model combinations to find optimal one
    (smallest set of models that meets specified criterion).
    :param model_hits_in_positive: dictionary with proteins from positive dataset (e.g. database of enzymatically active proteins) that are aligned to different HMMs
                                   {hmm_0: {protein.a, protein.c, (...) protein.z}, hmm_N: {protein.b, protein.f, (...) protein.x}}
    :param model_hits_in_background: dictionary with proteins from background dataset (e.g. UniRef90) that are aligned to different HMMs
                                     {hmm_0: {protein.x1, protein.x2, (...) protein.xZ}, hmm_N: {protein.x3, protein.x12, (...) protein.xX}}
    :param expected_size: size of the minimal representative model sets observed during stochastic search
    :param find_at_least: minimal number of proteins that align to set of models
                          (sets of HMMs that don't match this number are discarded)
    :param pickle_path: path to a temporary file storing results in case of crash or re-run with modified parameters
    :return: {model: positive hits} dictionary for all models present in minimal combinations of modules,
             {model: background hits} dictionary for all models present in minimal combinations of modules,
             minimal combinations of modules
    """

    variants = list(combinations(model_hits_in_positive.keys(), expected_size))
    print(f'Searching for optimal representation within {len(variants):.2e} model combinations')
    jobs = BatchParallel(shrink_exhaustively,
                         variants,
                         batch_size=10,
                         pre_dispatch='50 * n_jobs',
                         partition_size=1000,
                         kwargs={'positive': model_hits_in_positive,
                                 'find_at_least': find_at_least},
                         backend='loky')

    selected_combinations = set()
    selected_models = set()

    for combination in jobs.result:
        selected_combinations.add(combination)
        selected_models.update(combination)

    selected_hits_in_positive = {model: model_hits_in_positive[model] for model in selected_models}
    selected_hits_in_background = {model: model_hits_in_background[model] for model in selected_models}

    return selected_hits_in_positive, selected_hits_in_background, selected_combinations


def result_summary(selected_combinations: Set[Tuple[str]],
                   selected_positives: Dict[str, Set[str]],
                   selected_background: Dict[str, Set[str]],
                   output_path: Path):
    """
    Generate the report describing best model sets and their hits in positive and background datasets
    :param selected_combinations: all minimal combinations than align to atl least "find_at_least" proteins.
    :param selected_positives: dictionary with proteins from positive dataset (e.g. database of enzymatically active proteins) that are aligned to selected HMMs
                               {hmm_0: {protein.a, protein.c, (...) protein.z}, hmm_N: {protein.b, protein.f, (...) protein.x}}
    :param selected_background: dictionary with proteins from background dataset (e.g. UniRef90) that are aligned to selected HMMs
                                {hmm_0: {protein.x1, protein.x2, (...) protein.xZ}, hmm_N: {protein.x3, protein.x12, (...) protein.xX}}
    :param output_path: path to the result spreadsheet file with summary of all minimal representative model sets
    """
    records = []
    for model_combination in selected_combinations:
        n_models = len(model_combination)
        filtered_positives = {model: selected_positives[model] for model in model_combination}
        positives_covered = len(set([protein for model in filtered_positives.values() for protein in model]))
        positive_model_hits = {model: len(proteins) for model, proteins in filtered_positives.items()}
        filtered_background = {model: selected_background[model] for model in model_combination}
        background_covered = len(set([protein for model in filtered_background.values() for protein in model]))
        background_model_hits = {model: len(proteins) for model, proteins in filtered_background.items()}
        records.append({'models': model_combination,
                        'n_models': n_models,
                        'positives_covered': positives_covered,
                        'background_covered': background_covered,
                        'positive_model_hits': '; '.join([f'{model}: {n}' for model, n in positive_model_hits.items()]),
                        'background_model_hits': '; '.join([f'{model}: {n}' for model, n in background_model_hits.items()])})
    result_table = pd.DataFrame.from_records(records)
    result_table.to_excel(output_path.as_posix(), engine='xlsxwriter', index=False)


if __name__ == '__main__':

    # algorithm settings

    min_domain_score = 20  # minimal HMMER3 score of the domain required to include it into analysis
    covarage_threshold = 0.98  # minimal fraction of proteins that must be covered by the HMM set

    # input and output files

    positive_protein_database = Path(' ... /MERGED.enzybase.domtblout')  # e.g. database of enzymatically active proteins
    positive_hmmer_out = Path(' ... /MERGED.enzybase.domtblout')  # hmmer search against positive database
    background_hmmer_out = Path(' ... /MERGED.uniref.domtblout')  # hmmer search against background database (e.g. UniRef90)
    hmm_filter_file = Path(' ... /hmm_filter.list')  # file with models of interest (used to remove irrelevant hits prior to analysis)

    output_spreadsheet = Path(f' ... /Result_summary_t{covarage_threshold}.ods')

    # read inputs

    proteins_in_database = count_fasta(positive_protein_database)
    min_proteins_hit = int(proteins_in_database * covarage_threshold)

    positive_dict = read_domtblout(positive_hmmer_out,
                                   score_cutoff=min_domain_score,
                                   pickle_path=positive_hmmer_out.parent.joinpath(f'{positive_hmmer_out.name}.pkl'))

    background_dict = read_domtblout(background_hmmer_out,
                                     score_cutoff=min_domain_score,
                                     pickle_path=background_hmmer_out.parent.joinpath(f'{background_hmmer_out.name}.pkl'))

    # filtering the data to remove irrelevant hmms

    if hmm_filter_file:
        with hmm_filter_file.open() as l_handle:
            selected_hmm_set = set([hmm_id for hmm_id in l_handle.read().splitlines() if hmm_id.strip()])
    filtered_positive_dict = {hmm: proteins for hmm, proteins in positive_dict.items() if hmm in selected_hmm_set}
    filtered_background_dict = {hmm: proteins for hmm, proteins in background_dict.items() if hmm in selected_hmm_set}

    # searching for minimal representative model sets

    reduced_positive_dict, reduced_background_dict, e_size, m_prot_cov = stochastic_search(filtered_positive_dict,
                                                                                           find_at_least=min_proteins_hit,
                                                                                           pickle_path=Path(f' ... /Stochastic_selection_t{covarage_threshold}.pkl'))

    minimal_positive, minimal_background, minimal_combinations = exhaustive_search(model_hits_in_positive=reduced_positive_dict,
                                                                                   expected_size=e_size,
                                                                                   find_at_least=min_proteins_hit,
                                                                                   pickle_path=Path(f' ... /Exhaustive_selection_t{covarage_threshold}.pkl'))

    # writing the result table

    result_summary(selected_combinations=minimal_combinations,
                   selected_positives=minimal_positive,
                   selected_background=minimal_background,
                   output_path=output_spreadsheet)
