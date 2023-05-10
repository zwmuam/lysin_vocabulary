#!/usr/bin/env python3
# authors: Sophia Bałdysz
# coding: utf-8
"""
Calculate a p-values t-test checking if HMM is overrepresented in a selected set of sequences
(e.g. lytic enzymes, compared to provided background - e.g. all UniRef90 proteins)
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from scipy import stats

pd.options.mode.chained_assignment = None


class HmmTtest:
    # metrics to use in T-test
    metrics_to_use = ('max_scores', 'score_sums', 'proteins', 'proteinoccs')

    def __init__(self,
                 pickled_positive_hits: Path,
                 pickled_background_hits: Path,
                 positive_fasta: Path,
                 background_fasta: Path,
                 description_file: Path,
                 output_file: Path,
                 threshold: int = 3):
        """
        Pipeline parsing input files, calculating t-tests results
        for overrepresentation of HMMs in positive dataset and
        writing results as a xlsx spreadsheet.

        :param pickled_positive_hits: file containing a pickled dictionary with scores for alignments in positive protein set
                                      {'hmm1': {'protei_id1': [score1, score2], 'protei_id2': [score3, score4], (...)}, (...)}
        :param pickled_background_hits: file containing a pickled dictionary with scores for alignments in background protein set
                                      {'hmm11': {'protei_id11': [score11, score22], 'protei_id22': [score33, score44], (...)}, (...)}
        :param positive_fasta: path to protein fasta that was used to generate positive hits
        :param description_file: path to protein fasta that was used to generate background hits
        :param output_file: path to desired output xlsx file
        :param threshold: minimal number of proteins hits in the positive dataset to consider
        """

        self.sorted_hmms_positive = self.load_score_dict(pickled_positive_hits, threshold)
        self.sorted_hmms_background = self.load_score_dict(pickled_background_hits, threshold)

        self.population_positive = HmmTtest.dataset_size(positive_fasta)
        self.population_background = HmmTtest.dataset_size(background_fasta)

        self.description = self.load_descriptions(description_file)

        self.output = output_file

        self.run_ttest()

    @staticmethod
    def dataset_size(fastafile: Path) -> int:
        """
        Return a number of records (proteins)
        in the multi-fasta file
        :param fastafile: path to the multi-fasta file to assess
        :return: number of sequences
        """
        with fastafile.open() as handle:
            counter = 0
            for line in handle:
                if line.startswith(">"):
                    counter += 1
        return counter

    @staticmethod
    def load_descriptions(description_file: Path) -> Dict[str, str]:
        """
        Load pickle file with HMM descriptions
        :param description_file: path to pickle file with HMM descriptions
        :return: {'hmm1': 'human-readable description 1', (...), 'hmmX': 'human-readable description X'}
        """
        with description_file.open('rb') as handle:
            description_dict = pickle.load(handle)
        return description_dict

    @staticmethod
    def load_score_dict(inputfile,
                        threshold: int = 3) -> Dict[str, Dict[str, List[float]]]:
        """
        Load a dictionary of scores for protein alignments
        and filter out hmm that don't align to at least
        {threshold} proteins.

        :param inputfile: file containing a pickled dictionary with scores for alignments in positive protein set
        :param threshold: minimal number of proteins hits in the positive dataset to consider
        :return: {'hmm1': {'protei_id1': [score1, score2], 'protei_id2': [score3, score4], (...)}, (...)}
        """
        with inputfile.open('rb') as handle:
            dictionary = pickle.load(handle)
            # remove proteins with less than {threshold} protein hits
        dictionary = {hmm: proteinscores for hmm, proteinscores in dictionary.items() if
                      len(proteinscores) >= threshold}
        print(len(dictionary))
        return dictionary

    @staticmethod
    def _prepare_array_group(score_dict: Dict[str, List[float]],
                             population: int) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Prepare inputs for
        the T-test for single HMM
        :param score_dict: dictionary with protein alignment scores for a single HMM
        :param population: number of scanned proteins
        :return: [47.0, 33.44, (...)], [29.1, 12.14, (...)], [1, 0, 0, 1, (...)], [2, 0, 0, 1, (...)]
        """
        raw_scores = [[float(s) for s in scores] for scores in score_dict.values()]
        missing_values = (population - len(score_dict)) * [0]
        max_scores = [max(scores) for scores in raw_scores] + missing_values  # maximum score of all alignment in each protein
        score_sums = [sum(scores) for scores in raw_scores] + missing_values  # sum of all alignment scores in each protein
        proteins = [1 for _ in raw_scores] + missing_values  # HMM is present (1) or absent (0) in given protein
        protein_occurences = [len(scores) for scores in score_dict.values()] + missing_values  # how many times HMM has been aligned to proteins
        return max_scores, score_sums, proteins, protein_occurences

    def run_ttest(self):
        """
        Run actual T-test on prepared metric arrays
        and save results in output xlsx file
        """
        records = []
        for hmm, scores_positive in self.sorted_hmms_positive.items():
            record = {'hmm': hmm}
            tempkey = hmm.split('.')[0]
            poditive_array = HmmTtest._prepare_array_group(scores_positive, self.population_positive)
            if hmm in self.sorted_hmms_background:
                scores_background = self.sorted_hmms_background[hmm]
                background_array = HmmTtest._prepare_array_group(scores_background, self.population_background)
                for metric_name, positive_observations, background_observations in zip(HmmTtest.metrics_to_use, poditive_array, background_array):
                    record[f'{metric_name}_statistic'], record[f'{metric_name}_pvalue'] = stats.ttest_ind(positive_observations, background_observations)
                    record['status'] = 'Key appears in both sets.'
                    record['description'] = self.description[tempkey]
            else:
                for metric_name in HmmTtest.metrics_to_use:
                    record[f'{metric_name}_statistic'], record[f'{metric_name}_pvalue'] = 'NaN', 'NaN'
                    record['description'] = self.description[tempkey]
                record['status'] = 'Key does not appear in negative set.'
            records.append(record)

        results = pd.DataFrame.from_records(records, index='hmm')

        print(f'Writing results to {self.output}') # TODO exporter
        results.to_excel(self.output)


if __name__ == '__main__':
    pos_hits = Path(' ... /enzybase.pkl')  # file containing a pickled dictionary with scores for alignments in positive protein set
    bgn_hits = Path(' ... /uniref.pkl')  # file containing a pickled dictionary with scores for alignments in background protein set
    pos_fasta = Path(' ... /Lysin90.NonGM_EnzyBase.fasta')  # path to protein fasta that was used to generate positive hits
    bbgn_fasta = Path(' ... /UniRef90pb.Caudoviricetes_2731619_Bacteria_2_unclassified_dsDNA_phages_79205_UniProt.faa')  # path to protein fasta that was used to generate background hits
    descriptions = Path(' ... /totaldescription.pkl')  # path to protein fasta that was used to generate background hits

    output = Path(' ... /studentttest_20012023univsenzyALL.xlsx')

    myclass = HmmTtest(pickled_positive_hits=pos_hits,
                       pickled_background_hits=bgn_hits,
                       positive_fasta=pos_fasta,
                       background_fasta=bbgn_fasta,
                       description_file=descriptions,
                       output_file=output)
