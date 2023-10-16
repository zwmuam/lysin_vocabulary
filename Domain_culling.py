#!/usr/bin/env python3
# authors: Jakub Barylski, Sophia Bałdysz
# coding: utf-8
"""
Read the hmmsearch domtblout files, filter their content based on score,
filter out domains with significant overlap with other higer-scoring domains
and write culled results as gff, hmmer3-like domtblout
as well as txt with list of domains in each protein in the "SEQ_ID [tab] domain1, domain2, ... " (one sequence per line)
"""

from itertools import count
from pathlib import Path
from typing import List, Union, Dict

from tqdm import tqdm

PathLike = Union[Path, str]


# dictionary with column numbers for different HMMER3 domtblout file formats
PARSER_DICT = {'hmmsearch': {'prot_id': 0,
                             'hmm_id': 3,
                             'prot_start': 17,
                             'prot_end': 18,
                             'evalue': 6,
                             'ivalue': 12,
                             'score': 13},
               'hmmscan': {'hmm_id': 0,
                           'prot_id': 3,
                           'prot_start': 17,
                           'prot_end': 18,
                           'evalue': 6,
                           'ivalue': 12,
                           'score': 13}
               }


class Domain:
    """
    Representation of a single domain
    gathering all relevant attributes
    ane hosting relevant functions
    :ivar prot_id: identifier of the protein (e.g. EN64633788, A0A073K0L0)
    :ivar hmm_id: identifier of the domain/family model (e.g. begg_2ZCQV, PFAM_PF12671.8)
    :ivar prot_start: start of the domain in protein sequence (number of position in AA chain)
    :ivar prot_end: end of the domain in protein sequence (number of position in AA chain)
    :ivar score: hmmer3-formatted domains score
    :ivar uid: unique identifier of the object
    :ivar line: original line of the tblout file (used for debugging and tblout export)
    """

    instance_counter = count(1)

    def __init__(self, line: str, program: str = 'hmmsearch'):
        """
        Object initialisation (creation of the class instance)
        the functions assumes that has a line from hmmsearch
        and has to be told otherwise if hmmscan was used
        """
        line = line.rstrip('\n')
        split_line = line.split('\t') if program == 'mmseq' else line.split()
        self.prot_id = split_line[PARSER_DICT[program]['prot_id']]
        self.hmm_id = split_line[PARSER_DICT[program]['hmm_id']]
        self.prot_start, self.prot_end = int(split_line[PARSER_DICT[program]['prot_start']]), int(
            split_line[PARSER_DICT[program]['prot_end']])
        self.score = float(split_line[PARSER_DICT[program]['score']])
        self.uid = {next(Domain.instance_counter)}
        self.line = line

    def __repr__(self) -> str:
        """
        How an instance should look like in print etc.
        :return: human-readable representation of protein
        """
        return f'{self.hmm_id} [{self.prot_start} - {self.prot_end}] ({self.score})'

    def gff(self) -> str:
        """
        Represent a domain as a GFF line
        :return: gff-formatted line (no newline (\n) at the end)
        """
        attributes = {'ID': str(self.uid),
                      'HMM': self.hmm_id}
        attribute_string = ';'.join(f'{k}={v}' for k, v in attributes.items())
        return '\t'.join([self.prot_id,
                          'HMMER3',
                          'domain',
                          str(self.prot_start),
                          str(self.prot_end),
                          str(self.score),
                          '.', '.',
                          attribute_string])

    def seq_length(self) -> int:
        """
        Calculate length of domain in AA
        :return: length of domain in AA
        """
        return self.prot_end - self.prot_start + 1

    def overlaps(self, other: 'Domain') -> bool:
        """
        Check if any of the two analysed domains overlap
        on more than 50% of its length with the other
        :param other: domain that will be compared with 'self'
        :return: are these domains overlapping?
        """
        if self.prot_end >= other.prot_start and self.prot_start <= other.prot_end:
            overlap_start = max(self.prot_start, other.prot_start)
            overlap_end = min(self.prot_end, other.prot_end)
            overlap = overlap_end - overlap_start + 1
            if overlap < 1:
                raise NotImplementedError()
            if any([overlap > e.seq_length() * 0.5 for e in (self, other)]):
                return True
        elif other.prot_end >= self.prot_start and other.prot_start <= self.prot_end:
            return other.overlaps(self)
        return False


def resolve_overlap(domain_list: List[Domain]) -> Domain:
    """
    Given a list of domains choose top-scoring one
    :param domain_list: list of two or more domains
    :return: Top scoring domain from the cluster
    """
    ranking = sorted(domain_list, key=lambda dom: dom.score, reverse=True)
    return ranking[0]


def cull(file_path: PathLike,
         output_dir: PathLike,
         score_threshold: float = 20):
    """
    Choose only best (locally) domains for all proteins in a single file
    save a '.culling.gff', '.culling.txt' and '.culling.orgformat'
    files in the parent folder of the input file
    :param file_path: system path to a file
          (please avoid any non-standard characters
           including whitespaces)
    :param score_threshold: max domain e-value from hmmer or mmseq
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    print(f'reading file at {file_path.as_posix()}')
    with file_path.open() as inpt:
        domains_in_proteins = {}
        comment_lines, data_lines = 0, 0
        discarded, kept = 0, 0
        for ln, line in enumerate(inpt, 1):
            if line.strip() and not line.startswith('#'):
                data_lines += 1
                domain = Domain(line)
                if domain.score >= score_threshold:
                    if domain.prot_id not in domains_in_proteins:
                        domains_in_proteins[domain.prot_id] = []
                    domains_in_proteins[domain.prot_id].append(domain)
                    kept += 1
                else:
                    discarded += 1
            else:
                comment_lines += 1
    print(f'Read {ln} lines (including {data_lines} alignments and {comment_lines} comment or empty lines)')
    print(f'E-value filter: {score_threshold} ({kept} hits kept, {discarded} discarded)')
    write_files(domains_in_proteins, output_dir.joinpath(f'ALL.nin_score_{score_threshold}.{file_path.stem}'))

    with tqdm(total=len(domains_in_proteins)) as bar:
        bar.set_description('comparing')
        culled_domains_in_proteins = {}
        while domains_in_proteins:
            protein_id, domains = domains_in_proteins.popitem()
            domains.sort(key=lambda dom: dom.prot_start)
            filtered_domains = []
            while domains:
                new_domain = domains.pop(0)
                overlapping_domains = []
                non_overlapping_domains = []
                while domains:
                    old_domain = domains.pop(0)
                    if new_domain.overlaps(old_domain):
                        overlapping_domains.append(old_domain)
                    else:
                        non_overlapping_domains.append(old_domain)
                if overlapping_domains:
                    main_domain = resolve_overlap(overlapping_domains + [new_domain])
                    domains = non_overlapping_domains + [main_domain]
                    domains.sort(key=lambda dom: dom.prot_start)
                else:
                    filtered_domains.append(new_domain)
                    domains = non_overlapping_domains
            culled_domains_in_proteins[protein_id] = filtered_domains
            bar.update()
        write_files(culled_domains_in_proteins, output_dir.joinpath(f'Culled.nin_score_{score_threshold}.{file_path.stem}'))


def write_files(domain_dict: Dict[str, List[Domain]],
                path_stem: Path):
    """
    Write 3 files with selected (sub)set of domains
    - gff3 table for annotated proteins
    - hmmer3-like domtblout table
    - list of domains in each protein in the "SEQ_ID [tab] domain1, domain2, ... " (one sequence per line)
    :param domain_dict: dictionary of Domain objects {SEQ_ID: [domain1, domain2]}
    :param path_stem: desired path of output tiles (without the extension)
    """
    print('Writing files')
    gff_lines, domtblout_lines, txt_lines = [], [], []
    for protein_id, domains in domain_dict.items():
        gff_lines.extend([d.gff() for d in domains])
        domtblout_lines.extend([d.line for d in domains])
        dom_string = '; '.join([d.hmm_id for d in domains])
        txt_lines.append(f'{protein_id}\t{dom_string}')
    gff, domtblout, txt = [Path(f'{path_stem.as_posix()}.{ext}') for ext in ('gff', 'domtblout', 'txt')]
    for file_path, lines in ((gff, gff_lines), (domtblout, domtblout_lines), (txt, txt_lines)):
        with file_path.open('w') as fh:
            fh.write('\n'.join(lines))
    print(f'Files written at:\n{gff}\n{domtblout}\n{txt}')


if __name__ == '__main__':

    input_dir = Path('... /example_hmmsearch_results')
    output = input_dir.parent.joinpath(f'CULLED_{input_dir.name}')
    output.mkdir(parents=True)
    hmm_comparison_output = [f for f in input_dir.iterdir() if f.name.endswith('.domtblout')]

    min_score = 20
    """
    https://www.biorxiv.org/content/10.1101/2021.06.24.449764v2.full
    Searching sequence databases for functional homologs using profile HMMs: how to set bit score thresholds?
    (...). Bit scores were used as thresholds rather than E-values since they remain the same irrespective of the size of the database searched.
    In it roughly corresponds to e-value of ~1e-5 obtained in the HMMer3 search:
    enzy2clean_MMSEQS2_rep_seq.fasta (de-replicated database of lysins) × and PFAM 33.0
    """

    for input_path in hmm_comparison_output:
        cull(input_path,
             score_threshold=min_score,
             output_dir=output)


