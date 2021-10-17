import argparse
import json
import time
from glob import glob
import re

from aspects import Casing, WordOrder, Whitespace, CommonOther, SuffixPrefix, Spelling, Punctuation, Diacritics

if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser()
    parser.add_argument("m2_pattern",
                        help="Pattern to M2 files (e.g. /tmp/m2_folder/ or /tmp/m2_folder/lang8). Asterisk is appended to pattern.",
                        type=str)
    parser.add_argument("out", help="Path to file to store computed statistics (in JSON format).", type=str)
    args = parser.parse_args()

    start_m2load = time.time()
    m2_files_loaded = []
    for f in glob(args.m2_pattern + "*"):
        m2_file = re.sub(r'\n\n+', '\n\n', open(f).read().strip()).split("\n\n")
        m2_files_loaded.append(m2_file)
    print('M2 files loaded in {}'.format(time.time() - start_m2load))
    final_profile_statistics = {}

    print('Processing {} files'.format(len(m2_files_loaded)))

    # DIACRITICS
    start_diacritics = time.time()
    dict_key, values = Diacritics.estimate_probabilities(m2_files_loaded)

    final_profile_statistics[dict_key] = values
    print('Diacritics done in {}'.format(time.time() - start_diacritics))

    # SPELLING
    start_spelling = time.time()
    dict_key, values = Spelling.estimate_probabilities(m2_files_loaded)

    final_profile_statistics[dict_key] = values
    print('Spelling done in {}'.format(time.time() - start_spelling))

    # CASING
    start_casing = time.time()
    dict_key, values = Casing.estimate_probabilities(m2_files_loaded)

    final_profile_statistics[dict_key] = values
    print('Casing done in {}'.format(time.time() - start_casing))

    # WHITESPACE
    start_whitespace = time.time()
    dict_key, values = Whitespace.estimate_probabilities(m2_files_loaded)

    final_profile_statistics[dict_key] = values
    print('Whitespace done in {}'.format(time.time() - start_whitespace))

    # PUNCTUATION
    start_punct = time.time()
    dict_key, values = Punctuation.estimate_probabilities(m2_files_loaded)

    final_profile_statistics[dict_key] = values
    print('Punctuation done in {}'.format(time.time() - start_punct))

    # WORD ORDER
    start_wo = time.time()
    dict_key, values = WordOrder.estimate_probabilities(m2_files_loaded)

    final_profile_statistics[dict_key] = values
    print('Word-order done in {}'.format(time.time() - start_wo))

    # SUFFIX / PREFIX
    start_xfix = time.time()
    dict_key, values = SuffixPrefix.estimate_probabilities(m2_files_loaded)

    final_profile_statistics[dict_key] = values
    print('Suffix/prefix done in {}'.format(time.time() - start_xfix))

    # OTHER COMMON ERRORS
    start_other_common = time.time()
    dict_key, values = CommonOther.estimate_probabilities(m2_files_loaded)

    final_profile_statistics[dict_key] = values
    print('Common other done in {}'.format(time.time() - start_other_common))

    # FINAL DUMP

    with open(args.out, 'w') as fp:
        json.dump(final_profile_statistics, fp)
