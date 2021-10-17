import argparse
import csv

import numpy as np
import unidecode
import string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("infile", type=str, help="Path to tsv file with text to be noised.")

    args = parser.parse_args()

    m2_file = open(args.infile).read().strip().split("\n\n")

    for info in m2_file:
        info = info.split('\n')
        sentence = info[0].strip()[2:]
        print(f"S {sentence}")
        sentence = sentence.split()
        edits = info[1:]

        for edit in edits:
            span, error_type, replace_with, req, unk, aid = edit[2:].strip().split('|||')
            error_type = error_type.strip()
            replace_with = replace_with.strip()
            span_start, span_end = map(int, span.strip().split())
            original = ' '.join(sentence[span_start:span_end])


            if len(original) == 0 and len(replace_with) == 0:
                # even such data are in corpus :/
                pass
            elif original.lower() == replace_with.lower():
                # casing
                error_type = 'ORTH:CASING'
            elif error_type == 'Орфография' and len(original) > 0 and len(replace_with) > 0:
                # spelling
                error_type = 'SPELL'
            elif error_type == 'Вставить' or error_type == 'Заменить' or error_type == 'Убрать':
                if replace_with in string.punctuation or original in string.punctuation:
                    error_type = 'PUNCT'
            elif error_type == 'Пунктуация':
                error_type = 'PUNCT'
            elif unidecode.unidecode(original) == unidecode.unidecode(replace_with):
                error_type = 'DIACR'
            elif original.replace(' ', '') == replace_with.replace(' ', ''):
                error_type = 'ORTH:WSPACE'

            print(f"A {span}|||{error_type}|||{replace_with}|||{req}|||{unk}|||{aid}")

        print()


