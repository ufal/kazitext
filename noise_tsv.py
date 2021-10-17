import argparse
import csv

import numpy as np
from introduce_errors import get_aspects_generator, load_tokenizer, introduce_errors_in_line
from introduce_errors_levels import level_to_operations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("infile", type=str, help="Path to tsv file with text to be noised.")
    parser.add_argument("outfile", type=str, help="Path to file to store noised text.")
    parser.add_argument("profile_file", type=str, help="Path to file with noise estimates.")
    parser.add_argument("lang", type=str, help="Language.")
    parser.add_argument("level", type=str, help="Noise level.")
    parser.add_argument("--alpha", type=float, default=1.,
                        help="Chance multiplication factor. This is applied whenever these is any chance of introducing an error.")
    parser.add_argument("--beta", type=float, default=0.,
                        help="Uniformity smoothing factor. This is applied to all distributions from which we sample. When set to 1, "
                             "all distributions are uniform, when 0 all probabilities are estimated from data.")

    parser.add_argument("--alpha-min", type=float, default=None,
                        help="Minimum value of alpha to be used when either choosing from normal or uniform distribution. See "
                             "args.alpha-uniformity-prob for more details. If no value is provided and standard deviation is not 0, its "
                             "value is set to alpha - 3 * std")
    parser.add_argument("--alpha-max", type=float, default=None,
                        help="Maximum value of alpha to be used when either choosing from normal or uniform distribution. See "
                             "args.alpha-uniformity-prob for more details. If no value is provided and standard deviation is not 0, its "
                             "value is set to alpha + 3 * std")
    parser.add_argument("--alpha-std", type=float, default=0, help="Standard deviation to be be used for sampling alpha. ")
    parser.add_argument("--alpha-uniformity-prob", type=float, default=0,
                        help="Alpha sampling strategy. Set to 0 for sampling using normal distribution with mean in alpha and standard"
                             " deviation alpha-std; set to 1 for uniform sampling from [alpha-min, alpha-max]. Set in-between 0 and 1 "
                             " and each noise call will be selected randomly to use either uniform or sampling from normal distribution."
                             "Use 0 for generating testing data and try 0.5 for generating training data.")

    parser.add_argument("--no-error-sentence-boost", type=float, default=0,
                        help="This parameter serves to regularize number of sentences with and without induced errors."
                             "To be effective, it must non-zero number (0 acts as ignore this parameter). "
                             "If it is positive, anytime a noisy sentence should be outputted, it is with this"
                             " probability outputted without any introduced error."
                             "If this parameter is of negative value, then "
                             "anytime a sentence without error is to be outputted, than it is with this probability outputted with some "
                             " error.")

    parser.add_argument('-c', '--columns', nargs='+', help='Which columns to noise (indexing from 0).', required=True)
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--ignore-check", action='store_true', default=False, help="TODO")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose mode")

    args = parser.parse_args()

    np.random.seed(args.seed)

    strip_all_diacritics, no_diacritics, no_spelling, no_casing, no_whitespace, no_punctuation, no_word_order, no_suffix_prefix, no_common_other = level_to_operations(
        args.level)

    aspects_generator = get_aspects_generator(args.profile_file, args.lang, args.alpha, args.beta, strip_all_diacritics, 0.3,
                                              args.alpha_min, args.alpha_max, args.alpha_std, args.alpha_uniformity_prob)

    tokenizer = load_tokenizer(args.lang)

    with open(args.infile, 'r') as f, open(args.outfile, 'w') as writer:
        tsv_reader = csv.reader(f, delimiter="\t", quotechar='\x07')  # MRPC does have some bad lines -> quotechar cannot be "

        for i, tsv_line in enumerate(tsv_reader):

            if any([len(tsv_line) <= int(c) for c in args.columns]):
                if not args.ignore_check:
                    print('Weird line {}, not enough chunks: {}'.format(tsv_line, len(tsv_line)))

            else:
                # choose aspects for current text
                cur_aspect = aspects_generator()
                for column_ind in args.columns:
                    text_to_noise = tsv_line[int(column_ind)]

                    noised_word, _ = introduce_errors_in_line(text_to_noise, tokenizer, cur_aspect, no_diacritics, no_spelling, no_casing,
                                                              no_whitespace, no_punctuation, no_word_order, no_suffix_prefix,
                                                              no_common_other, args.verbose,
                                                              no_error_sentence_boost=args.no_error_sentence_boost)

                    tsv_line[int(column_ind)] = noised_word

            writer.write("\t".join(tsv_line) + '\n')
