import argparse
import numpy as np

from introduce_errors import get_aspects_generator, introduce_errors_in_line
from introduce_errors_levels import level_to_operations

def additional_postprocess(original, noisy):

    # ignore docstart lines (in german conll files)
    if original.strip() == '-DOCSTART-':
        return original

    if '\\' in noisy and '\\' not in " ".join(original):
        if np.random.rand() < 0.95:
            return noisy.replace('\\', '')

    return noisy

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

    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--debug", default=False, action='store_true', help="Random seed.")

    args = parser.parse_args()

    strip_all_diacritics, no_diacritics, no_spelling, no_casing, no_whitespace, no_punctuation, no_word_order, no_suffix_prefix, no_common_other = level_to_operations(
        args.level)

    # question and contexts in which the answer is not included are noised with args.level
    # the part of the context in which the answer is included is noised only with those operations that do not change number of tokens
    inside_entity_noise_level = str(min(float(args.level), 3))

    ca_strip_all_diacritics, ca_no_diacritics, ca_no_spelling, ca_no_casing, ca_no_whitespace, ca_no_punctuation, ca_no_word_order, ca_no_suffix_prefix, \
    ca_no_common_other = level_to_operations(inside_entity_noise_level)

    aspects_generator = get_aspects_generator(args.profile_file, args.lang, args.alpha, args.beta, strip_all_diacritics, 0.3,
                                              args.alpha_min, args.alpha_max, args.alpha_std, args.alpha_uniformity_prob)

    lines = open(args.infile).read().strip('\n').split('\n\n')
    outfile = open(args.outfile, 'w')

    for line in lines:
        cur_segment = []
        cur_segment_annotations = []
        line_tokens = line.split('\n')
        line_tokens.append("None\tNone")  # programmatic trick to make the final step easier
        for token in line_tokens:
            cur_word, cur_token_annotation = token.split('\t')
            cur_word = cur_word.strip()
            cur_token_annotation = cur_token_annotation.strip()

            prev_token_annotation = cur_segment_annotations[-1] if cur_segment_annotations else None

            # print(cur_word, cur_token_annotation, cur_segment, prev_token_annotation)

            # annotation same as in the previous token
            # if either first token or annotation did not change, i.e. previous was non-entity and this one is non-entity (both O), or previous
            # was entity and this one is also entity
            if not prev_token_annotation or (prev_token_annotation == cur_token_annotation) or (
                        prev_token_annotation != 'O' and cur_token_annotation != 'O' and cur_token_annotation != 'None'):
                cur_segment.append(cur_word)
                cur_segment_annotations.append(cur_token_annotation)
                continue

            # annotation changed - must apply changes now
            cur_aspect = aspects_generator()
            if prev_token_annotation == 'O':
                # outside entity - apply all available modifications
                # print(" ".join(cur_segment))
                noised_segment, line_changes = introduce_errors_in_line(" ".join(cur_segment), None, cur_aspect, no_diacritics, no_spelling, no_casing,
                                                             no_whitespace, no_punctuation, no_word_order, no_suffix_prefix,
                                                             no_common_other, no_error_sentence_boost=args.no_error_sentence_boost)
                # print(noised_segment)
                # print('-------')
                print(line_changes)

                noised_segment = additional_postprocess(" ".join(cur_segment), noised_segment)

                for noised_token in noised_segment.split(' '):
                    outfile.write(noised_token + "\t" + "O" + "\n")

            else:
                # inside entity, can apply only specific modifications
                # print(" ".join(cur_segment))
                noised_segment, line_changes = introduce_errors_in_line(" ".join(cur_segment), None, cur_aspect, ca_no_diacritics, ca_no_spelling,
                                                             ca_no_casing, ca_no_whitespace, ca_no_punctuation, ca_no_word_order,
                                                             ca_no_suffix_prefix, ca_no_common_other,
                                                             no_error_sentence_boost=args.no_error_sentence_boost)
                # print(noised_segment)
                # print('-------')
                print(line_changes)


                noised_segment = additional_postprocess(" ".join(cur_segment), noised_segment)

                for noised_token, annotation in zip(noised_segment.split(' '), cur_segment_annotations):
                    outfile.write(noised_token + "\t" + annotation + "\n")

            cur_segment_annotations = [cur_token_annotation]
            cur_segment = [cur_word]
        # print('end', cur_segment, prev_token_annotation)
        outfile.write('\n')

    outfile.close()
