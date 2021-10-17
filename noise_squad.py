import argparse
import json

from introduce_errors import get_aspects_generator, load_tokenizer, introduce_errors_in_line
from introduce_errors_levels import level_to_operations


def find_answer_borders_in_context(answers):
    min_left_border = 1e10
    max_right_border = 0

    for answer in answers:
        left_border = answer['answer_start']
        right_border = left_border + len(answer['text'])

        if left_border < min_left_border:
            min_left_border = left_border

        if right_border > max_right_border:
            max_right_border = right_border

    return min_left_border, max_right_border


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("infile", type=str, help="Path to JSON file in SQUAD format with texts to be noised.")
    parser.add_argument("outfile", type=str, help="Path to file to store JSON file in SQUAD format with noised texts.")
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

    args = parser.parse_args()

    strip_all_diacritics, no_diacritics, no_spelling, no_casing, no_whitespace, no_punctuation, no_word_order, no_suffix_prefix, no_common_other = level_to_operations(
        args.level)

    # question and contexts in which the answer is not included are noised with args.level
    # the part of the context in which the answer is included is noised only with those operations that do not change number of tokens
    context_answer_noise_level = str(min(float(args.level[0]), 3))

    ca_strip_all_diacritics, ca_no_diacritics, ca_no_spelling, ca_no_casing, ca_no_whitespace, ca_no_punctuation, ca_no_word_order, ca_no_suffix_prefix, ca_no_common_other = level_to_operations(
        context_answer_noise_level)

    aspects_generator = get_aspects_generator(args.profile_file, args.lang, args.alpha, args.beta, strip_all_diacritics, 0.3,
                                              args.alpha_min, args.alpha_max, args.alpha_std, args.alpha_uniformity_prob)

    tokenizer = load_tokenizer(args.lang)

    noised_squad = {}
    with open(args.infile, 'r') as reader:
        squad = json.load(reader)

        noised_squad['version'] = squad['version']
        noised_squad['data'] = []

        for datum in squad['data']:
            noised_datum = {'title': datum['title'], 'paragraphs': []}

            for paragraph in datum['paragraphs']:
                # for each original paragraph, create n new paragraphs, where n == number of qas items in the original paragraph
                for qas in paragraph['qas']:
                    cur_aspect = aspects_generator()
                    noised_paragraph = {}
                    noised_qas = {'id': qas['id'], 'is_impossible': qas['is_impossible']}
                    noised_qas['question'], _ = introduce_errors_in_line(qas['question'], tokenizer, cur_aspect, no_diacritics, no_spelling,
                                                                         no_casing, no_whitespace, no_punctuation, no_word_order,
                                                                         no_suffix_prefix, no_common_other,
                                                                         no_error_sentence_boost=args.no_error_sentence_boost)

                    if qas['is_impossible']:
                        noised_paragraph['context'], _ = introduce_errors_in_line(paragraph['context'], tokenizer, cur_aspect, no_diacritics,
                                                                                  no_spelling, no_casing, no_whitespace, no_punctuation,
                                                                                  no_word_order, no_suffix_prefix, no_common_other,
                                                                                  no_error_sentence_boost=args.no_error_sentence_boost)

                        if 'plausible_answers' in qas:
                            noised_qas['plausible_answers'] = qas['plausible_answers']

                        noised_qas['answers'] = []


                    else:
                        # find the left and right borders in the context of all answers
                        context = paragraph['context']
                        left_context_answer_border, right_context_answer_border = find_answer_borders_in_context(qas['answers'])

                        # noise part of the context before any answer start and part of the context after all answers
                        left_context_noised = ''
                        if left_context_answer_border > 0:
                            left_context_noised, _ = introduce_errors_in_line(context[:left_context_answer_border], tokenizer, cur_aspect,
                                                                              no_diacritics, no_spelling, no_casing, no_whitespace,
                                                                              no_punctuation, no_word_order, no_suffix_prefix,
                                                                              no_common_other, no_error_sentence_boost=args.no_error_sentence_boost)

                            # noising script ignores trailing whitespaces
                            if context[left_context_answer_border - 1] == ' ':
                                left_context_noised += ' '

                        left_context_difference = len(left_context_noised) - left_context_answer_border

                        right_context_noised = ''
                        if context[right_context_answer_border:].strip():
                            right_context_noised, _ = introduce_errors_in_line(context[right_context_answer_border:], tokenizer, cur_aspect,
                                                                               no_diacritics, no_spelling, no_casing, no_whitespace,
                                                                               no_punctuation, no_word_order, no_suffix_prefix,
                                                                               no_common_other, no_error_sentence_boost=args.no_error_sentence_boost)

                        # noise part of the context that belongs to answers with those operations that keep the number of words the same
                        ca_context_noised, _ = introduce_errors_in_line(
                            context[left_context_answer_border:right_context_answer_border], tokenizer, cur_aspect,
                            ca_no_diacritics, ca_no_spelling, ca_no_casing, ca_no_whitespace, ca_no_punctuation, ca_no_word_order,
                            ca_no_suffix_prefix, ca_no_common_other, no_error_sentence_boost=args.no_error_sentence_boost)

                        noised_paragraph['context'] = left_context_noised + ca_context_noised + right_context_noised
                        noised_qas['answers'] = []

                        for answer in qas['answers']:
                            answer_start_in_original_text = answer['answer_start']

                            # we need to recalculate start_index
                            # start_index changed in two ways:
                            # 1. left_context -> left_context_noised : + left_context_difference
                            # 2. start_index may not be left_context_answer_border : some tokens in ca_context may changed before answer_start_in_original_text

                            num_words_in_ca_before_this_answer_start = 0
                            inside_ca_before_my_start_char_difference = 0
                            if left_context_answer_border != answer_start_in_original_text:
                                # some words before answer_start_in_original_text may have changed inside ca
                                num_words_in_ca_before_this_answer_start = len(
                                    context[left_context_answer_border:answer_start_in_original_text].split(' '))

                                noised_char_len_before_my_start_in_ca = len(
                                    ' '.join(ca_context_noised.split(' ')[:num_words_in_ca_before_this_answer_start]))

                                inside_ca_before_my_start_char_difference = noised_char_len_before_my_start_in_ca - (
                                    answer_start_in_original_text - left_context_answer_border)

                            answer_start_in_noised_text = answer_start_in_original_text + left_context_difference + inside_ca_before_my_start_char_difference

                            # we also need to recalculate answer tokens
                            num_answer_text_tokens = len(answer['text'].split(' '))
                            answer_tokens_in_noised_text = ca_context_noised.split(' ')[
                                                           num_words_in_ca_before_this_answer_start:num_words_in_ca_before_this_answer_start + num_answer_text_tokens]

                            noised_qas['answers'].append({
                                'text': ' '.join(answer_tokens_in_noised_text),
                                'answer_start': answer_start_in_noised_text,
                                'text_translated': ' '.join(answer_tokens_in_noised_text)
                            })

                    noised_paragraph['qas'] = [noised_qas]
                    noised_datum['paragraphs'].append(noised_paragraph)

            noised_squad['data'].append(noised_datum)

    with open(args.outfile, 'w') as writer:
        json.dump(noised_squad, writer)
