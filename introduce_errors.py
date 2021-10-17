import argparse
import json
import time

import numpy as np
import udpipe_tokenizer
from aspects import Casing, WordOrder, Whitespace, CommonOther, SuffixPrefix, Spelling, Punctuation, Diacritics
from scipy.stats import truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    if sd == 0:
        sd = 0.00001

    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def get_aspects_generator(profile_file, lang, alpha_mean, beta, strip_all_diacritics, spelling_detailed_ratio, alpha_min=None,
                          alpha_max=None, alpha_std=None, alpha_uniformity_prob=0, num_aspects=1000):
    '''
     Returns generator that when called, returns next aspect to be used for noising
    '''

    if alpha_std == 0:
        aspect = load_basic_aspects(profile_file, lang, alpha_mean, beta, strip_all_diacritics, spelling_detailed_ratio)
        return lambda: aspect

    # if alpha_min / alpha_max are not specified, set them to cover most of the probability mass
    if alpha_min is None:
        alpha_min = max(0, alpha_mean - 3 * alpha_std)
    if alpha_max is None:
        alpha_max = max(0, alpha_mean + 3 * alpha_std)

    # divide into num_chunks in [alpha_min, alpha_max]
    chunk_size = (alpha_max - alpha_min) / (num_aspects - 1)

    alphas = []
    aspects = {}
    for i in range(num_aspects):
        cur_alpha = alpha_min + i * chunk_size
        aspects[cur_alpha] = load_basic_aspects(profile_file, lang, cur_alpha, beta, strip_all_diacritics, spelling_detailed_ratio)
        alphas.append(cur_alpha)

    if alpha_uniformity_prob < 1:
        # if uniform is not used always, instantiate generator of truncated normal values
        trunc_norm_alpha_generator = get_truncated_normal(alpha_mean, alpha_std, alpha_min, alpha_max)

    def get_next_aspect():
        if np.random.uniform(0, 1) < alpha_uniformity_prob:
            # select aspect (alpha) uniformly from whole interval
            return np.random.choice(list(aspects.values()))
        else:
            # sample alpha according to (truncated) normal normal distribution and return aspect with closest alpha
            sampled_alpha = trunc_norm_alpha_generator.rvs()
            closest_alpha = min(alphas, key=lambda x: abs(x - sampled_alpha))
            return aspects[closest_alpha]

    return get_next_aspect


def load_basic_aspects(profile_file, lang, alpha, beta, strip_all_diacritics, spelling_detailed_ratio):
    with open(profile_file, 'r') as f:
        profile = json.load(f)

    aspects = {
        'casing': Casing(profile, lang, alpha, beta),
        'common_other': CommonOther(profile, lang, alpha, beta),
        'diacritics': Diacritics(profile, lang, alpha, beta),
        'punctuation': Punctuation(profile, lang, alpha, beta),
        'spelling': Spelling(profile, lang, alpha, beta),
        'suffix_prefix': SuffixPrefix(profile, lang, alpha, beta),
        'whitespace': Whitespace(profile, lang, alpha, beta),
        'word_order': WordOrder(profile, lang, alpha, beta)
    }

    if strip_all_diacritics:
        aspects['diacritics'].all_wo_diacritics_perc = 1

    aspects['spelling'].spelling_detailed_ratio = spelling_detailed_ratio

    return aspects


def load_tokenizer(lang):
    return udpipe_tokenizer.UDPipeTokenizer(lang)


def _tokenize_line_and_get_whitespace_info(line, tokenizer):
    # tokenize input line and store space mapping

    if tokenizer is None:
        tokens = line.split(' ')
        text_whitespace_info = [True] * (len(tokens) - 1)
        return line, text_whitespace_info

    tokens = []
    for sentence_tokens in tokenizer.tokenize(line):
        for token in sentence_tokens:
            tokens.append(token.string.replace(' ', ''))

    text_whitespace_info = [True] * (
        len(tokens) - 1)  # stores information on whether there was space between adjacent tokens in text
    current_status = tokens[0]
    for i in range(len(tokens) - 1):
        if line.startswith(current_status + tokens[i + 1]):  # no space in between
            text_whitespace_info[i] = False
            current_status += tokens[i + 1]
        else:
            current_status += " " + tokens[i + 1]

    tokenized_line = " ".join(tokens)

    return tokenized_line, text_whitespace_info


def introduce_errors_in_line(line, tokenizer, aspects, no_diacritics, no_spelling, no_casing, no_whitespace, no_punctuation, no_word_order,
                             no_suffix_prefix, no_common_other, verbose=False, no_error_sentence_boost=0):
    original_tokenized_line, original_text_whitespace_info = _tokenize_line_and_get_whitespace_info(line, tokenizer)

    if verbose:
        print('Incoming line: {}'.format(original_tokenized_line), flush=True)

    assert len(original_text_whitespace_info) == len(original_tokenized_line.split(' ')) - 1

    num_iterations_done = 0
    max_iterations_to_try = 500
    random_number = np.random.uniform(0, 1)
    while True:
        tokenized_line = original_tokenized_line
        text_whitespace_info = original_text_whitespace_info
        line_changes = []

        if not no_common_other:
            tokenized_line, changes, text_whitespace_info = aspects['common_other'].apply(tokenized_line, text_whitespace_info)
            line_changes.extend(changes)

        assert len(text_whitespace_info) == len(tokenized_line.split(' ')) - 1

        if not no_suffix_prefix:
            tokenized_line, changes, text_whitespace_info = aspects['suffix_prefix'].apply(tokenized_line, text_whitespace_info)
            line_changes.extend(changes)

        assert len(text_whitespace_info) == len(tokenized_line.split(' ')) - 1

        if not no_spelling:
            tokenized_line, changes, text_whitespace_info = aspects['spelling'].apply(tokenized_line, text_whitespace_info)
            line_changes.extend(changes)

        assert len(text_whitespace_info) == len(tokenized_line.split(' ')) - 1

        if not no_word_order:
            tokenized_line, changes, text_whitespace_info = aspects['word_order'].apply(tokenized_line, text_whitespace_info)
            line_changes.extend(changes)

        assert len(text_whitespace_info) == len(tokenized_line.split(' ')) - 1

        if not no_diacritics:
            tokenized_line, changes, text_whitespace_info = aspects['diacritics'].apply(tokenized_line, text_whitespace_info)
            line_changes.extend(changes)

        assert len(text_whitespace_info) == len(tokenized_line.split(' ')) - 1

        if not no_casing:
            tokenized_line, changes, text_whitespace_info = aspects['casing'].apply(tokenized_line, text_whitespace_info)
            line_changes.extend(changes)

        assert len(text_whitespace_info) == len(tokenized_line.split(' ')) - 1

        if not no_whitespace:
            tokenized_line, changes, text_whitespace_info = aspects['whitespace'].apply(tokenized_line, text_whitespace_info)
            line_changes.extend(changes)

        assert len(text_whitespace_info) == len(tokenized_line.split(' ')) - 1

        if not no_punctuation:
            tokenized_line, changes, text_whitespace_info = aspects['punctuation'].apply(tokenized_line, text_whitespace_info)
            line_changes.extend(changes)

        assert len(text_whitespace_info) == len(tokenized_line.split(' ')) - 1

        num_iterations_done += 1
        if num_iterations_done < max_iterations_to_try and no_error_sentence_boost < 0 and len(line_changes) == 0 and abs(
                no_error_sentence_boost) < random_number:
            # we did not introduce any error but should have, try again
            continue

        if no_error_sentence_boost > 0 and len(line_changes) != 0 and random_number < no_error_sentence_boost:
            # we introduced some errors but to make the distribution more similar to reference, we remove the errors from the sentence
            tokenized_line, text_whitespace_info = original_tokenized_line, original_text_whitespace_info

        # detokenize text
        # if no tokenizer was provided, the text should be in a tokenized form and space should be in-between all tokens
        if not tokenizer:
            text_whitespace_info = [True] * len(text_whitespace_info)

        detokenized_line = ''
        for i in range(len(tokenized_line.split())):
            detokenized_line += tokenized_line.split(' ')[i]

            if i < len(text_whitespace_info) and text_whitespace_info[i]:
                detokenized_line += " "

        break

    return detokenized_line, line_changes


def introduce_errors_into_file(infile, outfile, profile_file, lang, debug, alpha, beta, save_input, strip_all_diacritics, no_diacritics,
                               no_spelling, no_casing, no_whitespace, no_punctuation, no_word_order, no_suffix_prefix, no_common_other,
                               spelling_detailed_ratio, verbose=False, random_seed=42, alpha_min=None, alpha_max=None, alpha_std=0,
                               alpha_uniformity_prob=0, no_error_sentence_boost=0):
    np.random.seed(random_seed)

    aspects_generator = get_aspects_generator(profile_file, lang, alpha, beta, strip_all_diacritics, spelling_detailed_ratio, alpha_min,
                                              alpha_max, alpha_std, alpha_uniformity_prob)

    tokenizer = load_tokenizer(lang)

    time_stats = []
    with open(infile, 'r', encoding='utf-8') as infile, open(outfile, 'w', encoding='utf-8') as outfile:
        line_ind = 0
        for line in infile:
            if not line.strip():  # if empty line, just copy it
                outfile.write("\n")
                continue

            line_ind += 1
            start_line_time = time.time()

            cur_aspects = aspects_generator()
            noised_line, line_changes = introduce_errors_in_line(line, tokenizer, cur_aspects, no_diacritics, no_spelling, no_casing,
                                                                 no_whitespace, no_punctuation, no_word_order, no_suffix_prefix,
                                                                 no_common_other, verbose, no_error_sentence_boost)

            if save_input:
                outfile.write(line.strip() + "\t" + noised_line.strip() + "\n")
            else:
                outfile.write(noised_line + "\n")

            if debug:
                print(line_changes)
                outfile.write(";".join([";".join(x) for x in line_changes]))
                outfile.write("\n")

            time_stats.append(time.time() - start_line_time)

            if verbose:
                if line_ind % 100 == 0:
                    print("{} processed. 1 line took on average {}".format(line_ind, np.mean(time_stats)), flush=True)

                    time_stats = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("infile", type=str, help="Path to file with text to be noised.")
    parser.add_argument("outfile", type=str, help="Path to file to store noised text.")
    parser.add_argument("profile_file", type=str, help="Path to file storing precomputed error statistics")
    parser.add_argument("lang", type=str, help="Language. E.g. cs, en, de, ru.")

    parser.add_argument("--debug", action='store_true', default=False,
                        help="If debug is enabled, changes will be written in the output file together with noised text.")

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

    parser.add_argument("--save-input", action='store_true', default=False,
                        help="If enabled, generated output file will store original lines and noised lines separated by tabulator.")

    parser.add_argument("--strip-all-diacritics", action='store_true', default=False, help="Strip all diacritics from all texts.")

    parser.add_argument("--no-diacritics", action='store_true', default=False, help="DO NOT introduce errors in diacritics")
    parser.add_argument("--no-spelling", action='store_true', default=False, help="DO NOT introduce errors in spelling")
    parser.add_argument("--no-casing", action='store_true', default=False, help="DO NOT introduce errors in casing")
    parser.add_argument("--no-whitespace", action='store_true', default=False, help="DO NOT introduce errors in whitespaces")
    parser.add_argument("--no-punctuation", action='store_true', default=False, help="DO NOT introduce errors in punctuation")
    parser.add_argument("--no-word-order", action='store_true', default=False, help="DO NOT introduce errors in word order")
    parser.add_argument("--no-suffix-prefix", action='store_true', default=False, help="DO NOT introduce errors in suffixes or prefixes")
    parser.add_argument("--no-common-other", action='store_true', default=False, help="DO NOT introduce common other errors")

    parser.add_argument("--spelling-detailed-ratio", type=float, default=0.3,
                        help="When introducing spelling errors, two distributions are used. One is estimate of general probabilities "
                             "(substitute, insert, delete, transpose), while the second one is more detailed and contains estimates of "
                             "substitutions of individual characters to individual characters .... (thus needs more annotated data). This"
                             " ratio is the proportion between the first and second method (value 0.3 means that the first method is used"
                             "with 0.3 chance and the second method with 1 - 0.3 = 0.7 chance).")

    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose mode")

    parser.add_argument("--seed", default=42, type=int, help="Random seed.")

    args = parser.parse_args()

    introduce_errors_into_file(args.infile, args.outfile, args.profile_file, args.lang, args.debug, args.alpha, args.beta,
                               args.save_input, args.strip_all_diacritics, args.no_diacritics, args.no_spelling, args.no_casing,
                               args.no_whitespace, args.no_punctuation, args.no_word_order, args.no_suffix_prefix, args.no_common_other,
                               args.spelling_detailed_ratio, args.verbose, args.seed, args.alpha_min, args.alpha_max, args.alpha_std,
                               args.alpha_uniformity_prob, args.no_error_sentence_boost)
