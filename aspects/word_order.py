from collections import Counter

import numpy as np
from aspects import apply_m2_edits
from aspects import utils
from aspects.base import Aspect


class WordOrder(Aspect):
    def __init__(self, profile, lang, alpha=1, beta=0):
        super(WordOrder, self).__init__(profile, alpha, beta)

        tuples_with_wo_percentage = profile['word_order']['tuples_with_wo_percentage']
        self.tuples_with_wo_percentage = utils._apply_smoothing([tuples_with_wo_percentage], alpha, beta)[0]

        num_words_per_wo_change_distrib = profile['word_order']['num_words_per_wo_change_distrib']
        self.num_words_per_wo_change_distrib = utils._apply_smoothing_on_simple_dict(num_words_per_wo_change_distrib, alpha, beta)

    def apply(self, text, whitespace_info):
        changes = []
        text_words = text.split(' ')
        if len(text_words) < 2:
            return text, changes, whitespace_info

        for start_word_i, start_word in enumerate(text_words):
            remaining_words = len(text_words) - 1 - start_word_i

            if remaining_words < 2:
                continue

            if np.random.uniform(0, 1) < self.tuples_with_wo_percentage:
                this_wo_possible_tuples_probs = {k: v for k, v in self.num_words_per_wo_change_distrib.items() if int(k) <= remaining_words}
                normalized_probabilities = np.array(list(this_wo_possible_tuples_probs.values())) / np.sum(
                    list(this_wo_possible_tuples_probs.values()))
                num_words_in_word_order_error = int(np.random.choice(list(this_wo_possible_tuples_probs.keys()),
                                                                     p=normalized_probabilities))

                while True:
                    # we do not want the permutation to "do nothing"
                    perm = np.random.permutation(num_words_in_word_order_error)
                    if not np.array_equal(perm, np.array(range(num_words_in_word_order_error))):
                        break

                new_words = [''] * num_words_in_word_order_error
                for original_word_index, p_index in enumerate(perm):
                    new_words[original_word_index] = text_words[start_word_i + p_index]

                for i in range(num_words_in_word_order_error):
                    text_words[start_word_i + i] = new_words[i]

                # "fix" whitespace_info (try to copy it as it was originally)
                # this is definitely suboptimal
                for i, p_index in enumerate(perm):
                    if (start_word_i + i) > 0 and (start_word_i + p_index) > 0:
                        whitespace_info[start_word_i + i - 1] = whitespace_info[start_word_i + p_index - 1]

                    if (start_word_i + i) < len(whitespace_info) and (start_word_i + p_index) < len(whitespace_info):
                        whitespace_info[start_word_i + i] = whitespace_info[start_word_i + p_index]

                changes.append(['WO', 'replace {} with {}'.format(
                    " ".join(text.split(' ')[start_word_i:start_word_i + num_words_in_word_order_error]),
                    " ".join(text_words[start_word_i:start_word_i + num_words_in_word_order_error]))])

        return ' '.join(text_words), changes, whitespace_info

    @staticmethod
    def estimate_probabilities(m2_records):
        num_words_per_wo_errors = []
        wo_percentage = []

        for m2_file in m2_records:

            for info in m2_file:
                orig_sent, coder_dict = apply_m2_edits.processM2(info, [])

                if coder_dict:
                    coder_id = list(coder_dict.keys())[0]
                    cor_sent = coder_dict[coder_id][0]
                    paragraph_len = len(cor_sent)

                    if paragraph_len < 2:
                        continue

                    num_wo_in_paragraph = 0
                    for edit in coder_dict[coder_id][1]:
                        orig_start, orig_end, error_type, cor_tok, cor_start, cor_end = edit

                        if 'WO' in error_type:
                            num_words_per_wo_errors.append(orig_end - orig_start)
                            num_wo_in_paragraph += 1

                    wo_percentage.append(num_wo_in_paragraph / (paragraph_len - 1))

        tuples_with_wo_percentage = np.mean(wo_percentage)

        cnt_num_words_per_wo_errors = Counter(num_words_per_wo_errors)

        num_words_per_wo_change_distrib = {}
        for k, v in cnt_num_words_per_wo_errors.most_common():
            num_words_per_wo_change_distrib[k] = v / len(num_words_per_wo_errors)

        return 'word_order', {
            'tuples_with_wo_percentage': tuples_with_wo_percentage,
            'num_words_per_wo_change_distrib': num_words_per_wo_change_distrib
        }
