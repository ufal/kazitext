import numpy as np
from aspects import apply_m2_edits
from aspects import utils
from aspects.base import Aspect


class Whitespace(Aspect):
    def __init__(self, profile, lang, alpha=1, beta=0):
        super(Whitespace, self).__init__(profile, alpha, beta)

        whitespace_errors_probs = profile['whitespace']['whitespace_errors_probs']
        self.whitespace_errors_probs = utils._apply_smoothing_on_simple_dict(whitespace_errors_probs, alpha, beta)

        probs_whitespace_in_other = profile['whitespace']['probs_whitespace_in_other']
        self.probs_whitespace_in_other = {}
        for k in probs_whitespace_in_other.keys():
            self.probs_whitespace_in_other[k] = utils._apply_smoothing_on_simple_dict(probs_whitespace_in_other[k], alpha, beta)

    def apply(self, text, whitespace_info):
        changes = []
        new_text = []
        text_words = text.split(' ')
        word_ind = 0
        while True:
            if word_ind >= len(text_words):
                break

            word = text_words[word_ind]

            if not word.isalpha():
                new_text.append(word)
                word_ind += 1
                continue

            if len(word) >= 2 and np.random.uniform(0, 1) < self.whitespace_errors_probs['insert']:
                # insert whitespace
                sep_index = np.random.randint(1, len(word))
                new_text.append(word[:sep_index] + " " + word[sep_index:])

                # if this is a last word, we must handle it differently
                if word_ind == len(text_words) - 1:
                    whitespace_info.insert(word_ind, True)  # word_ind==len(whitespace_info)
                else:
                    whitespace_info[word_ind] = ['I', True, 1, whitespace_info[word_ind]]
                word_ind += 1
                changes.append(['WHITESPACE', "insert: {}".format(new_text[-1])])
            elif word_ind < len(text_words) - 1 and word.isalpha() and text_words[word_ind + 1].isalpha() and np.random.uniform(0, 1) < \
                    self.whitespace_errors_probs['delete']:
                # delete
                new_text.append(word + text_words[word_ind + 1])
                whitespace_info[word_ind] = 'D'
                word_ind += 2
                changes.append(['WHITESPACE', "delete: {}".format(new_text[-1])])
            elif word_ind < len(text_words) - 1 and word.isalpha() and text_words[word_ind + 1].isalpha() and np.random.uniform(0, 1) < \
                    self.whitespace_errors_probs['other']:
                # remove spaces between multiple following tokens and insert some spaces at random

                # check alpha adjacent (we already checked that there are at least two of them)
                max_applicability = 2
                for token_right_plus in range(2, len(text_words) - word_ind):
                    if text_words[word_ind + token_right_plus].isalpha():
                        max_applicability += 1
                    else:
                        break

                this_word_whitespace_probs = {k: v for k, v in self.probs_whitespace_in_other.items() if int(k) <= max_applicability}

                this_word_whitespace_probs_flattened = {}  # { [num_space_in_cor, num_spaces_in_orig] = prob, ...}
                for num_spaces_in_cor in this_word_whitespace_probs:
                    for num_spaces_in_orig in this_word_whitespace_probs[num_spaces_in_cor]:
                        this_word_whitespace_probs_flattened[num_spaces_in_cor + "-" + num_spaces_in_orig] = \
                            this_word_whitespace_probs[num_spaces_in_cor][num_spaces_in_orig]

                this_word_whitespace_probs_flattened_normalized = np.array(list(this_word_whitespace_probs_flattened.values())) / np.sum(
                    list(this_word_whitespace_probs_flattened.values()))

                # select how many words to take from corrected and to how many words to transform them
                # while-cycle is to make sure that we do not select single token with single character in the corrected text
                while True:
                    num_words_to_take_tuple = np.random.choice(list(this_word_whitespace_probs_flattened.keys()),
                                                               p=this_word_whitespace_probs_flattened_normalized)

                    num_spaces_in_cor, num_spaces_in_orig = map(int, num_words_to_take_tuple.split('-'))
                    no_space_cor = "".join(text_words[word_ind:word_ind + num_spaces_in_cor + 1])
                    if len(no_space_cor) > 2:
                        break

                # insert spaces on random, but be sure, that it is not the first, last or next to a whitespace
                for _ in range(num_spaces_in_orig - 1):
                    while True:
                        index_to_insert_space = np.random.randint(1, len(no_space_cor))
                        if no_space_cor[index_to_insert_space - 1] != ' ' and no_space_cor[index_to_insert_space] != ' ':
                            break
                        else:
                            # check if there are still two adjacent non-space characters
                            insert_space_possible = False
                            for j in range(len(no_space_cor) - 1):
                                if no_space_cor[j] != ' ' and no_space_cor[j + 1] != ' ':
                                    insert_space_possible = True
                                    break

                            if not insert_space_possible:
                                index_to_insert_space = None
                                break

                    if not index_to_insert_space:
                        break

                    no_space_cor = no_space_cor[:index_to_insert_space] + " " + no_space_cor[index_to_insert_space:]

                new_text.append(no_space_cor)
                if num_spaces_in_cor > num_spaces_in_orig:
                    # new text has (num_spaces_in_cor -  num_spaces_in_orig) less tokens
                    for j in range(num_spaces_in_cor - num_spaces_in_orig):
                        whitespace_info[word_ind + j] = 'D'
                elif num_spaces_in_cor < num_spaces_in_orig:
                    # new text has (num_spaces_in_orig - num_spaces_in_cor) more tokens
                    whitespace_info[word_ind] = ['I', True, num_spaces_in_orig - num_spaces_in_cor, True]
                changes.append(
                    ['WHITESPACE', "other: {} -> {}".format(" ".join(text_words[word_ind:word_ind + num_spaces_in_cor]), new_text[-1])])

                word_ind += num_spaces_in_cor

            else:
                new_text.append(word)
                word_ind += 1

        # process whitespace_info
        ## first deletes
        whitespace_info = [x for x in whitespace_info if x != 'D']
        ##then inserts
        i = 0
        while True:
            if i >= len(whitespace_info):
                break

            if isinstance(whitespace_info[i], list) and len(whitespace_info[i]) == 4:
                _, insert_type, count, cur_type = whitespace_info[i]
                whitespace_info[i] = cur_type

                for _ in range(count):
                    whitespace_info.insert(i, insert_type)

            i += 1

        return " ".join(new_text), changes, whitespace_info

    @staticmethod
    def estimate_probabilities(m2_records):
        # TODO now we allow to delete space only between two alpha words, which may not be the best method

        whitespace_errors = {
            'insert': 0,  # a single space must be inserted to make the sentence correct
            'delete': 0,  # a single space must be deleted to make the sentence correct
            'other': 0  # some spaces must be deleted and some inserted to make the sentence correct
        }

        '''
        When other category is detected, multiple source tokens must be transformed into multiple output tokens. In this category, we collect
        statistics on number of tokens in source and in output (correction).
        '''
        stats_whitespaces_in_other = {}

        num_all_alpha_words = 0  # all words whose casing could be changed
        num_all_alpha_neighboring_pairs = 0  # number of adjacent words where both words are alpha

        for m2_file in m2_records:

            for info in m2_file:
                orig_sent, coder_dict = apply_m2_edits.processM2(info, [])

                if coder_dict:
                    coder_id = list(coder_dict.keys())[0]
                    cor_sent = coder_dict[coder_id][0]

                    for edit in coder_dict[coder_id][1]:
                        orig_start, orig_end, error_type, cor_tok, cor_start, cor_end = edit

                        if 'ORTH:WSPACE' in error_type or (
                                        'ORTH' in error_type and "".join(orig_sent[orig_start:orig_end]) == "".join(
                                    cor_sent[cor_start:cor_end])):
                            if orig_end - orig_start == 1:
                                whitespace_errors['insert'] += 1
                            elif cor_end - cor_start == 1:
                                whitespace_errors['delete'] += 1
                            else:
                                whitespace_errors['other'] += 1

                                num_spaces_in_orig = orig_end - orig_start - 1
                                num_spaces_in_cor = cor_end - cor_start - 1
                                if num_spaces_in_cor not in stats_whitespaces_in_other:
                                    stats_whitespaces_in_other[num_spaces_in_cor] = {}

                                if num_spaces_in_orig not in stats_whitespaces_in_other[num_spaces_in_cor]:
                                    stats_whitespaces_in_other[num_spaces_in_cor][num_spaces_in_orig] = 0

                                stats_whitespaces_in_other[num_spaces_in_cor][num_spaces_in_orig] += 1

                    num_all_alpha_words += sum([1 for x in orig_sent if x.isalpha()])
                    num_all_alpha_neighboring_pairs += sum(
                        [1 for i, x in enumerate(orig_sent) if i > 1 and x.isalpha() and orig_sent[i - 1].isalpha()])

        # ! NOTE that we are already switching insert and delete to be used in the direction corrected -> original
        whitespace_errors_probs = {
            'delete': whitespace_errors['insert'] / num_all_alpha_words,
            'insert': whitespace_errors['delete'] / num_all_alpha_neighboring_pairs,
            'other': whitespace_errors['other'] / num_all_alpha_neighboring_pairs,  # these are mostly two-tuples (approximation)
        }

        probs_whitespace_in_other = {}
        for num_spaces_in_cor in stats_whitespaces_in_other:
            probs_whitespace_in_other[num_spaces_in_cor] = {}
            for num_spaces_in_orig in stats_whitespaces_in_other[num_spaces_in_cor]:
                probs_whitespace_in_other[num_spaces_in_cor][num_spaces_in_orig] = stats_whitespaces_in_other[num_spaces_in_cor][
                                                                                       num_spaces_in_orig] / whitespace_errors['other']

        return 'whitespace', {
            'whitespace_errors_probs': whitespace_errors_probs,
            'probs_whitespace_in_other': probs_whitespace_in_other
        }
