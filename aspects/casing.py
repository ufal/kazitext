from aspects import apply_m2_edits
import numpy as np
from aspects.base import Aspect
from aspects import utils


# TODO ta distribuce char_change_case_probs asi neni uplne to prave orechove, protoze v nekterych slovech to bude asi treba cele upper, nebo jenom malo lower

class Casing(Aspect):
    def __init__(self, profile, lang, alpha=1, beta=0):
        super(Casing, self).__init__(profile, alpha, beta)

        word_casing_probs = profile['casing']['word_casing_probs']
        self.word_casing_probs = {}
        for k in word_casing_probs.keys():
            self.word_casing_probs[k] = utils._apply_smoothing_on_simple_dict(word_casing_probs[k], alpha, beta)

        char_change_case_prob = profile['casing']['char_change_case_prob']
        self.char_change_case_prob = utils._apply_smoothing([char_change_case_prob], alpha, beta)[0]

        self.final_punctuation_marks = ['.', '!', '?'] + ["\"", "„"]

    def apply(self, text, whitespace_info):
        changes = []
        new_text = []
        for word_ind, word in enumerate(text.split()):

            # if not word.isalpha():
            #     new_text.append(word)
            #     continue

            if word_ind == 0 or text[word_ind - 1] in self.final_punctuation_marks:
                applicability_place = 'start'
            else:
                applicability_place = 'other'

            if len(word) > 0 and word[0].isupper() and np.random.uniform(0, 1) < self.word_casing_probs[applicability_place]['first_lower']:
                new_text.append(word[0].lower() + word[1:])
                changes.append(['CASING', 'first_lower {}'.format(word)])
            elif len(word) > 0 and word.lower() != word and np.random.uniform(0, 1) < self.word_casing_probs[applicability_place]['all_lower']:
                new_text.append(word.lower())
                changes.append(['CASING', 'all_lower {}'.format(word)])
            # note that when doing mixed casing, we need to check that the word's upper- and lower- cased version actually differ (e.g. 鈔)
            elif word.lower() != word.upper() and np.random.uniform(0, 1) < self.word_casing_probs[applicability_place]['other']:
                new_word = list(word)

                while new_word == list(word):
                    for char_ind, char in enumerate(word):
                        if np.random.uniform(0, 1) < self.char_change_case_prob:
                            if char.isupper():
                                new_word[char_ind] = char.lower()
                            else:
                                new_word[char_ind] = char.upper()

                new_text.append("".join(new_word))
                changes.append(['CASING', 'other {} -> {}'.format(word, new_text[-1])])
            else:
                new_text.append(word)

        return " ".join(new_text), changes, whitespace_info

    @staticmethod
    def estimate_probabilities(m2_records):
        '''

            :param m2_files_loaded: list of loaded m2_files
             i.e. one item of the list was obtained by sequence of:
                m2_file = open(m2_file).read().strip().split("\n\n")
            :return:
            '''

        '''
        We differentiate between casing errors on the token right after dot or at the start of whole text, and casing errors in other places.
        We also note, whether the error is in the first letter or whole word was lowered or some other error.
        '''
        casing_errors = {
            'start': {
                'first_lower': 0,  # user wrote a lower-cased first letter of the word but should have written upper-cased letter
                'all_lower': 0,  # user wrote whole word in lower-case but some chars should have been uppercased
                'other': 0  # all other cases
            },
            'other': {
                'first_lower': 0,
                'all_lower': 0,
                'other': 0
            },
        }

        num_all_alpha_words_wo_start = 0  # all words whose casing could be changed
        num_tokens_with_first_upper_wo_start = 0 # all words starting with upper case letter
        num_tokens_with_any_upper_wo_start = 0  # all words containing at least one upper letter
        num_start_tokens = 0

        '''
        First lower and all_lower are revertible (we know what to do), but other is non-revertible, so we estimate
        number of chars that changed its casing
        '''
        char_change_case_probs = []

        def get_change_char_case_prob_for_pair(orig, cor):
            num_change = 0
            for o, c in zip(orig, cor):
                if o != c:
                    num_change += 1

            return num_change / len(orig)

        final_punctuation_marks = ['.', '!', '?'] + ["\"", "„"]  # " e.g. „Byls už v Grónsku?“

        for m2_file in m2_records:

            for info in m2_file:
                orig_sent, coder_dict = apply_m2_edits.processM2(info, [])

                if coder_dict:
                    coder_id = list(coder_dict.keys())[0]
                    cor_sent = coder_dict[coder_id][0]

                    for edit in coder_dict[coder_id][1]:
                        orig_start, orig_end, error_type, cor_tok, cor_start, cor_end = edit

                        if 'ORTH:CASING' in error_type or (
                                        'ORTH' in error_type and " ".join(orig_sent[orig_start:orig_end]).lower() == " ".join(
                                    cor_sent[cor_start:cor_end]).lower()):
                            if (orig_start == 0) or (orig_sent[orig_start - 1] in final_punctuation_marks):
                                applicability_place = 'start'
                            else:
                                applicability_place = 'other'

                            cor_tok_first_lower = cor_tok[0].lower() + cor_tok[1:]
                            if " ".join(orig_sent[orig_start:orig_end]) == cor_tok_first_lower:
                                diacr_type = 'first_lower'
                            elif " ".join(orig_sent[orig_start:orig_end]) == cor_tok.lower():
                                diacr_type = 'all_lower'
                            else:
                                diacr_type = 'other'
                                char_change_case_probs.append(
                                    get_change_char_case_prob_for_pair(" ".join(orig_sent[orig_start:orig_end]), cor_tok))

                            casing_errors[applicability_place][diacr_type] += 1

                    num_start_tokens += sum([1 for x in orig_sent if
                                             x in final_punctuation_marks])  # no + 1 for the first token, because there is no token after last punct

                    num_tokens_with_first_upper_wo_start += sum([1 for x in orig_sent[1:] if
                                                         len(x) > 0 and x[0].isupper()])

                    num_tokens_with_any_upper_wo_start += sum([1 for x in orig_sent[1:] if
                                                         len(x) > 0 and x.lower() != x])

                    num_all_alpha_words_wo_start += sum([1 for x in orig_sent if
                                                         x.isalpha()]) - 1  # -1 for the first token (punct tokens are filtered out as they are not alpha)

        word_casing_probs = {}
        for applicability_place in casing_errors:
            word_casing_probs[applicability_place] = {}
            for diacr_type in casing_errors[applicability_place]:
                if applicability_place == 'start':
                    word_casing_probs[applicability_place][diacr_type] = casing_errors[applicability_place][
                                                                             diacr_type] / num_start_tokens
                else:
                    if diacr_type == 'first_lower':
                        word_casing_probs[applicability_place][diacr_type] = casing_errors[applicability_place][
                                                                                 diacr_type] / num_tokens_with_first_upper_wo_start
                    elif diacr_type == 'all_lower':
                        word_casing_probs[applicability_place][diacr_type] = casing_errors[applicability_place][
                                                                                 diacr_type] / num_tokens_with_any_upper_wo_start
                    else:
                        word_casing_probs[applicability_place][diacr_type] = casing_errors[applicability_place][
                                                                                 diacr_type] / num_all_alpha_words_wo_start

        char_change_case_prob = np.mean(char_change_case_probs)
        return 'casing', {'word_casing_probs': word_casing_probs,
                          'char_change_case_prob': char_change_case_prob
                          }
