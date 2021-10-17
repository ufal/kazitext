from collections import Counter

import numpy as np
from aspects import apply_m2_edits
from aspects import utils
from aspects.base import Aspect


# TODO diakritika se vsechna Errantem rozsplituje, ale v cestine se obcas spatne pise " . vs . " -- prima rec a tak
# TODO ted ignoruju next_token_change_casing

class Punctuation(Aspect):
    def __init__(self, profile, lang, alpha=1, beta=0):
        super(Punctuation, self).__init__(profile, alpha, beta)

        punct_errors_aggregated_probs = profile['punctuation']['punct_errors_aggregated_probs']
        self.punct_errors_aggregated_probs = {}
        for k in punct_errors_aggregated_probs:
            self.punct_errors_aggregated_probs[k] = {}

            self.punct_errors_aggregated_probs[k]['I'] = utils._apply_smoothing([punct_errors_aggregated_probs[k]['I']], alpha,
                                                                               beta)[0]

            self.punct_errors_aggregated_probs[k]['S'] = utils._apply_smoothing_on_simple_dict(punct_errors_aggregated_probs[k]['S'],
                                                                                                alpha, beta)

        punct_errors_detailed_probs = profile['punctuation']['punct_errors_detailed_probs']
        self.punct_errors_detailed_probs = {}
        for k in punct_errors_detailed_probs:
            self.punct_errors_detailed_probs[k] = {}

            self.punct_errors_detailed_probs[k]['I'] = utils._apply_smoothing_on_simple_dict(punct_errors_detailed_probs[k]['I'],
                                                                                              alpha, beta)
            self.punct_errors_detailed_probs[k]['D'] = {}
            for kk in punct_errors_detailed_probs[k]['D']:
                self.punct_errors_detailed_probs[k]['D'][kk] = \
                    utils._apply_smoothing([punct_errors_detailed_probs[k]['D'][kk]], alpha, beta)[0]

            self.punct_errors_detailed_probs[k]['S'] = {}
            for kk in punct_errors_detailed_probs[k]['S']:
                self.punct_errors_detailed_probs[k]['S'][kk] = utils._apply_smoothing_on_simple_dict(
                    punct_errors_detailed_probs[k]['S'][kk], alpha, beta)

        self.final_punctuation_marks = ['.', '!', '?'] + ["\"", "„"]

    def apply(self, text, whitespace_info):

        '''
            Most of the punctuation tokens (in Czech) appends to the previous token (e.g. dot, question mark, colon).
            The following list stores those characters that in contrary append to the following token.
            '''
        punct_tokens_that_append_to_next_token = ['„', '(', '{', '[']

        changes = []
        original_text_splitted_into_tokens = text.split(' ')
        num_tokens_in_original_text = len(original_text_splitted_into_tokens)

        new_text = original_text_splitted_into_tokens.copy()
        whitespace_ind_difference = 0

        delete_applicable = sum([1 if x.isalpha() else 0 for x in new_text]) > 0 # apply delete punctuation if sure that whole text is not deleted
        for token_ind, token in enumerate(original_text_splitted_into_tokens):
            if token_ind == len(text) - 1:
                applicability_place = 'eos'
            else:
                applicability_place = 'middle'

            # Insert
            if np.random.uniform(0, 1) < self.punct_errors_aggregated_probs[applicability_place]['I']:
                # select one of punctuation-tokens according to its distribution
                punct_token = np.random.choice(list(self.punct_errors_detailed_probs[applicability_place]['I'].keys()),
                                               p=list(self.punct_errors_detailed_probs[applicability_place]['I'].values()))

                punct_token = punct_token.replace(" ", "")

                # do not insert anything before the first token (and do not insert anything after the last token)
                if (token_ind == 0 and num_tokens_in_original_text > 1) or (
                            (token_ind != num_tokens_in_original_text - 1) and np.random.uniform(0, 1) < 0.5):
                    new_text[
                        token_ind] = token + " " + punct_token

                    # if are about to insert "final-punctuation" token, we need to upper-case the following token
                    if punct_token in self.final_punctuation_marks:
                        new_text[token_ind + 1] = new_text[token_ind + 1][0].upper() + new_text[token_ind + 1][1:]

                    if punct_token in punct_tokens_that_append_to_next_token:
                        whitespace_info.insert(token_ind + 1 + whitespace_ind_difference,
                                               False)  # ] = ['I', False, whitespace_info[token_ind + 1]] # .insert(token_ind + 1, False)
                    else:
                        whitespace_info.insert(token_ind + whitespace_ind_difference,
                                               False)  # ] = ['I', False, whitespace_info[token_ind]] # .insert(token_ind, False)

                else:
                    new_text[token_ind] = punct_token + " "

                    if punct_token in self.final_punctuation_marks:
                        new_text[token_ind] += token[0].upper() + token[1:]
                    else:
                        new_text[token_ind] += token

                    if punct_token in punct_tokens_that_append_to_next_token:
                        whitespace_info.insert(token_ind + whitespace_ind_difference,
                                               False)  # ] = ['I', False, whitespace_info[token_ind]] # .insert(token_ind, False)
                    else:
                        whitespace_info.insert(token_ind - 1 + whitespace_ind_difference,
                                               False)  # ] = ['I', False, whitespace_info[token_ind - 1]] # .insert(token_ind - 1, False)

                whitespace_ind_difference += 1
                changes.append(['PUNCT', 'insert {} around {}'.format(punct_token, token)])

            # Delete
            elif delete_applicable and token in self.punct_errors_detailed_probs[applicability_place]['D'] and np.random.uniform(0, 1) < \
                    self.punct_errors_detailed_probs[applicability_place]['D'][token]:
                # if we are about to delete a "final-punctuation" token, we need to lower-case the following letter
                if token in self.final_punctuation_marks and token_ind < num_tokens_in_original_text - 1:
                    new_text[token_ind + 1] = new_text[token_ind + 1][0].lower() + new_text[token_ind + 1][1:]

                new_text[token_ind] = ''

                if token in punct_tokens_that_append_to_next_token and token_ind < num_tokens_in_original_text - 1:
                    del whitespace_info[token_ind + whitespace_ind_difference]
                else:
                    del whitespace_info[token_ind - 1 + whitespace_ind_difference]

                whitespace_ind_difference -= 1
                changes.append(
                    ['PUNCT',
                     'delete {} around {}'.format(token, " ".join(original_text_splitted_into_tokens[token_ind - 4: token_ind + 4]))])

            # Substitute
            elif token in self.punct_errors_detailed_probs[applicability_place]['S'] and np.random.uniform(0, 1) < \
                    self.punct_errors_aggregated_probs[applicability_place]['S'][token]:
                replace_token = np.random.choice(list(self.punct_errors_detailed_probs[applicability_place]['S'][token].keys()),
                                                 p=list(self.punct_errors_detailed_probs[applicability_place]['S'][token].values()))

                new_text[token_ind] = replace_token

                # if we substituted a "final-punctuation" token into "non-final-punctuation", or vica versa, we need to change casing of the following letter
                if token_ind >= num_tokens_in_original_text - 1:
                    break

                if token in self.final_punctuation_marks and replace_token not in self.final_punctuation_marks:
                    new_text[token_ind + 1] = new_text[token_ind + 1][0].lower() + new_text[token_ind + 1][1:]
                elif replace_token in self.final_punctuation_marks and token not in self.final_punctuation_marks:
                    new_text[token_ind + 1] = new_text[token_ind + 1][0].upper() + new_text[token_ind + 1][1:]

                # if we substituted "normal" punctuation token (that appends to the token on the left) with one of punct_tokens_that_append_to_next_token,
                # we need to swap two adjacent info in whitespace_info (or vica versa)
                if (token in punct_tokens_that_append_to_next_token and replace_token not in punct_tokens_that_append_to_next_token) or (
                                token not in punct_tokens_that_append_to_next_token and replace_token in punct_tokens_that_append_to_next_token):
                    if token_ind > 0:
                        whitespace_info[token_ind - 1 + whitespace_ind_difference] = not whitespace_info[
                            token_ind - 1 + whitespace_ind_difference]

                    if token_ind < len(original_text_splitted_into_tokens) - 1:
                        whitespace_info[token_ind + whitespace_ind_difference] = not whitespace_info[
                            token_ind + whitespace_ind_difference]

                changes.append(['PUNCT', 'replace {} with {}'.format(token, replace_token)])

        return " ".join([x for x in new_text if x]), changes, whitespace_info  # ignore empty (deleted) tokens

    @staticmethod
    def _get_punctuation_probabilities(punct_errors, coder_dicts):
        punct_errors_chars_only = {}
        for applicability_place, v in punct_errors.items():
            punct_errors_chars_only[applicability_place] = {}
            for op_type, v1 in punct_errors[applicability_place].items():
                if op_type == 'I' or op_type == 'D':
                    punct_errors_chars_only[applicability_place][op_type] = Counter(list(map(lambda x: x[0], v1)))
                else:  # S
                    punct_errors_chars_only[applicability_place][op_type] = {}
                    for tok in v1:
                        token_from, token_to = tok[0].split('SEPARATOR')
                        if token_from not in punct_errors_chars_only[applicability_place][op_type]:
                            punct_errors_chars_only[applicability_place][op_type][token_from] = Counter()

                        punct_errors_chars_only[applicability_place][op_type][token_from][token_to] += 1

        # # verbose dump
        # for applicability_place, v in punct_errors.items():
        #     print(applicability_place)
        #     for op_type, v1 in punct_errors[applicability_place].items():
        #         print(op_type)
        #         if op_type == 'I' or op_type == 'D':
        #             for k, v in punct_errors_chars_only[applicability_place][op_type].most_common():
        #                 print(k, v)
        #         else:
        #             for token_from in punct_errors_chars_only[applicability_place][op_type]:
        #                 print(token_from)
        #                 for k,v in punct_errors_chars_only[applicability_place][op_type][token_from].most_common():
        #                     print("   -> {} {}".format(k,v))


        num_applicable = {
            'eos': {
                'I': 0,  # num_para
                'D': {},
                'S': {}
            },
            'middle': {
                'I': 0,  # num_tokens - 1 (last one is separated)
                'D': {},
                'S': {}
            },
        }

        for coder_dict in coder_dicts:  # coder dict for each corrected sentence (paragraph)
            if coder_dict:
                coder_id = list(coder_dict.keys())[0]
                cor_sent = coder_dict[coder_id][0]

                num_applicable['eos']['I'] += 1
                num_applicable['middle']['I'] += len(cor_sent) - 1  # -1 as we do not apply it on the last token

                for tok in cor_sent:
                    for applicability_place in punct_errors_chars_only:  # for each applicability type
                        for op_type in ['S', 'D']:
                            if tok in punct_errors_chars_only[applicability_place][op_type]:
                                if tok not in num_applicable[applicability_place][op_type]:
                                    num_applicable[applicability_place][op_type][tok] = 0

                                num_applicable[applicability_place][op_type][tok] += 1

        # create detailed (up to [applicability_place][op_type][token_from][token_to]) probability dictionary
        punct_errors_detailed_probs = {}
        for applicability_place in punct_errors_chars_only:  # for each applicability type
            punct_errors_detailed_probs[applicability_place] = {}

            for op_type in ['I', 'D', 'S']:
                punct_errors_detailed_probs[applicability_place][op_type] = {}

            op_type = 'I'
            for tok in punct_errors_chars_only[applicability_place][op_type]:
                punct_errors_detailed_probs[applicability_place][op_type][tok] = punct_errors_chars_only[applicability_place][op_type][
                                                                                     tok] / \
                                                                                 num_applicable[applicability_place][op_type]

            op_type = 'D'
            for tok in punct_errors_chars_only[applicability_place][op_type]:
                punct_errors_detailed_probs[applicability_place][op_type][tok] = punct_errors_chars_only[applicability_place][op_type][
                                                                                     tok] / \
                                                                                 num_applicable[applicability_place][op_type][tok]
            op_type = 'S'
            for tok in punct_errors[applicability_place][op_type]:
                token_from, token_to = tok[0].split('SEPARATOR')
                if token_from not in punct_errors_detailed_probs[applicability_place][op_type]:
                    punct_errors_detailed_probs[applicability_place][op_type][token_from] = {}

                if token_from in num_applicable[applicability_place][op_type]:
                    punct_errors_detailed_probs[applicability_place][op_type][token_from][token_to] = \
                        punct_errors_chars_only[applicability_place][op_type][token_from][token_to] / \
                        num_applicable[applicability_place][op_type][token_from]

        # create shallow (up to [applicability_place][op_type]) probability dictionary
        punct_errors_aggregated_probs = {}
        for applicability_place in punct_errors_detailed_probs:  # for each applicability type
            punct_errors_aggregated_probs[applicability_place] = {}

            punct_errors_aggregated_probs[applicability_place]['I'] = np.sum(
                list(punct_errors_detailed_probs[applicability_place]['I'].values()))
            punct_errors_aggregated_probs[applicability_place]['S'] = {}

            for token_from in punct_errors_detailed_probs[applicability_place]['S']:
                punct_errors_aggregated_probs[applicability_place]['S'][token_from] = np.sum(
                    list(punct_errors_detailed_probs[applicability_place]['S'][token_from].values()))

        # normalize insert and substitute detailed probabilities to sum up to 1
        for applicability_place in punct_errors_chars_only:  # for each applicability type
            # insert
            for token in punct_errors_detailed_probs[applicability_place]['I']:
                punct_errors_detailed_probs[applicability_place]['I'][token] /= punct_errors_aggregated_probs[applicability_place]['I']

            # substitute
            for token_from in punct_errors_detailed_probs[applicability_place]['S']:
                for token_to in punct_errors_detailed_probs[applicability_place]['S'][token_from]:
                    punct_errors_detailed_probs[applicability_place]['S'][token_from][token_to] /= \
                        punct_errors_aggregated_probs[applicability_place]['S'][token_from]

        return punct_errors_aggregated_probs, punct_errors_detailed_probs

    @staticmethod
    def _get_operation_type_and_char(edit, orig):
        final_punctuation_marks = ['.', '!', '?']
        quotation_marks = ["\"", "„"]

        orig_start, orig_end, error_type, cor_tok, cor_start, cor_end = edit

        if orig_start == orig_end:
            op_type = 'I'

            if len(cor_tok.split(' ')) > 1:
                # TODO this is mainly because of emoji, which are now splitted into : - ( (three symbols), fix this after solving tokenisation
                return None, None, None

            char = cor_tok
            next_token_change_casing = False

        elif cor_start == cor_end:
            op_type = 'D'
            char = " ".join(orig[orig_start:orig_end])
            next_token_change_casing = False
        else:
            '''
            Punctuation errors are by ERRANT sometimes merged with the following word (if it makes sense as a single error, handle these special
            cases prior to defining this error type as substitute.

            e.g.
            word -> . Word (insert dot and upper-case next token)
            '''

            # insert eos (any from final_punctuation_marks) (e.g. monkey -> . Monkey)
            if orig_end - orig_start == 1 and len(cor_tok.split()) == 2 and orig[orig_start][0].islower() and \
                            cor_tok.split()[0] in final_punctuation_marks and cor_tok.split()[1][0].isupper():
                op_type = 'I'
                char = cor_tok.split()[0]
                next_token_change_casing = True
            # delete . (e.g. . Monkey -> monkey)
            elif orig_end - orig_start == 2 and len(cor_tok.split()) == 1 and orig[orig_start] in final_punctuation_marks and \
                    orig[orig_start + 1][0].isupper() and cor_tok[0].islower():
                op_type = 'D'
                char = orig[orig_start]
                next_token_change_casing = True
            else:
                op_type = 'S'

                # several hand-crafted rules
                # 1. , because -> . Because
                if orig_end - orig_start == 2 and len(cor_tok.split()) == 2 and orig[orig_start] == ',' and orig[orig_start + 1][
                    0].islower() \
                        and cor_tok.split()[0] == '.' and cor_tok.split()[1][0].isupper():
                    char = ',' + "SEPARATOR" + '.'
                    next_token_change_casing = True
                # 2. . Because -> , because
                elif orig_end - orig_start == 2 and len(cor_tok.split()) == 2 and orig[orig_start] == '.' and orig[orig_start + 1][
                    0].isupper() \
                        and cor_tok.split()[0] == ',' and cor_tok.split()[1][0].islower():
                    char = '.' + "SEPARATOR" + ','
                    next_token_change_casing = True
                # 3. . Because -> ? because (both punctuation and casing)
                elif orig_end - orig_start == 2 and len(cor_tok.split()) == 2 and orig[orig_start] in final_punctuation_marks and \
                                cor_tok.split()[0] in final_punctuation_marks and orig[orig_start + 1].lower() == cor_tok.split()[
                    1].lower():
                    char = orig[orig_start] + "SEPARATOR" + cor_tok.split()[0]
                    next_token_change_casing = True
                # 4. " Car -> „ car (Czech phenomenon)
                elif orig_end - orig_start == 2 and len(cor_tok.split()) == 2 and orig[orig_start] in quotation_marks and \
                                cor_tok.split()[0] in quotation_marks and orig[orig_start + 1].lower() == cor_tok.split()[1].lower():
                    char = orig[orig_start] + "SEPARATOR" + cor_tok.split()[0]
                    next_token_change_casing = True
                # In all other cases, we cannot be more specific, but for the sake of simplicity, we allow only 1:1 punctuation substititions
                elif orig_end - orig_start == 1 and cor_end - cor_start == 1:
                    char = orig[orig_start] + "SEPARATOR" + cor_tok
                    next_token_change_casing = False
                else:
                    # TODO …---. . . (tri tecky za znak trojtecky?)
                    return None, None, None

        # ! Note that currently we go in direction orig -> cor, but in fact, our noising script is supposed to run in opposite direction
        if op_type == 'I':
            op_type = 'D'
        elif op_type == 'D':
            op_type = 'I'
        elif op_type == 'S':
            orig, cor = char.split('SEPARATOR')
            char = cor + 'SEPARATOR' + orig

        if char == '; - )':
            print('WTF', edit)
        return op_type, char, next_token_change_casing

    @staticmethod
    def estimate_probabilities(m2_records):
        '''
            We differentiate between punctuation errors on the end of the text and in the middle of the text.
            '''
        punct_errors = {
            'eos': {
                'I': [],
                'D': [],
                'S': []
            },
            'middle': {
                'I': [],
                'D': [],
                'S': []
            },
        }

        num_files = 0  # where eos is applicable
        num_tokens = 0  # where middle is applicable

        cached_edits = []

        for m2_file in m2_records:
            for info in m2_file:
                orig_sent, coder_dict = apply_m2_edits.processM2(info, [])

                cached_edits.append(coder_dict)

                if coder_dict:
                    num_files += 1
                    coder_id = list(coder_dict.keys())[0]
                    cor_sent = coder_dict[coder_id][0]
                    num_tokens += len(cor_sent)

                    for edit in coder_dict[coder_id][1]:
                        orig_start, orig_end, error_type, cor_tok, cor_start, cor_end = edit

                        if 'PUNCT' in error_type:
                            if cor_end >= len(cor_sent):
                                applicability_place = 'eos'
                            else:
                                applicability_place = 'middle'

                            op_type, char, next_token_change_casing = Punctuation._get_operation_type_and_char(edit, orig_sent)

                            if op_type:
                                punct_errors[applicability_place][op_type].append([char, next_token_change_casing])

        punct_errors_aggregated_probs, punct_errors_detailed_probs = Punctuation._get_punctuation_probabilities(punct_errors, cached_edits)

        return 'punctuation', {
            'punct_errors_aggregated_probs': punct_errors_aggregated_probs,
            'punct_errors_detailed_probs': punct_errors_detailed_probs
        }
