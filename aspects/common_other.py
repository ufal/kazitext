from aspects import apply_m2_edits
import numpy as np
from aspects.base import Aspect
from aspects import utils


class CommonOther(Aspect):
    def __init__(self, profile, lang, alpha=1, beta=0):
        super(CommonOther, self).__init__(profile, alpha, beta)

        common_other_all_pairs_probs = profile['common_other']['all_pairs_probs']

        self.common_other_all_pairs_probs = {}
        for cor_from in common_other_all_pairs_probs:
            self.common_other_all_pairs_probs[cor_from] = utils._apply_smoothing_on_simple_dict(common_other_all_pairs_probs[cor_from],
                                                                                                alpha, beta)

    def apply(self, text, whitespace_info):
        def _get_occurence_start_indices_of_tokens_in_text(text, substring):
            '''
            Return all occurences of substring in text, but make sure that each occurence of substring is bordered by non-alpha characters, so
            that substring is not a part of another word (e.g. for substring "se" we do not want to count its occurence in text "prase")
            '''
            occurence_start_indices = []
            last_occurence_start_index = -1
            while substring in text[last_occurence_start_index + 1:]:
                start_index = text.index(substring, last_occurence_start_index + 1)

                if (start_index == 0 or (start_index != 0 and [start_index - 1] == ' ')) and (
                            (start_index + len(substring) == len(text)) or text[start_index + len(substring)] == ' '):
                    occurence_start_indices.append(start_index)

                last_occurence_start_index = start_index

            return occurence_start_indices

        changes = []
        '''
        Go over all keys (corrected tokens) in all_pairs_probs and try to apply each of them on its each occurence in text.
        '''
        # substitutes / deletes
        for cor_tok in self.common_other_all_pairs_probs:
            if not cor_tok:  # insertions are done separately
                continue

            occurence_start_indices = _get_occurence_start_indices_of_tokens_in_text(text.lower(), cor_tok)
            if len(occurence_start_indices) > 0:
                char_relative_change = 0
                for start_index in occurence_start_indices:
                    start_index = start_index + char_relative_change
                    if np.random.uniform(0, 1) < np.sum(list(self.common_other_all_pairs_probs[cor_tok].values())):
                        replace_tokens_probs = np.array(list(self.common_other_all_pairs_probs[cor_tok].values()))
                        replace_tokens_probs_normalized = replace_tokens_probs / np.sum(replace_tokens_probs)

                        chosen_replace_tokens = np.random.choice(list(self.common_other_all_pairs_probs[cor_tok].keys()),
                                                                 p=replace_tokens_probs_normalized)

                        # if this is delete and would delete whole text, do not perform it
                        if len(chosen_replace_tokens) == 0 and len(cor_tok) == len(text):
                            continue

                        if text[start_index].isupper() and len(chosen_replace_tokens) > 0:
                            chosen_replace_tokens = chosen_replace_tokens[0].upper() + chosen_replace_tokens[1:]

                        if len(chosen_replace_tokens) == 0:  # delete
                            if start_index == 0:
                                text = text[start_index + 1 + len(cor_tok):]
                            else:
                                text = text[:start_index - 1] + chosen_replace_tokens + text[start_index + len(cor_tok):]
                        else:
                            text = text[:start_index] + chosen_replace_tokens + text[start_index + len(cor_tok):]

                        # fix whitespace info
                        num_tokens_in_correct = len(cor_tok.split(' '))
                        num_tokens_in_noised = len(chosen_replace_tokens.split(' ')) if chosen_replace_tokens else 0

                        start_index_token_num = np.sum([1 for x in text[:start_index] if x == ' '], dtype=np.int32)
                        if num_tokens_in_correct > num_tokens_in_noised:
                            for _ in range(num_tokens_in_correct - num_tokens_in_noised):
                                del whitespace_info[start_index_token_num]
                        elif num_tokens_in_correct < num_tokens_in_noised:
                            for _ in range(num_tokens_in_noised - num_tokens_in_correct):
                                whitespace_info.insert(start_index_token_num, True)

                        char_relative_change += len(chosen_replace_tokens) - len(cor_tok)
                        if len(chosen_replace_tokens) == 0:
                            char_relative_change -= 1  # if we delete a token, we also remove one space next to it

                        if len(chosen_replace_tokens) == 0:
                            changes.append(['COMMON-OTHER', 'delete {}'.format(cor_tok)])
                        else:
                            changes.append(['COMMON-OTHER', 'change {} -> {}'.format(cor_tok, chosen_replace_tokens)])

        # inserts
        insert_into_whitespace = [None] * len(whitespace_info)
        if '' in self.common_other_all_pairs_probs:
            for whitespace_ind in range(len(insert_into_whitespace)):
                for tokens_to in np.random.permutation(
                        list(self.common_other_all_pairs_probs[''].keys())):  # do permutation to allow all tokens when alpha-smoothing is high
                    if np.random.uniform(0, 1) < self.common_other_all_pairs_probs[''][tokens_to]:
                        insert_into_whitespace[whitespace_ind] = tokens_to
                        break  # do just one insert per each whitespace

        num_inserted_spaces = 0
        new_text = ""
        for token_ind, token in enumerate(text.split(' ')):
            if token_ind != 0:
                new_text += " "

            new_text += token

            if token_ind < len(insert_into_whitespace) and insert_into_whitespace[token_ind]:
                new_text += " " + insert_into_whitespace[token_ind]

                for _ in range(len(insert_into_whitespace[token_ind].split(' '))):
                    whitespace_info.insert(token_ind + num_inserted_spaces, True)

                num_inserted_spaces += len(insert_into_whitespace[token_ind].split(' '))

                changes.append(['COMMON-OTHER', 'insert {}'.format(insert_into_whitespace[token_ind])])

        return new_text, changes, whitespace_info

    @staticmethod
    def _get_occurence_count_of_tokens_in_text(text, substring):
        '''
        Get all occurences of substring in text, but make sure that each occurence of substring is bordered by non-alpha characters, so that
        substring is not a part of another word (e.g. for substring "se" we do not want to count its occurence in text "prase")
        '''
        num_occurences_of_cor_tok_in_cor = 0
        last_occurence_start_index = -1
        while substring in text[last_occurence_start_index + 1:]:
            start_index = text.index(substring, last_occurence_start_index + 1)

            if (start_index == 0 or (start_index != 0 and not text[start_index - 1].isalpha())) and (
                        (start_index + len(substring) == len(text)) or not text[start_index + len(substring)].isalpha()):
                num_occurences_of_cor_tok_in_cor += 1

            last_occurence_start_index = start_index

        return num_occurences_of_cor_tok_in_cor

    @staticmethod
    def _get_occurence_counts(cor_toks, coder_dicts):
        num_occurence = {}

        for coder_dict in coder_dicts:
            if coder_dict:
                coder_id = list(coder_dict.keys())[0]

                cor_sent = coder_dict[coder_id][0]

                cor_sent = " ".join(cor_sent).lower()

                for cor_tok in cor_toks:
                    if cor_tok not in num_occurence:
                        num_occurence[cor_tok] = 0

                    if not cor_tok.strip():  # replace empty string with some tokens (= insert)
                        num_occurence[cor_tok] += len(cor_sent.split(' ')) + 1  # number of places where the token could be inserted
                    else:
                        num_occurence[cor_tok] += CommonOther._get_occurence_count_of_tokens_in_text(cor_sent, cor_tok)
        return num_occurence

    @staticmethod
    def estimate_probabilities(m2_records):
        num_files = 0
        exclude_error_types = ['noop', 'PUNCT', 'CASING']
        num_all_alpha_words = 0

        all_pairs = {}
        cached_coder_dicts = []

        for m2_file in m2_records:
            num_files += 1

            for info in m2_file:
                orig_sent, coder_dict = apply_m2_edits.processM2(info, [])
                if coder_dict:
                    cached_coder_dicts.append(coder_dict)
                    coder_id = list(coder_dict.keys())[0]

                    for edit in coder_dict[coder_id][1]:
                        orig_start, orig_end, error_type, cor_tok, cor_start, cor_end = edit
                        cor_tok = cor_tok.lower()

                        if all([x not in error_type for x in exclude_error_types]):
                            orig_toks = " ".join(orig_sent[orig_start:orig_end]).lower()

                            # # check that this error is not already handled by suffix/prefix estimator
                            # if (orig_end - orig_start == 0) or (cor_end - cor_start == 0) or (
                            #                 cor_tok[0] != orig_toks[0] and cor_tok[-1] != cor_tok[-1]) or (orig_end - orig_start) > 1:

                            if cor_tok not in all_pairs:
                                all_pairs[cor_tok] = {}

                            if orig_toks not in all_pairs[cor_tok]:
                                all_pairs[cor_tok][orig_toks] = 0

                            all_pairs[cor_tok][orig_toks] += 1

                num_all_alpha_words += sum([1 for x in orig_sent if x.isalpha()])

        num_occurence = CommonOther._get_occurence_counts(list(all_pairs.keys()), cached_coder_dicts)

        all_pairs_probs = {}
        filter_out_min_occ_count = 3
        for cor_toks in all_pairs:
            if np.sum(list(all_pairs[cor_toks].values())) >= filter_out_min_occ_count:
                all_pairs_probs[cor_toks] = {}
                if not cor_toks.strip():
                    for orig_toks in all_pairs[cor_toks]:
                        if all_pairs[cor_toks][orig_toks] >= filter_out_min_occ_count:
                            all_pairs_probs[cor_toks][orig_toks] = all_pairs[cor_toks][orig_toks] / num_all_alpha_words
                else:
                    for orig_toks in all_pairs[cor_toks]:
                        if all_pairs[cor_toks][orig_toks] >= filter_out_min_occ_count:
                            all_pairs_probs[cor_toks][orig_toks] = all_pairs[cor_toks][orig_toks] / num_occurence[cor_toks]

                if len(all_pairs_probs[cor_toks]) == 0:
                    del all_pairs_probs[cor_toks]

        return 'common_other', {
            "all_pairs_probs": all_pairs_probs
        }
