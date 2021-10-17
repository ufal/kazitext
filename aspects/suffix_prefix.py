import numpy as np

from aspects import apply_m2_edits
from aspects.base import Aspect
from aspects import utils


class SuffixPrefix(Aspect):
    def __init__(self, profile, lang, alpha=1, beta=0):
        super(SuffixPrefix, self).__init__(profile, alpha, beta)

        self.suffix_table = profile['suffix_prefix']['suffix_table']
        self.suffix_occurence_counts = profile['suffix_prefix']['suffix_occurence_counts']
        self.prefix_table = profile['suffix_prefix']['prefix_table']
        self.prefix_occurence_counts = profile['suffix_prefix']['prefix_occurence_counts']

        # alfa-self.beta smoothing must be done inside each call (because of counts instead of probabilities)
        self.alpha = alpha
        self.beta = beta

    def apply(self, text, whitespace_info):
        def _introduce_suffix_errors(local_text, suffix_table, suffix_occurence_counts):
            new_text = []
            changes = []
            # first introduce suffix errors
            for word in local_text.split(' '):
                word_suffixes = [word[i:] for i in range(1, len(word))] + [""]
                found_word_suffixes = []
                word_suffixes_probs = []

                sum_rewrites_that_go_on = 0
                last_match_sum_applicable = 0
                for word_suffix in word_suffixes:
                    if word_suffix in suffix_table:
                        found_word_suffixes.append(word_suffix)
                        sum_rewrites_in_data = np.sum(list(suffix_table[word_suffix].values())) - sum_rewrites_that_go_on
                        sum_applicable_in_data = suffix_occurence_counts[word_suffix] - last_match_sum_applicable

                        sum_rewrites_that_go_on = 0
                        if word_suffix:  # if empty, this is an inserting of the suffix and is the last element of the for-cycle
                            for word_suffix_alternative in suffix_table[word_suffix]:
                                if len(word_suffix_alternative) > 0 and word_suffix_alternative[0] == word_suffix[0]:
                                    sum_rewrites_that_go_on += suffix_table[word_suffix][word_suffix_alternative]

                        last_match_sum_applicable = sum_applicable_in_data

                        if sum_applicable_in_data == 0:
                            word_suffixes_probs.append(0)
                        else:
                            word_suffixes_probs.append(sum_rewrites_in_data / sum_applicable_in_data)

                if not word_suffixes_probs:
                    # no edit is applicable
                    new_text.append(word)
                    continue

                # select suffix (according to probability distribution)
                word_suffixes_probs = np.array(word_suffixes_probs)
                word_suffixes_probs_normalized = word_suffixes_probs / np.sum(word_suffixes_probs)
                word_suffixes_probs_normalized_smoothed = utils._apply_smoothing(word_suffixes_probs_normalized, self.alpha, self.beta)
                chosen_suffix_ind = np.random.choice(len(found_word_suffixes), p=word_suffixes_probs_normalized_smoothed)

                word_suffixes_probs_smoothed = utils._apply_smoothing(word_suffixes_probs, self.alpha, self.beta)
                chosen_suffix, chosen_suffix_sum_prob = found_word_suffixes[chosen_suffix_ind], word_suffixes_probs_smoothed[
                    chosen_suffix_ind]

                # toss a coin for the chosen suffix
                chosen_suffix_sum_prob_smoothed = utils._apply_smoothing([chosen_suffix_sum_prob], self.alpha, self.beta)[0]
                if np.random.uniform(0, 1) < chosen_suffix_sum_prob_smoothed:
                    # choose what to rewrite the suffix into
                    rewrite_into_probs = np.array(list(suffix_table[chosen_suffix].values())) / np.sum(
                        np.array(list(suffix_table[chosen_suffix].values())))

                    rewrite_into_probs_smoothed = utils._apply_smoothing(rewrite_into_probs, self.alpha, self.beta)
                    chosen_rewrite_into_tokens = np.random.choice(list(suffix_table[chosen_suffix].keys()), p=rewrite_into_probs_smoothed)

                    if len(chosen_suffix) == 0:  # inserting suffix after this word
                        new_word = word + chosen_rewrite_into_tokens
                    else:
                        new_word = word[:-len(chosen_suffix)] + chosen_rewrite_into_tokens
                    new_text.append(new_word)
                    changes.append(['SUFFIX', 'change {} -> {}'.format(word, new_word)])
                else:
                    new_text.append(word)

            return " ".join(new_text), changes

        text, suffix_changes, = _introduce_suffix_errors(text, self.suffix_table, self.suffix_occurence_counts)

        text = " ".join(["".join(reversed(word)) for word in text.split(' ')])
        text, prefix_changes = _introduce_suffix_errors(text, self.prefix_table, self.prefix_occurence_counts)
        text = " ".join(["".join(reversed(word)) for word in text.split(' ')])
        return text, suffix_changes + prefix_changes, whitespace_info

    @staticmethod
    def _get_occurence_count_of_tokens_in_text(text, suffix):
        '''
        Get all occurences of suffix in text.

        e.g. ed is suffix in "I have walked"; but is not suffix in "I love reddish"
        '''
        num_occurences_of_suffix_in_cor = 0
        last_occurence_start_index = -1
        while suffix in text[last_occurence_start_index + 1:]:
            start_index = text.index(suffix, last_occurence_start_index + 1)

            # matched suffix is either suffix of whole text or the character after the matched suffix is not self.alpha
            if (start_index + len(suffix) == len(text)) or not text[start_index + len(suffix)].isalpha():
                num_occurences_of_suffix_in_cor += 1

            last_occurence_start_index = start_index

        return num_occurences_of_suffix_in_cor

    @staticmethod
    def _get_xfix_occurence_counts(xfix_table, coder_dicts, reverse=False):
        ''''
        For each possible xfix (in xfix_table), count, how many times this suffix appeared in all corrected sentences (coder_dicts).
        '''
        num_occurence = {}

        for coder_dict in coder_dicts:
            if coder_dict:
                coder_id = list(coder_dict.keys())[0]

                cor_sent = coder_dict[coder_id][0]

                if reverse:
                    cor_sent = ["".join(reversed(word)) for word in cor_sent]

                cor_sent = " ".join(cor_sent).lower()

                for cor_toks_xfix in xfix_table:
                    if cor_toks_xfix not in num_occurence:
                        num_occurence[cor_toks_xfix] = 0

                    if not cor_toks_xfix.strip():  # empty suffix replace with some other (=insert some substring after token)
                        num_alpha_in_sent = sum([1 for x in cor_sent.split(' ') if x.isalpha()])
                        num_occurence[cor_toks_xfix] += num_alpha_in_sent
                    else:
                        num_occurence[cor_toks_xfix] += SuffixPrefix._get_occurence_count_of_tokens_in_text(cor_sent, cor_toks_xfix)

        return num_occurence

    @staticmethod
    def _update_xfix_table(edit, orig_sent, xfix_table):
        '''
        Update suffix/prefix table with all xfix changes that appeared in the edit.

        Although this method is called update_xfix_table, it internally computes everything as suffixes and when used for prefixes, orig_sent
        and cor_tok in edit must be reversed.

        e.g. given edit "hezký" for original token "hezkej", it increments all following pairs: (ezký, ezkej), (zký, zkej), (ký, kej), (ý, ej)
        :param edit:
        :param orig_sent:
        :param xfix_table: dict in format xfix_table[cor_xfix][orig_xfix] = count
        :return:
        '''
        orig_start, orig_end, error_type, cor_toks, _, _ = edit

        # lower down all tokens to cope with lack of data
        orig_toks = " ".join(orig_sent[orig_start:orig_end]).lower()
        cor_toks = cor_toks.lower()

        if orig_toks == cor_toks:
            # this should not happen if data were annotated by Errant
            return

        # # we allow whitespaces only in original (noisy) text and do not allow them in corrected (difficult to apply). Both tokens must be self.alpha
        # if not orig_toks.replace(" ", "").isalpha() or not cor_toks.isalpha():
        #     return

        # we do not allow whitespaces in original (noisy) text and also do not allow them in corrected. Both tokens must be self.alpha
        if not orig_toks.isalpha() or not cor_toks.isalpha():
            return

        # if the first characters are not same, this is not suffix
        if orig_toks[0] != cor_toks[0]:
            return

        i = 1
        end_after_this_iteration = False
        '''
        We iterate through the orig_toks and cor_toks until we pass the first non-matching-block and then immediately stop
        e.g.
        Given orig = akdyž and cor = až, we want to save (když, ž) as frequent pair, but do not want to proceed and do not want to save (dyž, )
        Similarly, orig = hezký, cor = hezkej, we want to save all pairs up to (ý, ej)
        '''

        while True:
            cor_toks_current_suffix = cor_toks[i:]
            orig_toks_current_suffix = orig_toks[i:]

            if len(cor_toks_current_suffix) == 0 or len(orig_toks_current_suffix) == 0 or cor_toks_current_suffix[0] != \
                    orig_toks_current_suffix[0]:
                end_after_this_iteration = True

            if cor_toks_current_suffix not in xfix_table:
                xfix_table[cor_toks_current_suffix] = {}

            if orig_toks_current_suffix not in xfix_table[cor_toks_current_suffix]:
                xfix_table[cor_toks_current_suffix][orig_toks_current_suffix] = 0

            xfix_table[cor_toks_current_suffix][orig_toks_current_suffix] += 1

            i += 1

            if end_after_this_iteration:
                break

    @staticmethod
    def estimate_probabilities(m2_records):
        suffix_table = {}
        prefix_table = {}

        cached_coder_dicts = []
        for m2_file in m2_records:
            for info in m2_file:
                orig_sent, coder_dict = apply_m2_edits.processM2(info, [])
                if coder_dict:
                    coder_id = list(coder_dict.keys())[0]

                    cached_coder_dicts.append(coder_dict)

                    for edit in coder_dict[coder_id][1]:
                        orig_start, orig_end, error_type, cor_tok, cor_start, cor_end = edit

                        exclude_error_types = ['noop', 'PUNCT', 'CASING', 'DIACR', 'UNK']

                        if all([x not in error_type for x in exclude_error_types]):
                            SuffixPrefix._update_xfix_table(edit, orig_sent, suffix_table)

                            # reverse the sentence to update prefix table
                            orig_sent_reversed_words = ["".join(reversed(word)) for word in orig_sent]
                            cor_tok_reversed = " ".join(["".join(reversed(word)) for word in cor_tok.split(' ')])
                            edit = (orig_start, orig_end, error_type, cor_tok_reversed, cor_start, cor_end)

                            SuffixPrefix._update_xfix_table(edit, orig_sent_reversed_words, prefix_table)

        # filter out edits that were done not often enough (are rather noise)
        def filter_table_by_min_occurence_count(table, local_filter_out_min_occ_count):
            table_filtered = {}
            for cor_tok_local in table:
                if np.sum(list(table[cor_tok_local].values())) < local_filter_out_min_occ_count:
                    continue

                for orig_tok in table[cor_tok_local]:
                    if table[cor_tok_local][orig_tok] >= local_filter_out_min_occ_count:
                        if cor_tok_local not in table_filtered:
                            table_filtered[cor_tok_local] = {}

                        table_filtered[cor_tok_local][orig_tok] = table[cor_tok_local][orig_tok]

            return table_filtered

        filter_out_min_occ_count = 3
        suffix_table_filtered = filter_table_by_min_occurence_count(suffix_table, filter_out_min_occ_count)
        prefix_table_filtered = filter_table_by_min_occurence_count(prefix_table, filter_out_min_occ_count)

        suffix_occurence_counts = SuffixPrefix._get_xfix_occurence_counts(suffix_table_filtered, cached_coder_dicts)
        prefix_occurence_counts = SuffixPrefix._get_xfix_occurence_counts(prefix_table_filtered, cached_coder_dicts, reverse=True)

        return 'suffix_prefix', {
            'suffix_table': suffix_table_filtered,
            'suffix_occurence_counts': suffix_occurence_counts,
            'prefix_table': prefix_table_filtered,
            'prefix_occurence_counts': prefix_occurence_counts
        }
