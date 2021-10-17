import numpy as np
from aspects import apply_m2_edits
from aspects import utils
from aspects.base import Aspect
from aspects.diacritization_stripping import strip_diacritics_single_line


class Diacritics(Aspect):
    def __init__(self, profile, lang, alpha=1, beta=0):
        super(Diacritics, self).__init__(profile, alpha, beta)

        all_wo_diacritics_perc = profile['diacritics']['all_wo_diacritics_perc']
        self.all_wo_diacritics_perc = utils._apply_smoothing([all_wo_diacritics_perc], alpha, beta)[0]

        wrong_char_diacritics_perc = profile['diacritics']['wrong_char_diacritics_perc']
        self.wrong_char_diacritics_perc = utils._apply_smoothing([wrong_char_diacritics_perc], alpha, beta)[0]

        wrongly_diacritized_chars_probs = profile['diacritics']['wrongly_diacritized_chars_probs']
        self.wrongly_diacritized_chars_probs = {
            k: utils._apply_smoothing_on_simple_dict(wrongly_diacritized_chars_probs[k], alpha, beta) for k in
            wrongly_diacritized_chars_probs}

    def apply(self, text, whitespace_info):
        changes = []

        if strip_diacritics_single_line(text) != text and np.random.uniform(0, 1) < self.all_wo_diacritics_perc:
            changes.append(['DIACR', 'all_strip_diacritics'])
            return strip_diacritics_single_line(text), changes, whitespace_info

        new_text = ''

        for c in text:
            if c in self.wrongly_diacritized_chars_probs:
                if np.random.uniform(0, 1) < self.wrong_char_diacritics_perc:
                    new_text += np.random.choice(list(self.wrongly_diacritized_chars_probs[c].keys()),
                                                 p=list(self.wrongly_diacritized_chars_probs[c].values()))
                else:
                    new_text += c
            elif c.lower() in self.wrongly_diacritized_chars_probs:
                if np.random.uniform(0, 1) < self.wrong_char_diacritics_perc:
                    new_text += np.random.choice(list(self.wrongly_diacritized_chars_probs[c.lower()].keys()),
                                                 p=list(self.wrongly_diacritized_chars_probs[c.lower()].values())).upper()
                else:
                    new_text += c
            elif c.upper() in self.wrongly_diacritized_chars_probs:
                if np.random.uniform(0, 1) < self.wrong_char_diacritics_perc:
                    new_text += np.random.choice(list(self.wrongly_diacritized_chars_probs[c.upper()].keys()),
                                                 p=list(self.wrongly_diacritized_chars_probs[c.upper()].values())).lower()
                else:
                    new_text += c
            else:
                new_text += c

            if new_text[-1] != c:
                changes.append(['DIACR', 'replace {} with {}'.format(c, new_text[-1])])

        return new_text, changes, whitespace_info

    @staticmethod
    def estimate_probabilities(m2_records):

        czech_diacritics_tuples = [('a', 'á'), ('c', 'č'), ('d', 'ď'), ('e', 'é', 'ě'), ('i', 'í'), ('n', 'ň'), ('o', 'ó'), ('r', 'ř'),
                                   ('s', 'š'), ('t', 'ť'), ('u', 'ů', 'ú'), ('y', 'ý'), ('z', 'ž')]
        czech_diacritizables_chars = [char for sublist in czech_diacritics_tuples for char in sublist] + [char.upper() for sublist in
                                                                                                          czech_diacritics_tuples for char
                                                                                                          in sublist]
        num_all_wo_diacritics = 0
        num_files = 0
        num_badly_diacritized_chars = 0
        num_could_be_diacritized_char = 0

        wrongly_diacritized_chars_map = {}

        for m2_file in m2_records:
            num_files += 1

            original_paragraphs, corrected_paragraphs_wo_diacr, corrected_paragraphs = [], [], []
            for info in m2_file:
                # Get the original and corrected sentence + edits for each annotator.
                orig_sent, coder_dict = apply_m2_edits.processM2(info, ["DIACR"])
                orig_sent = " ".join(orig_sent)
                if coder_dict:
                    coder_id = list(coder_dict.keys())[0]
                    cor_sent = " ".join(coder_dict[coder_id][0])
                else:
                    cor_sent = orig_sent

                original_paragraphs.append(orig_sent)
                corrected_paragraphs_wo_diacr.append(cor_sent)

                _, coder_dict = apply_m2_edits.processM2(info, [])
                if coder_dict:
                    coder_id = list(coder_dict.keys())[0]
                    cor_sent = " ".join(coder_dict[coder_id][0])
                else:
                    cor_sent = orig_sent
                corrected_paragraphs.append(cor_sent)

            this_text_is_all_wo_diacritics_and_should_contain_some = False
            # if original sentence does not contain diacritics
            if strip_diacritics_single_line(" ".join(original_paragraphs)) == " ".join(original_paragraphs):
                # and there is a change in diacritics
                if (corrected_paragraphs_wo_diacr != corrected_paragraphs):
                    num_all_wo_diacritics += 1
                    this_text_is_all_wo_diacritics_and_should_contain_some = True

            # individual characters
            if not this_text_is_all_wo_diacritics_and_should_contain_some:
                for p_orig, p_cor in zip(corrected_paragraphs_wo_diacr, corrected_paragraphs):
                    for c_orig, c_cor in zip(p_orig, p_cor):
                        if c_orig != c_cor and strip_diacritics_single_line(c_orig) == strip_diacritics_single_line(c_cor):
                            num_badly_diacritized_chars += 1
                            num_could_be_diacritized_char += 1

                            if c_cor not in wrongly_diacritized_chars_map:
                                wrongly_diacritized_chars_map[c_cor] = {}

                            if c_orig not in wrongly_diacritized_chars_map[c_cor]:
                                wrongly_diacritized_chars_map[c_cor][c_orig] = 0

                            wrongly_diacritized_chars_map[c_cor][c_orig] += 1
                        elif c_cor in czech_diacritizables_chars:
                            num_could_be_diacritized_char += 1

        all_wo_diacritics_perc = num_all_wo_diacritics / num_files
        wrong_char_diacritics_perc = num_badly_diacritized_chars / num_could_be_diacritized_char

        # filter out characters whose correction in diacritics appeared too little (<3)
        filtered_wrongly_diacritized_chars_map = {}
        for c_cor in wrongly_diacritized_chars_map:
            if np.sum(list(wrongly_diacritized_chars_map[c_cor].values())) > 3:
                filtered_wrongly_diacritized_chars_map[c_cor] = {c_orig: c_orig_value for c_orig, c_orig_value in
                                                                 wrongly_diacritized_chars_map[c_cor].items()}

        # normalize wrongly_diacritized_chars_map
        wrongly_diacritized_chars_probs = {}
        for c_cor in filtered_wrongly_diacritized_chars_map:
            normalize_sum = np.sum(list(filtered_wrongly_diacritized_chars_map[c_cor].values()))
            wrongly_diacritized_chars_probs[c_cor] = {c_orig: c_orig_value / normalize_sum for c_orig, c_orig_value in
                                                      filtered_wrongly_diacritized_chars_map[c_cor].items()}

        return 'diacritics', {'all_wo_diacritics_perc': all_wo_diacritics_perc,
                              'wrong_char_diacritics_perc': wrong_char_diacritics_perc,
                              'wrongly_diacritized_chars_probs': wrongly_diacritized_chars_probs
                              }
