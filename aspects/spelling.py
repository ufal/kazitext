import string
from collections import Counter
from difflib import SequenceMatcher

import aspell
import numpy as np
from aspects import apply_m2_edits, utils
from aspects.base import Aspect
from aspects.utils import get_cheapest_align_seq


class Spelling(Aspect):
    def __init__(self, profile, lang, alpha=1, beta=0):
        super(Spelling, self).__init__(profile, alpha, beta)

        spelling_word_to_invalid_word = profile['spelling']['spelling_word_to_invalid_word']
        spelling_word_to_other_valid_word = profile['spelling']['spelling_word_to_other_valid_word']
        self.spelling_word_to_invalid_word, self.spelling_word_to_other_valid_word = utils._apply_smoothing(
            [spelling_word_to_invalid_word, spelling_word_to_other_valid_word], alpha, beta)

        spelling_noise_operation_probs = profile['spelling']['spelling_noise_operation_probs']
        self.spelling_noise_operation_probs = utils._apply_smoothing_on_simple_dict(spelling_noise_operation_probs, alpha,
                                                                                    beta)

        self.spelling_noise_operation_detailed_probs = profile['spelling']['spelling_noise_operation_detailed_probs']
        self.spelling_noise_operation_detailed_probs = {'D': {}, 'I': {}, 'S': {}}

        for char, char_delete_prob in self.spelling_noise_operation_detailed_probs['D'].items():
            self.spelling_noise_operation_detailed_probs['D'][char] = utils._apply_smoothing([char_delete_prob], alpha, beta)[0]

        for from_char, v in self.spelling_noise_operation_detailed_probs['S'].items():
            self.spelling_noise_operation_detailed_probs['S'][from_char] = utils._apply_smoothing_on_simple_dict(v, alpha,
                                                                                                                 beta)

        for context, v in self.spelling_noise_operation_detailed_probs['I'].items():
            self.spelling_noise_operation_detailed_probs['I'][context] = utils._apply_smoothing_on_simple_dict(v, alpha,
                                                                                                               beta)

        self.aspell_speller = aspell.Speller('lang', lang)
        self.spelling_detailed_ratio = 0.3

        if lang == 'cs':
            czech_chars_with_diacritics = 'áčďěéíňóšřťůúýž'
            self.all_chars_in_language = list(string.ascii_lowercase + czech_chars_with_diacritics) + list(
                string.ascii_lowercase.upper() + czech_chars_with_diacritics.upper())
        elif lang == 'en':
            self.all_chars_in_language = list(string.ascii_lowercase + string.ascii_uppercase)
        elif lang == 'de':
            self.all_chars_in_language = list(string.ascii_lowercase + 'äöüß') + list(string.ascii_uppercase + 'ÄÖÜẞ')
        elif lang == 'ru':
            russian_special = 'бвгджзклмнпрстфхцчшщаэыуояеёюий'
            russian_special += russian_special.upper()
            russian_special += 'ЬьЪъ'
            self.all_chars_in_language = list(russian_special)
        else:
            self.all_chars_in_language = None

    def apply(self, text, whitespace_info):

        changes = []
        new_text = []
        all_alpha_chars_in_text_and_language = set([c for c in text if c.isalpha()])
        if self.all_chars_in_language:
            all_alpha_chars_in_text_and_language.update(self.all_chars_in_language)

        for word in text.split():
            if not word.isalpha():
                new_text.append(word)
                continue

            if np.random.uniform(0, 1) < self.spelling_word_to_other_valid_word:
                top_aspell_suggestions = self.aspell_speller.suggest(word)[:10]

                if word in top_aspell_suggestions:
                    top_aspell_suggestions.remove(word)

                # for some Words, Aspell does not provide any alternative and it also sometimes provides "multi-token alternatives" (e.g "zažívacího" -> "zažívací ho")
                if len(top_aspell_suggestions) > 0 and any([x.isalpha() for x in top_aspell_suggestions]):
                    chosen_suggestion = np.random.choice(top_aspell_suggestions)
                    while not chosen_suggestion.isalpha():
                        chosen_suggestion = np.random.choice(top_aspell_suggestions)

                    new_text.append(chosen_suggestion)
                    changes.append(['SPELL', 'Aspell replace {} with {}'.format(word, chosen_suggestion)])
                else:
                    new_text.append(word)
            elif np.random.uniform(0, 1) < self.spelling_word_to_invalid_word:
                new_word = list(word)

                detailed_spelling_applicable = False
                if len(word) >= 2:
                    detailed_spelling_applicable = True
                elif len(word) == 1:
                    # we need to be sure that the substitute/delete probability is high enough (so that we do not cycle here too long)
                    if new_word[0] in self.spelling_noise_operation_detailed_probs['S'] and np.sum(
                            list(self.spelling_noise_operation_detailed_probs['S'][new_word[0]].values())) > 0.1:
                        detailed_spelling_applicable = True

                    if new_word[0] in self.spelling_noise_operation_detailed_probs['D'] and \
                                    self.spelling_noise_operation_detailed_probs['D'][
                                        new_word[0]] > 0.1:
                        detailed_spelling_applicable = True

                if detailed_spelling_applicable and np.random.uniform(0, 1) < 1 - self.spelling_detailed_ratio:
                    num_iterations_spent = 0
                    # we must ensure that once we select the word to noisy, it will be actually noised and not an empty world
                    while ''.join(new_word) == word or not ''.join(new_word).strip():
                        new_word = list(word)
                        num_iterations_spent += 1
                        if num_iterations_spent > 1e4:
                            # it should not happen very often, better check whether we do not cycle here too long
                            break

                        for i in range(len(new_word)):
                            # try substitute
                            if new_word[i] in self.spelling_noise_operation_detailed_probs['S'] \
                                    and np.random.uniform(0, 1) < np.sum(
                                        list(self.spelling_noise_operation_detailed_probs['S'][new_word[i]].values())):

                                substitute_probabilites = np.array(
                                    list(self.spelling_noise_operation_detailed_probs['S'][new_word[i]].values()))
                                substitute_probabilites_normalized = substitute_probabilites / np.sum(substitute_probabilites)

                                new_word[i] = np.random.choice(list(self.spelling_noise_operation_detailed_probs['S'][new_word[i]].keys()),
                                                               p=substitute_probabilites_normalized)
                                continue
                            # try delete
                            elif new_word[i] in self.spelling_noise_operation_detailed_probs['D'] \
                                    and np.random.uniform(0, 1) < self.spelling_noise_operation_detailed_probs['D'][new_word[i]]:

                                new_word[i] = ''
                                continue
                            # try transpose
                            elif i < len(word) - 1 and np.random.uniform(0, 1) < self.spelling_noise_operation_probs['T']:
                                temp = new_word[i]
                                new_word[i] = new_word[i + 1]
                                new_word[i + 1] = temp
                                continue
                            # try insert
                            else:
                                left_context = '^' if i == 0 else word[i - 1]
                                right_context = "$" if i >= len(word) else word[i]
                                context = left_context + right_context

                                if context in self.spelling_noise_operation_detailed_probs['I'] \
                                        and np.random.uniform(0, 1) < np.sum(
                                            list(self.spelling_noise_operation_detailed_probs['I'][context].values())):
                                    insert_probabilites = np.array(
                                        list(self.spelling_noise_operation_detailed_probs['I'][context].values()))
                                    insert_probabilites_normalized = insert_probabilites / np.sum(insert_probabilites)

                                    insert_char = np.random.choice(list(self.spelling_noise_operation_detailed_probs['I'][context].keys()),
                                                                   p=insert_probabilites_normalized)
                                    new_word[i] = insert_char + new_word[i]
                                continue

                    new_text.append(''.join(new_word))
                    changes.append(['SPELL detailed', 'Char replace {} with {}'.format(word, new_text[-1])])
                else:
                    num_iterations_spent = 0
                    # we must ensure that once we select the word to noisy, it will be actually noised and not an empty world
                    while ''.join(new_word) == word or not ''.join(new_word).strip():
                        new_word = list(word)
                        num_iterations_spent += 1
                        if num_iterations_spent > 1e4:
                            # it should not happen very often, better check whether we do not cycle here too long
                            break

                        for i in range(len(new_word)):

                            no_op_prob = max(0, 1 - self.spelling_noise_operation_probs['S'] - self.spelling_noise_operation_probs['T'] - \
                                             self.spelling_noise_operation_probs['I'] - self.spelling_noise_operation_probs['D'])
                            op_type = np.random.choice(['0', 'S', 'T', 'T', 'D'],
                                                       p=[no_op_prob, self.spelling_noise_operation_probs['S'],
                                                          self.spelling_noise_operation_probs['T'],
                                                          self.spelling_noise_operation_probs['I'],
                                                          self.spelling_noise_operation_probs['D']])
                            # substitute
                            if op_type == 'S':
                                if all_alpha_chars_in_text_and_language.difference(word[i]):
                                    new_word[i] = np.random.choice(list(all_alpha_chars_in_text_and_language.difference(word[i])))
                                continue
                            # transpose
                            elif op_type == 'T' and i < len(word) - 1:
                                temp = new_word[i]
                                new_word[i] = new_word[i + 1]
                                new_word[i + 1] = temp
                                continue
                            # insert
                            elif op_type == 'I':
                                if np.random.uniform(0, 1) < 0.5:  # insert to the left of the current char
                                    new_word[i] = np.random.choice(list(all_alpha_chars_in_text_and_language.difference(word[i]))) + \
                                                  new_word[i]
                                else:
                                    new_word[i] = new_word[i] + np.random.choice(
                                        list(all_alpha_chars_in_text_and_language.difference(word[i])))
                                continue
                            # delete
                            elif op_type == 'D':
                                new_word[i] = ''
                                continue
                    new_text.append(''.join(new_word))
                    changes.append(['SPELL generalized', 'Char replace {} with {}'.format(word, new_text[-1])])

            else:
                new_text.append(word)

        return " ".join(new_text), changes, whitespace_info

    @staticmethod
    def estimate_probabilities(m2_records):
        num_files = 0
        num_spelling_incorrect_word = 0  # number of words that contain spelling error and the misspelled word is not a valid word
        num_spelling_valid_word = 0  # number of words that contain spelling error but the misspelled word is a valid word

        num_all_alpha_words = 0

        spelling_noise_operation_distrib = {
            'S': [],
            'I': [],
            'D': [],
            'T': []
        }

        spelling_noise_operation_detailed = {
            'S': {},
            'I': {},
            'D': {}
        }

        character_counter_in_corrected_spelling = Counter()
        twocharacter_counter_in_corrected_spelling = Counter()

        cached_coder_dicts = []
        for m2_file in m2_records:
            num_files += 1

            for info in m2_file:
                orig_sent, coder_dict = apply_m2_edits.processM2(info, [])
                if coder_dict:
                    coder_id = list(coder_dict.keys())[0]
                    cor_sent = coder_dict[coder_id][0]

                    for edit in coder_dict[coder_id][1]:
                        orig_start, orig_end, error_type, cor_tok, cor_start, cor_end = edit

                        exclude_error_types = ['noop', 'DIACR', 'PUNCT', 'ORTH']
                        # print(orig_sent, orig_start)
                        # print(cor_sent, cor_start)
                        '''
                        There are two types of spelling error:
                            1. invalid_word -> valid_word (e.g. thiis -> this) (Errant: SPELL)
                            2. valid_word -> valid_word (e.g. than -> then)

                        For the first type, we also estimate the noise in terms of number of bad insertions, deletions and transpositions.

                        '''
                        if "SPELL" in error_type and "WO:SPELL" not in error_type:
                            num_spelling_incorrect_word += 1

                            cached_coder_dicts.append(coder_dict)

                            # ! NOTE that it is important to calculate it from corrected to original (as we are gonna use it in this way)
                            align_seq = get_cheapest_align_seq(" ".join(cor_sent[cor_start:cor_end]),
                                                               " ".join(orig_sent[orig_start:orig_end]))

                            # first fill in spelling_noise_operation (which only stores probabilities of substituting individual char, inserting ...)
                            cor_len = len(" ".join(cor_sent[cor_start:cor_end]))

                            num_substitute = sum([1 for x in align_seq if x[0] == 'S'])
                            num_insert = sum([1 for x in align_seq if x[0] == 'I'])
                            num_delete = sum([1 for x in align_seq if x[0] == 'D'])
                            num_transpose = sum([1 for x in align_seq if x[0] == 'T'])

                            spelling_noise_operation_distrib['S'].append(num_substitute / cor_len)
                            spelling_noise_operation_distrib['I'].append(num_insert / cor_len)
                            spelling_noise_operation_distrib['D'].append(num_delete / cor_len)
                            spelling_noise_operation_distrib['T'].append(num_transpose / cor_len)

                            # then fill in detailed occurence counts for substitute, insert and delete
                            original_word = " ".join(orig_sent[orig_start:orig_end])
                            corrected_word = " ".join(cor_sent[cor_start:cor_end])
                            character_counter_in_corrected_spelling.update(corrected_word)
                            twocharacter_counter_in_corrected_spelling.update(
                                [corrected_word[i] + corrected_word[i + 1] for i in range(len(corrected_word) - 1)])
                            twocharacter_counter_in_corrected_spelling.update(['^' + corrected_word[0], corrected_word[-1] + '$'])

                            for operation in align_seq:
                                op_type, cor_start, cor_end, orig_start, orig_end = operation

                                if op_type == 'S':
                                    cor_char = corrected_word[cor_start]
                                    orig_char = original_word[orig_start]

                                    if cor_char not in spelling_noise_operation_detailed['S']:
                                        spelling_noise_operation_detailed['S'][cor_char] = {}

                                    if orig_char not in spelling_noise_operation_detailed['S'][cor_char]:
                                        spelling_noise_operation_detailed['S'][cor_char][orig_char] = 0

                                    spelling_noise_operation_detailed['S'][cor_char][orig_char] += 1
                                elif op_type == 'I':
                                    left_context = "^" if cor_start == 0 else corrected_word[cor_start - 1]
                                    right_context = "$" if cor_start >= cor_len else corrected_word[cor_start]

                                    insert_into_tuple = left_context + right_context
                                    if insert_into_tuple not in spelling_noise_operation_detailed['I']:
                                        spelling_noise_operation_detailed['I'][insert_into_tuple] = {}

                                    insert_char = original_word[orig_start]

                                    if insert_char not in spelling_noise_operation_detailed['I'][insert_into_tuple]:
                                        spelling_noise_operation_detailed['I'][insert_into_tuple][insert_char] = 0

                                    spelling_noise_operation_detailed['I'][insert_into_tuple][insert_char] += 1
                                elif op_type == 'D':
                                    delete_char = corrected_word[cor_start]

                                    if delete_char not in spelling_noise_operation_detailed['D']:
                                        spelling_noise_operation_detailed['D'][delete_char] = 0

                                    spelling_noise_operation_detailed['D'][delete_char] += 1

                                    # # non alpha edits
                                    # if not " ".join(cor_sent[cor_start:cor_end]).isalpha():
                                    #     print('not alpha cor', edit)
                                    #
                                    # if not " ".join(orig_sent[orig_start:orig_end]).isalpha():
                                    #     print('not alpha orig', edit)

                        elif all([x not in error_type for x in exclude_error_types]) and \
                                        orig_start == orig_end - 1 and orig_start < len(orig_sent) and cor_start == cor_end - 1 and \
                                        SequenceMatcher(None, orig_sent[orig_start], cor_sent[cor_start]).ratio() > 0.5:
                            num_spelling_valid_word += 1
                            # print(orig_sent[orig_start], cor_sent[cor_start])

                            # # non alpha edits
                            # if not " ".join(cor_sent[cor_start:cor_end]).isalpha():
                            #     print('not alpha cor', edit)
                            #
                            # if not " ".join(orig_sent[orig_start:orig_end]).isalpha():
                            #     print('not alpha orig', edit)

                            # TODO WO:SPELL

                num_all_alpha_words += sum([1 for x in orig_sent if x.isalpha()])

        # normalization to probabilities for word_to_other_valid_word and word_to_invalid_word
        spelling_word_to_other_valid_word = num_spelling_valid_word / num_all_alpha_words
        spelling_word_to_invalid_word = num_spelling_incorrect_word / num_all_alpha_words

        # normalization to probabilities for aggregated spelling noise operations
        spelling_noise_operation_probs = {}

        for k, v in spelling_noise_operation_distrib.items():
            spelling_noise_operation_probs[k] = np.mean(v)

        # normalization to probabilities for detailed spelling noise operations
        spelling_noise_operation_detailed_probs = {
            'S': {},
            'I': {},
            'D': {}
        }
        ## substitutes
        for cor_from, v in spelling_noise_operation_detailed['S'].items():
            spelling_noise_operation_detailed_probs['S'][cor_from] = {}
            for cor_to, val in spelling_noise_operation_detailed['S'][cor_from].items():
                spelling_noise_operation_detailed_probs['S'][cor_from][cor_to] = spelling_noise_operation_detailed['S'][cor_from][
                                                                                     cor_to] / \
                                                                                 character_counter_in_corrected_spelling[cor_from]
        ## inserts
        for two_char_context, v in spelling_noise_operation_detailed['I'].items():
            spelling_noise_operation_detailed_probs['I'][two_char_context] = {}
            for insert_char, val in spelling_noise_operation_detailed['I'][two_char_context].items():
                spelling_noise_operation_detailed_probs['I'][two_char_context][insert_char] = \
                    spelling_noise_operation_detailed['I'][two_char_context][insert_char] / \
                    twocharacter_counter_in_corrected_spelling[two_char_context]

        ## deletes
        for delete_char, v in spelling_noise_operation_detailed['D'].items():
            spelling_noise_operation_detailed_probs['D'][delete_char] = spelling_noise_operation_detailed['D'][delete_char] / \
                                                                        character_counter_in_corrected_spelling[delete_char]

        return 'spelling', {
            'spelling_word_to_other_valid_word': spelling_word_to_other_valid_word,
            'spelling_word_to_invalid_word': spelling_word_to_invalid_word,
            'spelling_noise_operation_probs': spelling_noise_operation_probs,
            'spelling_noise_operation_detailed_probs': spelling_noise_operation_detailed_probs
        }
