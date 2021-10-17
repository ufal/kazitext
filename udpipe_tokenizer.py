#!/usr/bin/env python3
import ufal.udpipe


class UDPipeTokenizer:
    MODELS = {
        "cs": "/home/arahusky/Downloads/czech-pdt-ud-2.5-191206.udpipe",
        "de": "",
        "en": "",
        "ru": ""
    }

    class Token:
        def __init__(self, string, start, end):
            self.string = string
            self.start = start
            self.end = end

    def __init__(self, lang):
        self._model = ufal.udpipe.Model.load(self.MODELS[lang])

    def tokenize(self, text):
        """ Return tokenized text as a list of sentences, each a list of tokens. """

        tokenizer = self._model.newTokenizer(self._model.TOKENIZER_RANGES)
        if not tokenizer:
            raise RuntimeError("The model does not have a tokenizer")

        tokenizer.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while tokenizer.nextSentence(sentence, error):
            sentences.append([])

            multiword_token = 0
            for word in sentence.words[1:]:
                while multiword_token < len(sentence.multiwordTokens) and \
                                word.id > sentence.multiwordTokens[multiword_token].idLast:
                    multiword_token += 1
                if multiword_token < len(sentence.multiwordTokens) and \
                                word.id >= sentence.multiwordTokens[multiword_token].idFirst and \
                                word.id <= sentence.multiwordTokens[multiword_token].idLast:
                    if word.id > sentence.multiwordTokens[multiword_token].idFirst:
                        continue
                    word = sentence.multiwordTokens[multiword_token]
                sentences[-1].append(self.Token(word.form, word.getTokenRangeStart(), word.getTokenRangeEnd()))

        if error.occurred():
            raise RuntimeError(error.message)

        return sentences


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("lang", type=str, help="Language to use")
    args = parser.parse_args()

    tokenizer = UDPipeTokenizer(args.lang)

    for line in sys.stdin:
        for sentence in tokenizer.tokenize(line):
            print(*[token.string for token in sentence])
