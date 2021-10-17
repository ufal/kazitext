from abc import ABC


class Aspect(ABC):
    """
        Base, abstract class.
    """

    def __init__(self, profile, lang, alpha=1, beta=0):
        pass

    def apply(self, text, whitespace_info):
        """
            Apply specific noise to given tokenized text.
            Whitespace_info stores information on whether space should be inserted between adjacent tokens when detokenizing.
        """
        pass

    @staticmethod
    def estimate_probabilities(m2_records):
        pass
