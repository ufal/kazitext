import sys

from aspects import diacritization_stripping_data

def strip_diacritics(list_of_texts):
    for line in list_of_texts:
        output = ""
        for c in line:
            if c in diacritization_stripping_data.strip_diacritization_uninames:
                output += diacritization_stripping_data.strip_diacritization_uninames[c]
            else:
                output += c

        yield output

def strip_diacritics_single_line(textline):
    output = ""
    for c in textline:
        if c in diacritization_stripping_data.strip_diacritization_uninames:
            output += diacritization_stripping_data.strip_diacritization_uninames[c]
        else:
           output += c

    return output
