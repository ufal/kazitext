# KaziText

KaziText is a tool for modelling common human errors. It estimates probabilities of individual error types (so called aspects) from grammatical error correction corpora in M2 format.
 
The tool was introduced in [Understanding Model Robustness to User-generated Noisy Texts](https://arxiv.org/pdf/2110.07428.pdf).

## Requirements

A set of requirements is listed in ```requirements.txt```. Moreover, UDPipe model has to be downloaded for used languages (see http://hdl.handle.net/11234/1-3131) and linked in ```udpipe_tokenizer.py```.

## Overview

KaziText defines a set of aspects located in [aspects](aspects). These model following phenomena:
- Casing Errors 
- Common Other Errors (for most common phrases)
- Errors in Diacritics
- Punctuation Errors
- Spelling Errors
- Errors in wrongly used suffix/prefix
- Whitespace Errors
- Word-Order Errors

Each aspect has a set of internal probabilities (e.g. the probability of a user typing first letter of a starting word in lower-case instead of upper-case) that are estimated from M2 GEC corpora. 

A complete set of aspects with their internal probabilities is called *profile*. We provide precomputed profiles for Czech, English, Russian and German in [profiles](profiles) as json files. The profiles are additionally split into dev and test. Also there are 4 profiles for Czech and 2 profiles for English differing in the underlying user domain (e.g. natives vs second learners).   

To **noise** a text using a profile, use:
```
python introduce_errors.py $infile $outfile $profile $lang 
```

```introduce_errors.py``` script offers a variety of switches (run ```python introduce_errors.py --help``` to display them). 
One noteworthy is ```--alpha``` that serves for regulating final text error rate (set it to value lower than 1 to reduce number of errors; set to to value bigger than 1 to have more noisy texts).
Apart for profiles themselves, we also precomputed set of alphas that are stored as .csv files in respective [profiles](profiles) folders and store values for alphas to reach 5-30 final text word error rates as well as so called *reference-alpha* word error rate that corresponds to the same error rate as the original M2 files the profile was estimated from had. To have for example noisy text at circa 5% word error rate noised by Romani profile, use ```--profile dev/cs_romi.json --alpha 0.2```.
 
Moreover, we provide several scripts (```noise*.py```) for noising specific data formats.

To **estimate** a profile for given M2 file, run:
```
python estimate_all_ratios.py $m2_pattern outfile
```

To **estimate** normalization alphas file, see ```estimate_alpha.sh``` that describes iterative process of noising clean texts with an alpha, measuring text's noisiness and changing alpha respectively. 

## Other notes

- Russian RULEC-GEC was normalized using ```normalize_russian_m2.py```

