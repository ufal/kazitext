#!/usr/bin/env bash

set -ex

lang=$1 # only cs and en currently supported
profile_file=$2
reference_m2_files=$3
out_file=$4
required_error_levels=${@:5} # in percentages; all arguments from 4th on (i.e. w/o 1,2,3)

if [[ $lang == "en" ]]; then
    monolingual_data="/ha/home/naplava/czesl_experiments/data/monolingual/en/news.2017.en.tokenized_cleaned.txt"
    tokenized=True

elif [[ $lang == "cs" ]]; then
    monolingual_data="/ha/home/naplava/czesl_experiments/data/monolingual/cs/syn_v4_tokenized_cleaned.txt"
    tokenized=True

elif [[ $lang == "de" ]]; then
    monolingual_data="/ha/home/naplava/czesl_experiments/data/monolingual/de/news.2017.de.tokenized_cleaned.txt"
    tokenized=True

elif [[ $lang == "ru" ]]; then
    monolingual_data="/ha/home/naplava/czesl_experiments/data/monolingual/ru/news.2017.ru.tokenized_cleaned.txt"
    tokenized=True
fi

## COMPUTE REFERENCE ERROR RATE AND ERROR RATES FOR SPECIFIC ALPHAS

echo "Computing error rate on reference M2 file: $reference_m2_files"
reference_error_rate=$(python3 compute_error_rate.py $reference_m2_files | tail -n 1)
echo "Reference error rate is $reference_error_rate"

alphas_to_measure_on="0.2 0.4 0.5 0.6 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.7 1.9 2 2.3 2.6 2.9 3 3.3 3.6 4 4.5 5 6 7 8"
array_alphas_to_measure_on=($alphas_to_measure_on)
declare -a error_rates

for alpha in ${array_alphas_to_measure_on[@]}; do
    # introduce errors to first 20000 lines
    monolingual_data_head="/tmp/$(basename $monolingual_data)-20000-$BASHPID.txt"
    monolingual_data_head_noised="/tmp/$(basename $monolingual_data)-20000-noised-$alpha-$BASHPID.txt"
    head -n 20000 $monolingual_data > $monolingual_data_head
    /home/naplava/virtualenvs/aspell/bin/python ../scripts/introduce_errors.py $monolingual_data_head $monolingual_data_head_noised $profile_file --lang $lang --alpha $alpha

    # create M2 file using Errant
    monolingual_data_head_m2="/tmp/$(basename $monolingual_data)-20000-$alpha-$BASHPID.m2"

    if [[ $lang == "en" ]] || [[ $lang == "ru" ]] || [[ $lang == "de" ]]; then
        /ha/home/naplava/troja/new_errant/english_errant/errant/venv/bin/errant_parallel -cor $monolingual_data_head -orig $monolingual_data_head_noised -out $monolingual_data_head_m2
    elif [[ $lang == "cs" ]]; then
        /home/naplava/troja/new_errant/errant/venvs/czech/bin/python /home/naplava/troja/new_errant/errant/try_parallel_to_m2.py -cor $monolingual_data_head -orig $monolingual_data_head_noised -out $monolingual_data_head_m2 -lang $lang
    fi

    # num errors per token
    cur_err_rate=$(python3 compute_error_rate.py $monolingual_data_head_m2 | tail -n 1)
    error_rates+=($cur_err_rate)

    echo "$alpha: $cur_err_rate"

    rm $monolingual_data_head
    rm $monolingual_data_head_noised
    # rm $monolingual_data_head_m2

    # just not to loop for too long in certain cases
    if (( $(echo "$cur_err_rate > 40" |bc -l) )); then
        break
    fi

done

### GIVEN COMPUTED ERROR RATES FOR SPECIFIC ALPHAS, FIND CLOSEST ALPHA FOR ERROR RATES IN SPECIFIED VALUES

# first estimate best alpha for reference M2 file
i=0
best_dif=1000000000000
best_dif_alpha=0
for (( i=0; i<${#error_rates[*]}; ++i)); do
    dif=$(bc -l <<< "${error_rates[i]} - $reference_error_rate")
    dif=$(bc -l <<< "$dif*$dif") # ~ abs

    if (( $(bc -l <<< "$dif < $best_dif") )); then
        best_dif=$dif
        best_dif_alpha=${array_alphas_to_measure_on[i]}
    fi
done

reference_best_alpha=$best_dif_alpha
echo "reference-alpha,$best_dif_alpha"
echo "reference-alpha;$best_dif_alpha" > $out_file


# then estimate best alpha for each required error level
for rel in $required_error_levels; do
    i=0
    best_dif=1000000000000
    best_dif_alpha=0
    for (( i=0; i<${#error_rates[*]}; ++i)); do
        dif=$(bc -l <<< "${error_rates[i]} - $rel")
        dif=$(bc -l <<< "$dif*$dif") # ~ abs
        if (( $(bc -l <<< "$dif < $best_dif") )); then
            best_dif=$dif
            best_dif_alpha=${array_alphas_to_measure_on[i]}
        fi
    done

    echo "$rel;$best_dif_alpha" >> $out_file
done


### FOR REFERENCE ALPHA, COMPUTE ALSO SUCH STANDARD DEVIATION WHICH BEST MATCHES REFERENCE STANDARD DEVIATION
echo "Computing std on reference M2 file: $reference_m2_files"
reference_std=$(python3 compute_error_rate.py $reference_m2_files | tail -n 2 | head -n 1)
echo "Reference std is $reference_std"

stds_to_measure_on=$(seq 0.01 0.05 1)
array_stds_to_measure_on=($stds_to_measure_on)
declare -a computed_stds

for cur_std in ${array_stds_to_measure_on[@]}; do
    # introduce errors to first 20000 lines
    monolingual_data_head="/tmp/$(basename $monolingual_data)-20000-$BASHPID.txt"
    monolingual_data_head_noised="/tmp/$(basename $monolingual_data)-20000-noised-$reference_best_alpha-$BASHPID.txt"
    head -n 20000 $monolingual_data > $monolingual_data_head
    /home/naplava/virtualenvs/aspell/bin/python ../scripts/introduce_errors.py $monolingual_data_head $monolingual_data_head_noised $profile_file --lang $lang --alpha $reference_best_alpha --alpha-std $cur_std --alpha-uniformity-prob 0

    # create M2 file using Errant
    monolingual_data_head_m2="/tmp/$(basename $monolingual_data)-20000-$reference_best_alpha-$BASHPID.m2"

    if [[ $lang == "en" ]] || [[ $lang == "ru" ]] || [[ $lang == "de" ]]; then
        /ha/home/naplava/troja/new_errant/english_errant/errant/venv/bin/errant_parallel -cor $monolingual_data_head -orig $monolingual_data_head_noised -out $monolingual_data_head_m2
    elif [[ $lang == "cs" ]]; then
        /home/naplava/troja/new_errant/errant/venvs/czech/bin/python /home/naplava/troja/new_errant/errant/try_parallel_to_m2.py -cor $monolingual_data_head -orig $monolingual_data_head_noised -out $monolingual_data_head_m2 -lang $lang
    fi

    # num errors per token
    real_std=$(python3 compute_error_rate.py $monolingual_data_head_m2 | tail -n 2 | head -n 1)
    computed_stds+=($real_std)

    echo "$cur_std: $real_std"

    rm $monolingual_data_head
    rm $monolingual_data_head_noised
    #rm $monolingual_data_head_m2
done

i=0
best_dif=1000000000000
best_dif_std=0
for (( i=0; i<${#computed_stds[*]}; ++i)); do
    dif=$(bc -l <<< "${computed_stds[i]} - $reference_std")
    dif=$(bc -l <<< "$dif*$dif") # ~ abs

    if (( $(bc -l <<< "$dif < $best_dif") )); then
        best_dif=$dif
        best_dif_std=${array_stds_to_measure_on[i]}
    fi
done

echo "reference-std,$best_dif_std"
echo "reference-std;$best_dif_std" >> $out_file


## FINALLY, COMPUTE NO-ERROR-SENTENCE-BOOST USING BEST ALPHA AND ALPHA-STD
echo "Final"
# introduce errors to first 40000 lines
monolingual_data_head="/tmp/$(basename $monolingual_data)-40000-$BASHPID.txt"
monolingual_data_head_noised="/tmp/$(basename $monolingual_data)-40000-noised-$reference_best_alpha-$BASHPID.txt"
head -n 40000 $monolingual_data > $monolingual_data_head
/home/naplava/virtualenvs/aspell/bin/python ../scripts/introduce_errors.py $monolingual_data_head $monolingual_data_head_noised $profile_file --lang $lang --alpha $reference_best_alpha --alpha-std $best_dif_std --alpha-uniformity-prob 0
echo "Final M2"
# create M2 file using Errant
monolingual_data_head_m2="/tmp/$(basename $monolingual_data)-40000-$reference_best_alpha-$BASHPID.m2"

if [[ $lang == "en" ]] || [[ $lang == "ru" ]] || [[ $lang == "de" ]]; then
    /ha/home/naplava/troja/new_errant/english_errant/errant/venv/bin/errant_parallel -cor $monolingual_data_head -orig $monolingual_data_head_noised -out $monolingual_data_head_m2
elif [[ $lang == "cs" ]]; then
    /home/naplava/troja/new_errant/errant/venvs/czech/bin/python /home/naplava/troja/new_errant/errant/try_parallel_to_m2.py -cor $monolingual_data_head -orig $monolingual_data_head_noised -out $monolingual_data_head_m2 -lang $lang
fi

# get ratio of no-error sentences in reference and in actual
num_sentences_with_no_edit_in_reference=$(cat $reference_m2_files | grep -A 1 "^S" | grep "^A -1 -1|||noop" | wc -l)
# Russian sentences do not contain "noop" operations, but only single ^S line with a following empty line
num_additional_sentences_with_no_edit_in_reference=$(cat $reference_m2_files | grep -A 1 "^S" | grep "^$" | wc -l)
num_sentences_with_no_edit_in_reference=$(bc -l <<< "$num_sentences_with_no_edit_in_reference + $num_additional_sentences_with_no_edit_in_reference")

num_sentences_in_reference=$(cat $reference_m2_files | grep "^S" | wc -l)
ratio_no_edit_in_reference=$(bc -l <<< "$num_sentences_with_no_edit_in_reference / $num_sentences_in_reference")

num_sentences_with_no_edit_in_actual=$(cat $monolingual_data_head_m2 | grep -A 1 "^S" | grep "^A -1 -1|||noop" | wc -l)
num_sentences_in_actual=$(cat $monolingual_data_head_m2 | grep "^S" | wc -l)
ratio_no_edit_in_actual=$(bc -l <<< "$num_sentences_with_no_edit_in_actual / $num_sentences_in_actual")
ratio_with_edit_in_actual=$(bc -l <<< "1 - $ratio_no_edit_in_actual")
echo "$num_sentences_with_no_edit_in_actual"
echo "$ratio_no_edit_in_reference"
echo "$ratio_no_edit_in_actual"

# and compute what percentage of already noised sentences must be "denoised" or how many sentences that were not noised should be noised so that similar percentage of sentences as in actual has no noise

if (( $(echo "$ratio_no_edit_in_reference > $ratio_no_edit_in_actual" |bc -l) )); then
    no_error_sentence_boost_constant=$(bc -l <<< "($ratio_no_edit_in_reference - $ratio_no_edit_in_actual) / $ratio_with_edit_in_actual")
else
    no_error_sentence_boost_constant=$(bc -l <<< "($ratio_no_edit_in_reference - $ratio_no_edit_in_actual) / $ratio_no_edit_in_actual")
fi

echo "no-error-sentence-boost;$no_error_sentence_boost_constant"
echo "no-error-sentence-boost;$no_error_sentence_boost_constant" >> $out_file

rm $monolingual_data_head
rm $monolingual_data_head_noised

