import argparse


# Input: A list of edit lines for a sentence in an m2 file.
# Output: An edit dictionary; key is coder id, value is a list of edits.
def processEdits(edits):
    edit_dict = {}
    for edit in edits:
        edit = edit.split("|||")
        span = edit[0][2:].split()  # [2:] ignore the leading "A "
        start = int(span[0])
        end = int(span[1])
        cat = edit[1]
        cor = edit[2]
        id = edit[-1]
        # Save the useful info as a list
        proc_edit = [start, end, cat, cor]
        # Save the proc edit inside the edit_dict using coder id.
        if id in edit_dict.keys():
            edit_dict[id].append(proc_edit)
        else:
            edit_dict[id] = [proc_edit]
    return edit_dict


# Input: A sentence + edit block in an m2 file.
# Output 1: The original sentence (a list of tokens)
# Output 2: A dictionary; key is coder id, value is a tuple.
# tuple[0] is the corrected sentence (a list of tokens), tuple[1] is the edits.
# Process M2 to extract sentences and edits.
def processM2(info, ignore_edit_types):
    info = info.split("\n")
    orig_sent = info[0][2:].split()  # [2:] ignore the leading "S "
    all_edits = info[1:]
    # Simplify the edits and group by coder id.
    edit_dict = processEdits(all_edits)
    out_dict = {}
    # Loop through each coder and their edits.
    for coder, edits in edit_dict.items():
        # Copy orig_sent. We will apply the edits to it to make cor_sent
        cor_sent = orig_sent[:]
        gold_edits = []
        offset = 0
        for edit in edits:
            # Do not apply noop or Um edits, but save them
            if edit[2] in {"noop", "Um"}:
                gold_edits.append(edit + [-1, -1])
                continue

            if ignore_edit_types:
                is_ignore_edit_type = False
                for ignore_edit_type in ignore_edit_types:
                    if ignore_edit_type in edit[2]: # substring match
                        is_ignore_edit_type = True

                if is_ignore_edit_type:
                    continue

            orig_start = edit[0]
            orig_end = edit[1]
            cor_toks = edit[3].split()
            # Apply the edit.
            cor_sent[orig_start + offset:orig_end + offset] = cor_toks
            # Get the cor token start and end positions in cor_sent
            cor_start = orig_start + offset
            cor_end = cor_start + len(cor_toks)
            # Keep track of how this affects orig edit offsets.
            offset = offset - (orig_end - orig_start) + len(cor_toks)
            # Save the edit with cor_start and cor_end
            gold_edits.append(edit + [cor_start] + [cor_end])
        # Save the cor_sent and gold_edits for each annotator in the out_dict.
        out_dict[coder] = (cor_sent, gold_edits)
    return orig_sent, out_dict


def main(args):
    # Setup output m2 file
    out_parallel = open(args.out, "w")

    print("Processing files...")
    # Open the m2 file and split into sentence+edit chunks.
    m2_file = open(args.m2).read().strip().split("\n\n")
    for info in m2_file:
        # Get the original and corrected sentence + edits for each annotator.
        orig_sent, coder_dict = processM2(info, args.ignore_edit_types)
        # Save info about types of edit groups seen
        # Only process sentences with edits.
        if coder_dict:
            # Save marked up original sentence here, if required.
            # Loop through the annotators
            for coder, coder_info in sorted(coder_dict.items()):
                cor_sent = coder_info[0]
                out_parallel.write(" ".join(orig_sent) + "\t" + " ".join(cor_sent) + "\n")

    out_parallel.close()


if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser()
    parser.add_argument("-m2", help="The M2 file.", required=True)
    parser.add_argument("-out", help="The output filepath to the parallel file.", required=True)
    parser.add_argument('-iet', '--ignore_edit_types', nargs='+', default=None, help='List of error categories that will not be applied', required=False)
    args = parser.parse_args()

    print(args)
    main(args)
