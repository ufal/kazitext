import argparse
import glob
import numpy as np
import apply_m2_edits

def get_edits_info(m2_pattern):
    simple_edits = []  # ratios of bad / all_tokens
    detailed_edits = []  # in each edit we also count its span both in original and corrected (whole sentence rewrite is one simple but many detailed edits)

    for f in glob.glob(m2_pattern + "*"):
        m2_file = open(f).read().strip().split("\n\n")
        for info in m2_file:
            orig_sent, coder_dict = apply_m2_edits.processM2(info, [])
            num_tokens = len(orig_sent)  # in tokens

            if coder_dict:
                coder_id = list(coder_dict.keys())[0]

                edits = coder_dict[coder_id][1]
                filtered_edits = [edit for edit in edits if edit[2] != 'noop']
                simple_edits.append(len(filtered_edits) / num_tokens)

                num_detailed_edits = 0
                for edit in filtered_edits:
                    orig_start, orig_end, error_type, cor_tok, cor_start, cor_end = edit
                    num_detailed_edits += orig_end - orig_start
                    num_detailed_edits += cor_end - cor_start

                detailed_edits.append(num_detailed_edits / num_tokens)
            else:
                simple_edits.append(0)
                detailed_edits.append(0)

    return simple_edits, detailed_edits


if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser()
    parser.add_argument("m2_pattern", help="Pattern to match F2 files to evaluate.", type=str)
    parser.add_argument("--vis_name", default="", help="Visualize hist into vis_name if provided.")
    args = parser.parse_args()

    simple_edits, detailed_edits = get_edits_info(args.m2_pattern)

    simple_edits_ratio = np.mean(simple_edits)
    detailed_edits_ratio = np.mean(detailed_edits)
    print('Simple edits ratio: {}'.format(np.mean(simple_edits)))
    print('Detailed edits ratio: {}'.format(np.mean(detailed_edits)))

    # standard-deviation of simple edits
    print(np.std(simple_edits))

    # custom-metric to select best matching alpha
    print(100 * ((8 * simple_edits_ratio + detailed_edits_ratio) / 9))  # weighted average reported in percentages

    if args.vis_name:
        import matplotlib.pyplot as plt
        import numpy as np

        H, bins = np.histogram(simple_edits, bins=100, range=(0,1))

        print(H)
        print(bins)

        plt.subplot(3, 3, 1)

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, H, align='center', width=width)
        #
        # plt.bar(bins[:-1], H)  # `density=False` would make counts
        # plt.xlim(-0.1, 1)
        # # plt.yscale('log')
        plt.ylabel('Probs')
        plt.xlabel('Simple edits ratio')

        plt.subplot(3, 3, 2)
        H, bins = np.histogram(simple_edits, bins=100, density=True)

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, H, align='center', width=width)

        # plt.yscale('log')
        plt.ylabel('Probs')
        plt.xlabel('Simple edits ratio')

        plt.subplot(3, 3, 3)
        H, bins = np.histogram(simple_edits, bins=100)

        H = [x / np.sum(H) for x in H]

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, H, align='center', width=width)
        # plt.yscale('log')
        plt.ylabel('Probs')
        plt.xlabel('Simple edits ratio')
        
        plt.subplot(3, 3, 4)
        plt.hist(simple_edits, density=True, bins=100)  # `density=False` would make counts
        plt.xlim(-0.1, 1)
        plt.ylim(0, 100)
        # plt.yscale('log')
        plt.ylabel('Probs')
        plt.xlabel('Simple edits ratio')

        plt.subplot(3, 3, 5)
        plt.hist(simple_edits, density=True, bins=100)  # `density=False` would make counts
        plt.xlim(-0.1, 1)
        plt.ylim(0, 15)
        # plt.yscale('log')
        plt.ylabel('Probs')
        plt.xlabel('Simple edits ratio')

        plt.subplot(3, 3, 6)
        plt.hist(simple_edits, density=True, bins=20)  # `density=False` would make counts
        plt.xlim(-0.1, 1)
        plt.ylim(0, 10)
        # plt.yscale('log')
        plt.ylabel('Probs')
        plt.xlabel('Simple edits ratio')


        plt.subplot(3, 3, 7)
        plt.hist(detailed_edits, density=True, bins=100)  # `density=False` would make counts
        plt.ylabel('Probs')
        plt.xlabel('Detailed edits ratio')
        # plt.yscale('log')
        plt.ylim(0, 100)

        plt.subplot(3, 3, 8)
        plt.hist(detailed_edits, density=True, bins=100)  # `density=False` would make counts
        plt.ylabel('Probs')
        plt.xlabel('Detailed edits ratio')
        # plt.yscale('log')
        plt.ylim(0, 15)

        plt.subplot(3, 3, 9)
        plt.hist(detailed_edits, density=True, bins=20)  # `density=False` would make counts
        plt.ylabel('Probs')
        plt.xlabel('Detailed edits ratio')
        # plt.yscale('log')
        plt.ylim(0, 10)

        plt.savefig(args.vis_name)

