import argparse
import test_classifires
import commons
import utils


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='test_classifiers',
                    choices=['test_classifiers', 'gen_unwanted_cache', 'gen_corpora'], )


def main(mode):
    if mode == "test_classifiers":
        test_classifires.test_classifiers()
    elif mode == "gen_unwanted_cache":
        utils.generate_unwanted_words_cache()
    elif mode == "gen_corpora":
        utils.generate_corpora()


if __name__ == "__main__":
    args = parser.parse_args()
    main(str(args.mode))
