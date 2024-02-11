#!/usr/bin/env python3
"""
  merge.py
  This program snippet shows how to merge two files
"""
file1 = "../data/shockwave/tweet_train.csv"
file2 = "../data/shockwave/tweet_test.csv"
file_out = "../data/shockwave/tweet_all.csv"


def merge(file1: str, file2: str):
    """
    Merge two files
    """
    # Creating a list of filenames
    filenames = [file1, file2]

    # Open file3 in write mode
    with open(file_out, 'w') as outfile:

        # Iterate through list
        for names in filenames:

            # Open each file in read mode
            with open(names) as infile:

                # read the data from file1 and
                # file2 and write it in file3
                outfile.write(infile.read())

            # Add '\n' to enter data of file2
            # from next line
            outfile.write("\n")


# The driver function (confirm that code is under main function)
if __name__ == "__main__":
    merge(file1, file2)

    print("\nDone!")
