#!/usr/bin/env python3
"""
  concat_files.py
  This program concatenates multiple text files
"""
import glob


def concat_files(file_name: str, pattern: str):
    seperator = "======================"

    # Load all txt files in path
    files = glob.glob()

    # Concatenate files to new file
    with open(file_name, 'w') as out_file:
        for name in files:
            with open(name) as in_file:
                out_file.write(in_file.read())

            out_file.write("\n\n")
            out_file.write(seperator)
            out_file.write("\n\n")


def read_file(file_name: str):
    # Read file and print
    with open(file_name, 'r') as new_file:
        lines = [line.strip() for line in new_file]

    for line in lines: print(line)


def main(pattern: str):
    concat_files(pattern)


# Check that code is under main function
if __name__ == "__main__":
    out_file = "output.txt"
    pattern = "../data/text_files/*.txt"
    main()
    print("\nDone")