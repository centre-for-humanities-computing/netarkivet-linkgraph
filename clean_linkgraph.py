"""
Script for cleaning up the linkgraph and saving it in a new directory.
"""
import glob
import os

import pandas as pd

SOURCE_PATH = "/work/linkgraph/"
DEST_PATH = "/work/linkgraph_cleaned/"


def main() -> None:
    """
    Main function of the script
    """
    # Collecting all file names of the original linkgraph
    files = glob.glob(
        os.path.join(SOURCE_PATH, "[0-9][0-9][0-9][0-9]_linkgraph_timestamps.csv/*.csv")
    )
    for file_name in files:
        # Extracting year from the file names
        # An example path would look something like this:
        # "/work/linkgraph/<year>_linkgraph_first_level.csv/<filename>.csv"
        # aka the year is the first four characters of the second last part of the path
        year = file_name.split("/")[-2][:4]
        print(f"Processing year: {year}")
        # Create a directory for each year
        out_directory = os.path.join(DEST_PATH, str(year))
        # Only create directory if it doesn't exist yet
        if not os.path.isdir(out_directory):
            os.mkdir(out_directory)
        # Loading the csv file containing the linkgraph
        linkgraph = pd.read_csv(
            file_name,
            names=["source", "target", "connections", "earliest_link", "latest_link"],
        )
        linkgraph = linkgraph[["source", "target", "connections"]]
        # Writing to a feather file for faster loading times
        # and more efficient storage
        # NOTE: remember to install pyarrow, it's in the requirements file
        linkgraph.to_feather(os.path.join(out_directory, "edges.feather"))
        print("DONE")


if __name__ == "__main__":
    main()
