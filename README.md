# Netarkivet Linkgraph

Repo containing code for cleaning, processing and creating link graphs based on Netarkivet.

## Setup

To be able to run the code in the repo, make sure to install all dependencies in
requirements.txt.

## Scripts

### Cleaning

`clean_linkgraph.py` cleans up the original link graphs from Netarkivet
and saves them in `.feather` files for faster loading times and more efficient storage.

### Extracting link graphs

If you intend to create link graphs purely from the textual content of Netarkivet,
use `extract_linkgraph.py`. The script extracts link information from the content and creates
link graphs.

### Analysing topics

#### Preprocessing

`preprocess_topics.py` collects all records relevant for the analysis of
all topics per year, as well as creates internal linkgraphs,
and creates a semantic summary of all domains in the collected data set.

#### Plotting

`create_topic_plots.py` plots preprocessed topics and saves them as `.html` files.
These plots contain both semantic and link graph information.
