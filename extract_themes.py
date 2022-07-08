"""
Script to clean up the themes in the spreadsheet and save it in a new dataframe in long format.
"""

import pandas as pd

def main():
    """Main function of the script"""
    #Loading the raw file
    emner = pd.read_csv("/work/linkgraph_cleaned/emner_raw.csv")
    emner = (
        emner
        .fillna("") #Filling all empty fields with empty strings so they can be joined
        .apply("\n".join) #Joining all fields with a newline
        .str
        .split("\n") #Splitting all fields into a list of themes
        .explode() #Exploding all lists to their own fields
    )
    emner = emner[emner != ""] #Remove empty fields in the series
    emner.index.rename("year", inplace=True) #Renaming the index column to 'year'
    emner = emner.reset_index(name="emne") #We reset the index so it becomes a DataFrame
    emner = emner.assign(year=emner.year.str.slice(0,4)) #removing '(reexp)' tags
    #Setting correct types for all columns
    emner = emner.astype({
        "year": int,
        "emne": str
    })
    emner.to_csv("/work/linkgraph_cleaned/emner.csv")

if __name__=="__main__":
    main()
    