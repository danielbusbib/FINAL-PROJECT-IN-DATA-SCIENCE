import numpy as np
import pandas as pd


# Transformation matrix
def transformation_matrix(date1, date2):
    df = pd.read_csv("RPE_MATCH_22_23.csv")
    # df by games
    df1 = df.loc[df['Timestamp'].str.contains(str(date1), case=False)]
    df2 = df.loc[df['Timestamp'].str.contains(str(date2), case=False)]
    mat = np.zeros((11, 11)).astype(int)
    dict1 = dict()
    # first pass over game 1
    for index, row in df1.iterrows():
        dict1[row["שם מלא - Full Name"]] = row['Individual rating']
    # second pass over game 2
    for index, row in df2.iterrows():
        if row["שם מלא - Full Name"] in dict1:
            a = dict1[row["שם מלא - Full Name"]]
            b = row['Individual rating']
            mat[a][b] += 1
            dict1.pop(row["שם מלא - Full Name"])
        else:
            a = 10
            b = row['Individual rating']
            mat[a][b] += 1

    for name, rating in dict1.items():
        a = rating
        b = 10
        mat[a][b] += 1

    transformation_matrix_df = pd.DataFrame(mat,
                                            index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'unfilled']
                                            , columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'unfilled'])
    normalized_df = transformation_matrix_df.div(transformation_matrix_df.sum(axis=1), axis=0).fillna(0)
    return normalized_df


print(transformation_matrix('8/27/2022', '9/3/2022'))


def filter_games(result_label, bound_bet_1, bound_bet_2):
    pass