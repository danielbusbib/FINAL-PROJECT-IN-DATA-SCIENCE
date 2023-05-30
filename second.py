import numpy as np
import pandas as pd
from scipy.stats import multinomial
from scipy.stats import chi2

# Transformation matrix
import config


def transformation_matrix(data, k, name_players=None, name_csv=""):
    """
    data -> data frame of games
    k -> min minutes played filter
    name_players -> list of name players
    """
    df = pd.read_csv("RPE_data.csv")
    if name_players is None:
        name_players = list(config.PLAYERS.values())

    # data = pd.read_csv("data.csv")
    mat = np.zeros((13, 13)).astype(int)
    cols = [f'{i}' for i in range(11)] + ['X', 'Y']  # Y -> NOT PLAYED UNDER K MIN

    # iterate over each pair of game
    for index_game in range(data.shape[0] - 1):
        # df by games
        df_game1 = data.iloc[index_game]
        df_game2 = data.iloc[index_game + 1]

        df_rpe1 = df.loc[df['date'].str.contains(str(df_game1['date']))]
        df_rpe2 = df.loc[df['date'].str.contains(str(df_game2['date']))]

        players_dict = {i: ['Y', 'Y'] for i in name_players}

        for player in name_players:
            # notes game 1
            if not df_rpe1.empty:
                time = df_game1[player]
                if 'yy' in str(time):
                    time = time[:time.find('yy')]

                elif 'y' in str(time):
                    time = time[:time.find('y')]

                if int(time) >= k:
                    note = df_rpe1[player + '_individual_rating'].values[0]
                    if note == -1:
                        players_dict[player][0] = 'X'
                    else:
                        players_dict[player][0] = note

            # notes game 2
            if df_rpe2.empty:
                break
            time = df_game2[player]
            if 'yy' in str(time):
                time = time[:time.find('yy')]

            elif 'y' in str(time):
                time = time[:time.find('y')]

            if int(time) >= k:
                note = df_rpe2[player + '_individual_rating'].values[0]
                if note == -1:
                    players_dict[player][1] = 'X'
                else:
                    players_dict[player][1] = note
        # add to transformation matrix
        for val in players_dict.values():
            mat[cols.index(str(val[0]))][cols.index(str(val[1]))] += 1
    transformation_matrix_df = pd.DataFrame(mat, index=cols, columns=cols)

    # normalize
    # transformation_matrix_df = transformation_matrix_df.div(transformation_matrix_df.sum(axis=1), axis=0).fillna(0)

    # save to csv
    transformation_matrix_df.to_csv(f'transformation_matrix_examples/{name_csv}.csv',
                                    index=cols, encoding='utf-8')

    return transformation_matrix_df


def filters(x=0, y=100, result_label=None, players=None, date=None):
    df = pd.read_csv('data.csv')
    # players
    if players is not None:
        cols_to_drop_players = df.columns[df.columns.get_loc('adeley_adebayo'):]
        df = df.drop(columns=[col for col in cols_to_drop_players if col not in players])

    # result
    if result_label is not None:
        df = df.drop(df[df['result_label'] != result_label].index, inplace=True)

    # bets
    df = df[(((df['bet_team_home'] <= y) & (df['bet_team_home'] >= x)) & (df['team_home'] == "Hapoel Jerusalem")) |
            (((df['bet_team_away'] <= y) & (df['bet_team_away'] >= x)) & (df['team_away'] == "Hapoel Jerusalem"))]

    # dates
    if date is not None:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        df.sort_values(by='date')
        df = df[(df['date'] >= pd.to_datetime(date[0], dayfirst=True)) &
                (df['date'] <= pd.to_datetime(date[1], dayfirst=True))]
    return df


def likelihood_test(vector1, vector2):
    # Compute the total number of observations
    n = sum(vector1)

    # Perform the multinomial test
    p_value = multinomial.pmf([vector2], n=n, p=[vector1 / n])

    # Print the p-value
    print("p-value:", p_value)


def likelihood_ratio_multinomial(observed1, observed2):
    """
    Perform a likelihood ratio multinomial test on two vectors of different sums.

    Arguments:
    observed1 -- observed frequencies in vector 1 (numpy array or list)
    observed2 -- observed frequencies in vector 2 (numpy array or list)

    Returns:
    stat -- test statistic (chi-square value)
    p_value -- p-value associated with the test statistic
    """
    # Calculate the sums of the observed frequencies
    sum1 = np.sum(observed1)
    sum2 = np.sum(observed2)
    observed1 = observed1 + 1e-10
    # Calculate the expected frequencies assuming the null hypothesis of equal probabilities
    expected = (sum1 * observed1 + sum2 * observed2) / (sum1 + sum2)

    # Calculate the test statistic (likelihood ratio)
    stat = 2 * np.sum(observed1 * np.log(observed1 / expected))

    # Calculate the degrees of freedom
    df = len(observed1) - 1

    # Calculate the p-value associated with the test statistic
    p_value = 1 - chi2.cdf(stat, df)

    return stat, p_value


mat1 = transformation_matrix(filters(x=0, y=3), k=10, name_csv='transformation_matrix_no_surprise_win.csv')
mat2 = transformation_matrix(filters(x=3, y=100), k=10, name_csv='transformation_matrix_surprise_win.csv')


stat, p_value = likelihood_ratio_multinomial(mat1.iloc[5].values, mat2.iloc[5].values)
print(f"Test statistic: {stat}")
print(f"P-value: {p_value}")
# print(transformation_matrix(filters(players=list(config.PLAYERS.values())[:10]), k=20,
#                             name_players=list(config.PLAYERS.values())[:10]))
# date=['04/03/2023', '18/04/2023']
