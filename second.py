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
        cols_to_drop_players = df.columns[df.columns.get_loc('adeley_adebayo'):].values
        ll = [col for col in cols_to_drop_players if col not in players]
        df.drop(columns=ll, inplace=True)
        # print(df.columns)
        # print(players)
        # print([col for col in cols_to_drop_players if col not in players])

    # result
    if result_label is not None:
        df.drop(df[df['result_label'] != result_label].index, inplace=True)

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


def matrix_test(mat1, mat2, notes):
    print(mat1)
    print()
    print(mat2)
    T = 0
    df = 9 * (len(notes) - 1)
    for n in notes:
        stat, p_value = likelihood_ratio_multinomial(mat1.iloc[n].values[1:11], mat2.iloc[n].values[1:11])
        T += stat
    test_p_value = 1 - chi2.cdf(T, df)
    return test_p_value


def create_transformation_matrix_combinations():
    combinations = list()
    # generate regular matrix without any filter
    reg = transformation_matrix(filters(), k=10, name_csv='transformation_basic')
    # combinations.append(reg)
    results = [1, 0, -1]
    str_results = ["win", "draw", "loss"]

    def matrix_result(x, y, title):
        for i in range(len(results)):
            rows_condition = filters(x=x, y=y, result_label=results[i])
            t1 = transformation_matrix(rows_condition, k=10, name_csv=title + f"{str_results[i]}")
            df = pd.read_csv('data.csv')
            dates_to_delete = rows_condition["date"]
            overall_dates = df["date"]
            for date in overall_dates:
                if date in dates_to_delete:
                    df = df.drop(df[df['date'] == date].index)
            t2 = transformation_matrix(df, k=10, name_csv=title + f'{str_results[i]}_mashlim')
            combinations.append((t1, t2))

    # generate for each result label a transformation matrix when the result was expected
    matrix_result(0, 3, "transformation_matrix_no_surprise_")

    # generate for each result label a transformation matrix when the result was not expected
    matrix_result(3, 100, "transformation_matrix_surprise_")

    positions_str = ["GK", "DF", "ATT", "MF"]

    def matrix_position():
        for pos in positions_str:
            players = [players for players in config.POSITION.keys() if config.POSITION[players] == pos]
            t1 = transformation_matrix(filters(players=players), k=10,
                                       name_csv=f"transformation_matrix_{pos}", name_players=players)
            df = pd.read_csv('data.csv')
            df = df.drop(columns=players)
            index = df.columns.get_loc('result_label') + 1
            name_players_col = df.columns[index:]
            t2 = transformation_matrix(df, k=10, name_csv=f"transformation_matrix_{pos}_mashlim",
                                       name_players=name_players_col.tolist())
            combinations.append((t1, t2))

    # generate transformation matrix for each position on the pitch
    # matrix_position()

    def matrix_position_result(x, y, title):
        for pos in positions_str:
            for i in range(len(results)):
                rows_condition = filters(x=x, y=y, result_label=results[i])
                players = [players for players in config.POSITION.keys() if config.POSITION[players] == pos]
                t1 = transformation_matrix(rows_condition, k=10,
                                           name_csv=title + f'{str_results[i]}_{pos}.csv',
                                           name_players=players)
                df = pd.read_csv('data.csv')
                df = df.drop(columns=players)
                index = df.columns.get_loc('result_label') + 1
                name_players_col = df.columns[index:]
                dates_to_delete = rows_condition["date"]
                overall_dates = df["date"]
                for date in overall_dates:
                    if date in dates_to_delete:
                        df = df.drop(df[df['date'] == date].index)
                t2 = transformation_matrix(df, k=10,
                                           name_csv=title + f"{str_results[i]}_{pos}_mashlim",
                                           name_players=name_players_col.tolist())
                combinations.append((t1, t2))

    # # generate transformation matrix for each position on the pitch with expected result
    # matrix_position_result(0, 3, "transformation_matrix_surprise_")
    # # generate transformation matrix for each position on the pitch with not expected result
    # matrix_position_result(3, 100, "transformation_matrix_surprise_")

    return combinations


# l = create_transformation_matrix_combinations()
# d = l[0][0]
reg = transformation_matrix(filters(), k=10, name_csv='transformation_basic')
print(reg)
# print(likelihood_ratio_multinomial(reg.iloc[6].values[:11], reg.iloc[8].values[:11]))
# p_value = matrix_test(l[0][0], l[0][1], notes=[6, 7, 8])
# print(f"P-value: {p_value}")


def compress(mat, num_categories=2):
    if num_categories == 2:
        # L(1-5), H(6-10)
        # Divide the DataFrame into two parts
        n = 6
        df1 = mat.iloc[:n].sum()
        df2 = mat.iloc[n:].sum()
        return pd.DataFrame([df1, df2], index=['L', 'H'])

    if num_categories == 4:
        df1 = mat.iloc[:5].sum()
        df2 = mat.iloc[5:6].sum()
        df3 = mat.iloc[6:7].sum()
        df4 = mat.iloc[7:].sum()
        return pd.DataFrame([df1, df2, df3, df4], index=['L', 'ML', 'MH', 'H'])

# mat1 = transformation_matrix(filters(x=0, y=3), k=10, name_csv='transformation_matrix_no_surprise_win.csv')
# print(mat1)
# print()
# print(compress(mat1, num_categories=4))
# mat2 = transformation_matrix(filters(x=3, y=100), k=10, name_csv='transformation_matrix_surprise_win.csv')
#
# # print(mat1.iloc[3].values[:11])
# # stat, p_value = likelihood_ratio_multinomial(mat1.iloc[5].values[:11], mat1.iloc[6].values[:11])
# # print(f"Test statistic: {stat}")
# p_value = matrix_test(mat1, mat2, notes=[3, 4, 5, 6, 7, 8])
# print(f"P-value: {p_value}")
# print(transformation_matrix(filters(players=list(config.PLAYERS.values())[:10]), k=20,
#                             name_players=list(config.PLAYERS.values())[:10]))
# date=['04/03/2023', '18/04/2023']

# create_transformation_matrix_combinations()
