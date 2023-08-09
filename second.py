import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import multinomial, chi2_contingency
from scipy.stats import chi2
import matplotlib.pyplot as plt
import scipy.stats

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
    # print(transformation_matrix_df)

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


def likelihood_ratio_test(vector1, vector2, alpha):
    # Calculate the total counts for each vector
    n1 = np.sum(vector1)
    n2 = np.sum(vector2)

    # Calculate the probabilities for each category
    p1 = vector1 / n1
    p2 = vector2 / n2

    # Calculate the likelihood under H0 (same distribution)
    L0 = comb(n1, vector1).prod() * (p1 ** vector1).prod()

    # Calculate the likelihood under H1 (different distributions)
    L1 = comb(n2, vector2).prod() * (p2 ** vector2).prod()

    # Calculate the likelihood ratio
    LR = L1 / L0

    # Calculate the critical value from the chi-squared distribution
    df = len(vector1) - 1  # Degrees of freedom
    critical_value = chi2.ppf(1 - alpha, df)

    # Compare the likelihood ratio to the critical value
    if LR > critical_value:
        return "Reject H0: The two vectors do not come from the same distribution."
    else:
        return "Fail to reject H0: The two vectors may come from the same distribution."


def matrix_test(mat1, mat2, notes=[3, 4, 5, 6, 7]):
    # print(mat1)
    # print()
    # print(mat2)
    T = 0
    df = 9 * (len(notes) - 1)
    # for n in notes:
    #     stat, p_value = multinomial_likelihood_ratio_test(mat1.iloc[n].values[notes], mat2.iloc[n].values[notes])
    #     T += stat
    print(T, '  B   ', chi2.cdf(T, df))
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
            dates_to_delete = rows_condition["date"].values
            overall_dates = df["date"].values
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
                dates_to_delete = rows_condition["date"].values
                overall_dates = df["date"].values
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


def wilks_likelihood_test(note):
    ALL_GAMES = pd.read_csv('transformation_matrix_examples/transformation_basic.csv', index_col=0)
    ALL_GAMES = compress(ALL_GAMES, 2)

    A = pd.read_csv('transformation_matrix_examples/transformation_matrix_no_surprise_loss.csv', index_col=0)
    A = compress(A, 2)

    B = pd.read_csv('transformation_matrix_examples/transformation_matrix_no_surprise_loss_mashlim.csv', index_col=0)
    B = compress(B, 2)

    # H0
    p_H0 = ALL_GAMES.iloc[note].values[1:11] / (ALL_GAMES.iloc[note].values[1:11].sum())
    # H1
    p_A_H1 = A.iloc[note].values[1:11] / (A.iloc[note].values[1:11].sum())
    p_B_H1 = B.iloc[note].values[1:11] / (B.iloc[note].values[1:11].sum())

    L_H0, L_H1 = 1, 1
    for i in range(len(p_H0)):
        L_H0 *= p_H0[i] ** (ALL_GAMES.iloc[note].values[1:11][i])
        L_H1 *= p_A_H1[i] ** (A.iloc[note].values[1:11][i])
        L_H1 *= p_B_H1[i] ** (B.iloc[note].values[1:11][i])
    W = L_H1 / L_H0
    T = 2 * np.log(W)

    # find Chi-Square critical value
    M = scipy.stats.chi2.ppf(1 - .05, df=9)
    if T >= M:
        print("H1 IS TRUE.")
    else:
        print("H0 IS TRUE.")


# for i in range(1, 11):
#     print(i)
# wilks_likelihood_test(note=0)


def plot_p_values_heatmap():
    reg = transformation_matrix(filters(), k=10, name_csv='transformation_basic')
    # Generate the heatmap matrix
    heatmap = np.zeros((10, 10))
    for a in range(10):
        for b in range(10):
            print('need fix')
            # heatmap[a, b] = \
            #     multinomial_likelihood_ratio_test(reg.iloc[a + 1].values[1:11], reg.iloc[b + 1].values[1:11])[1] < 0.05

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(heatmap, cmap=plt.get_cmap('viridis'))

    ax.set_title("P-value")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    # And to label them with the relevant parties names
    ax.set_xticklabels(range(1, 11), rotation=90)
    ax.set_yticklabels(range(1, 11))
    ax.set_xlabel("note")
    ax.set_ylabel("note")

    # # Loop over data dimensions to create text annotations.
    # for i in range(n_21):
    #     for j in range(n_22):
    #         text = ax.text(j, i, round(M[i, j], 3), ha="center", va="center", color="w")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('votes probability', rotation=-90, va="bottom")

    plt.show()

# d = l[0][0]
# reg = l[0]
# reg = transformation_matrix(filters(), k=10, name_csv='transformation_basic')


# print(reg.iloc[5].values[:11])
# print(multinomial_likelihood_ratio_test(reg.iloc[6].values[1:11], reg.iloc[5].values[1:11]))
# print(likelihood_ratio_test(reg.iloc[5].values[:11], reg.iloc[7].values[:11], 0.05))
# p_value = matrix_test(l[0][0], l[0][1], notes=[6, 7, 8])
# print(f"P-value: {p_value}")

# combinations = create_transformation_matrix_combinations()
# win_mat, no_win_mat = combinations[0]
# print(win_mat)
# print(no_win_mat)
# # print(combinations[1][1])
# print(matrix_test(win_mat, no_win_mat))


# mat1 = transformation_matrix(filters(x=0, y=3), k=10, name_csv='transformation_matrix_no_surprise_win.csv')
# # print(mat1)
# # print()
# # print(compress(mat1, num_categories=4))
# mat2 = transformation_matrix(filters(x=3, y=100), k=10, name_csv='transformation_matrix_surprise_win.csv')
# #
# # # print(mat1.iloc[3].values[:11])
# stat, p_value = likelihood_ratio_multinomial(mat1.iloc[5].values[:11], mat1.iloc[6].values[:11])
# # print(f"Test statistic: {stat}")
# p_value = matrix_test(mat1, mat2)
# print(f"P-value: {p_value}")
# print(transformation_matrix(filters(players=list(config.PLAYERS.values())[:10]), k=20,
#                             name_players=list(config.PLAYERS.values())[:10]))
# date=['04/03/2023', '18/04/2023']

# create_transformation_matrix_combinations()
