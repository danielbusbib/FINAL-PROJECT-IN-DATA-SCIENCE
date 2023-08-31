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

    # bets win
    if result_label != 0:
        df = df[(((df['bet_team_home'] <= y) & (df['bet_team_home'] >= x)) & (df['team_home'] == "Hapoel Jerusalem")) |
                (((df['bet_team_away'] <= y) & (df['bet_team_away'] >= x)) & (df['team_away'] == "Hapoel Jerusalem"))]

    # bets draw
    else:
        df = df[((df['bet_draw'] <= y) & (df['bet_draw'] >= x))]

    # dates
    if date is not None:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        df.sort_values(by='date')
        df = df[(df['date'] >= pd.to_datetime(date[0], dayfirst=True)) &
                (df['date'] <= pd.to_datetime(date[1], dayfirst=True))]
    return df


def matrix_test(mat1, mat2, notes=[3, 4, 5, 6, 7]):
    T = 0
    df = (len(mat1) - 1) * (len(notes) - 1)
    # for n in notes:
    #     stat, p_value = multinomial_likelihood_ratio_test(mat1.iloc[n].values[notes], mat2.iloc[n].values[notes])
    #     T += stat
    print(T, '  B   ', chi2.cdf(T, df))
    test_p_value = 1 - chi2.cdf(T, df)
    return test_p_value


# def create_matrices_combinations():
#     results = [1, 0, -1]
#     str_results = ["win", "draw", "loss"]
#     dict_matrices = {}
#
#     def create_matrices(title, rows_condition, i):
#         t1 = transformation_matrix(rows_condition, k=10, name_csv=title + f"{str_results[i]}")
#         df = pd.read_csv('data.csv')
#         dates_to_delete = rows_condition["date"].values
#         overall_dates = df["date"].values
#         for date in overall_dates:
#             if date in dates_to_delete:
#                 df = df.drop(df[df['date'] == date].index)
#         t2 = transformation_matrix(df, k=10, name_csv=title + f'{str_results[i]}_complementary')
#         dict_matrices[title + f"{str_results[i]}"] = (t1, t2)
#
#     def surprise_matrix(title):
#         for i in range(len(results)):
#             if results[i] == -1:  # loss
#                 rows_condition = filters(x=0, y=3, result_label=results[i])
#             else:
#                 rows_condition = filters(x=3, y=100, result_label=results[i])
#             create_matrices(title, rows_condition, i)
#
#     def no_surprise_matrix(title):
#         for i in range(len(results)):
#             if results[i] == -1:  # loss
#                 rows_condition = filters(x=3, y=100, result_label=results[i])
#             else:
#                 rows_condition = filters(x=0, y=3, result_label=results[i])
#             create_matrices(title, rows_condition, i)
#
#     surprise_matrix(title="surprise_")
#     no_surprise_matrix(title="no_surprise_")
#
#     return dict_matrices


def create_matrices_combinations():
    results = [1, 0, -1]
    str_results = ["win", "draw", "loss"]
    dict_matrices = {}

    def create_matrices(title, rows_condition, i):
        t1 = transformation_matrix(rows_condition, k=10, name_csv=title + f"{str_results[i]}")
        df = pd.read_csv('data.csv')
        dates_to_delete = rows_condition["date"].values
        overall_dates = df["date"].values
        for date in overall_dates:
            if date in dates_to_delete:
                df = df.drop(df[df['date'] == date].index)
        t2 = transformation_matrix(df, k=10, name_csv=title + f'{str_results[i]}_complement')
        dict_matrices[title + f"{str_results[i]}"] = (t1, t2)

    def create_matrix_surprise(title, x_loss, y_loss, x_other, y_other):
        for i in range(len(results)):
            if results[i] == -1:  # loss
                rows_condition = filters(x=x_loss, y=y_loss, result_label=results[i])
            else:
                rows_condition = filters(x=x_other, y=y_other, result_label=results[i])
            create_matrices(title, rows_condition, i)

    # surprise_matrix
    create_matrix_surprise('matrix_surprise_', 0, 3, 3, 100)
    # no_surprise_matrix
    create_matrix_surprise('no_matrix_surprise_', 3, 100, 0, 3)

    positions_str = ["GK", "DF", "ATT", "MF"]

    # # # surprise_matrix per position
    # matrix_surprise_position('matrix_surprise_', 0, 3, 3, 100)
    # # no_surprise_matrix per position
    # matrix_surprise_position('no_matrix_surprise_', 3, 100, 0, 3)

    def matrix_position_result(title):
        for pos in positions_str:
            for i in range(len(results)):
                if results[i] == -1:  # loss
                    rows_condition = filters(x=0, y=3, result_label=results[i])
                else:
                    rows_condition = filters(x=3, y=100, result_label=results[i])

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
                                           name_csv=title + f"{str_results[i]}_{pos}_complement",
                                           name_players=name_players_col.tolist())
                dict_matrices[title + pos + '_' + str_results[i]] = (t1, t2)

    matrix_position_result("surprise_")
    return dict_matrices


def compress(mat, num_categories):
    if num_categories == 2:
        # L(1-5), H(6-10)
        # Divide the DataFrame into two parts
        n = 6
        df1 = mat.iloc[:n].sum()
        df2 = mat.iloc[n:11].sum()
        df = pd.DataFrame([df1, df2], index=['L', 'H']).T
        df1 = df.iloc[:n].sum()
        df2 = df.iloc[n:11].sum()
        return pd.DataFrame([df1, df2], index=['L', 'H']).T

    if num_categories == 3:
        df1 = mat.iloc[:4].sum()
        df2 = mat.iloc[4:7].sum()
        df3 = mat.iloc[7:11].sum()
        df = pd.DataFrame([df1, df2, df3], index=['L', 'M', 'H']).T
        df1 = df.iloc[:4].sum()
        df2 = df.iloc[4:7].sum()
        df3 = df.iloc[7:11].sum()
        return pd.DataFrame([df1, df2, df3], index=['L', 'M', 'H']).T


# ALL_GAMES = pd.read_csv('transformation_matrix_examples/transformation_basic.csv', index_col=0)
# ALL_GAMES = compress(ALL_GAMES, 3)
# print(ALL_GAMES)


def wilks_likelihood_test(note, A, B, num_categories):
    ALL_GAMES = pd.read_csv('transformation_matrix_examples/transformation_basic.csv', index_col=0)
    ALL_GAMES = compress(ALL_GAMES, num_categories=2)
    # print(ALL_GAMES)
    # A = pd.read_csv('transformation_matrix_examples/transformation_matrix_no_surprise_loss.csv', index_col=0)
    A = compress(A, num_categories=2)
    # print(A)
    # B = pd.read_csv('transformation_matrix_examples/transformation_matrix_no_surprise_loss_mashlim.csv', index_col=0)
    B = compress(B, num_categories=2)
    # print(B)
    # H0
    p_H0 = ALL_GAMES.iloc[note].values / (ALL_GAMES.iloc[note].values.sum())

    # H1
    if A.iloc[note].values.sum() == 0 or B.iloc[note].values.sum() == 0:
        return False, False

    p_A_H1 = A.iloc[note].values / (A.iloc[note].values.sum())
    p_B_H1 = B.iloc[note].values / (B.iloc[note].values.sum())

    L_H0, L_H1 = 1, 1
    for i in range(len(p_H0)):
        # LIKELIHOOD OF H0
        L_H0 *= p_H0[i] ** (ALL_GAMES.iloc[note].values[i])
        # LIKELIHOOD OF H1
        L_H1 *= p_A_H1[i] ** (A.iloc[note].values[i])
        L_H1 *= p_B_H1[i] ** (B.iloc[note].values[i])

    W = L_H1 / L_H0
    T = 2 * np.log(W)

    # find Chi-Square critical value
    M = scipy.stats.chi2.ppf(1 - .05, df=len(p_H0) - 1)
    # Calculate the p-value based on the chi-square statistic T
    p_value = 1 - scipy.stats.chi2.cdf(T, df=len(p_H0) - 1)

    if T >= M:
        print("H1 IS TRUE.")
        return p_value, 'YES'
    else:
        print("H0 IS TRUE.")
        return p_value, 'NO'


def table():
    matrices_combinations = create_matrices_combinations()
    df = pd.DataFrame(
        columns=['what do we check ?', 'all the grades/compress', 'matrix/vector', 'p-value', 'H1 is true'])

    for name in matrices_combinations:
        parts = name.split('_')
        if 'matrix' in parts:
            parts.remove('matrix')
        check_name = ' '.join(parts[:])
        num_categories = 2
        p_value, result = wilks_likelihood_test(note=1, A=matrices_combinations[name][0],
                                                B=matrices_combinations[name][1],
                                                num_categories=num_categories)
        if not result:
            continue

        new_row = {
            'what do we check ?': [check_name],
            'all the grades/compress': ['vector'],
            'matrix/vector': [p_value],
            'p-value': [p_value],
            'H1 is true': [result]
        }

        # Convert the new_row dictionary into a DataFrame
        new_row_df = pd.DataFrame(new_row, columns=df.columns)

        # Concatenate the existing DataFrame and the new row DataFrame
        df = pd.concat([df, new_row_df], ignore_index=True)

    # Write DataFrame to CSV file
    csv_filename = 'table_combinations_results.csv'
    df.to_csv(csv_filename, index=False)
    return df


table()


# print(dicti)
# for i in range(1, 11):
#     print(i)
# print(wilks_likelihood_test(note=1))


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
