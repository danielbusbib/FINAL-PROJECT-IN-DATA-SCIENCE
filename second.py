import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import multinomial
import scipy.stats

# Import configuration settings from the 'config' module
import config


def transformation_matrix(data, k, name_players=None, name_csv=""):
    """
    Generate a transformation matrix based on player performance data.

    Parameters:
    data -> data frame of games
    k -> minimum minutes played filter
    name_players -> list of name players
    name_csv -> name for the resulting CSV file

    Returns:
    transformation_matrix_df -> the transformation matrix as a DataFrame
    """

    # Load the player performance data from RPE csv file
    df = pd.read_csv("RPE_data.csv")

    # If the list of player names is not provided, use default names from the config module
    if name_players is None:
        name_players = list(config.PLAYERS.values())

    # Initialize an empty matrix to store transformation values
    mat = np.zeros((13, 13)).astype(int)

    # Define column names for the transformation matrix
    cols = [f'{i}' for i in range(11)] + ['X', 'Y']  # Y -> NOT PLAYED UNDER K MIN

    # Iterate over each pair of consecutive games
    for index_game in range(data.shape[0] - 1):
        # Retrieve data for the two games in the pair
        df_game1 = data.iloc[index_game]
        df_game2 = data.iloc[index_game + 1]

        # Filter RPE data for the specific dates of these games
        df_rpe1 = df.loc[df['date'].str.contains(str(df_game1['date']))]
        df_rpe2 = df.loc[df['date'].str.contains(str(df_game2['date']))]

        # Initialize a dictionary to store player notes for both games
        players_dict = {i: ['Y', 'Y'] for i in name_players}

        for player in name_players:
            # Process notes for game 1
            if not df_rpe1.empty:
                time = df_game1[player]
                if 'yy' in str(time):
                    time = time[:time.find('yy')]
                elif 'y' in str(time):
                    time = time[:time.find('y')]

                # Check if player played for at least 'k' minutes
                if int(time) >= k:
                    note = df_rpe1[player + '_individual_rating'].values[0]
                    if note == -1:
                        players_dict[player][0] = 'X'
                    else:
                        players_dict[player][0] = note

            # Process notes for game 2
            if df_rpe2.empty:
                break
            time = df_game2[player]
            if 'yy' in str(time):
                time = time[:time.find('yy')]
            elif 'y' in str(time):
                time = time[:time.find('y')]

            # Check if player played for at least 'k' minutes
            if int(time) >= k:
                note = df_rpe2[player + '_individual_rating'].values[0]
                if note == -1:
                    players_dict[player][1] = 'X'
                else:
                    players_dict[player][1] = note

        # Update the transformation matrix based on player notes
        for val in players_dict.values():
            mat[cols.index(str(val[0]))][cols.index(str(val[1]))] += 1

    # Create a DataFrame from the transformation matrix
    transformation_matrix_df = pd.DataFrame(mat, index=cols, columns=cols)

    # Normalize the matrix (optional)
    # transformation_matrix_df = transformation_matrix_df.div(transformation_matrix_df.sum(axis=1), axis=0).fillna(0)

    # Save the transformation matrix as a CSV file
    transformation_matrix_df.to_csv(f'transformation_matrix_examples/{name_csv}.csv',
                                    index=cols, encoding='utf-8')

    return transformation_matrix_df


def filters(x=0, y=100, result_label=None, players=None, date=None):
    """
    Filter and manipulate a DataFrame of game data based on various criteria.

    Parameters:
    x -> Lower bound for filtering bets
    y -> Upper bound for filtering bets
    result_label -> Label for filtering game results (WIN, DRAW, or LOSS)
    players -> List of player names to filter columns
    date -> Date range for filtering game dates

    Returns:
    df -> Filtered DataFrame based on the specified criteria
    """

    # Read the game data from an external CSV file into a DataFrame
    df = pd.read_csv('data.csv')

    # Filter columns based on the list of player names
    if players is not None:
        # Identify columns related to player names
        cols_to_drop_players = df.columns[df.columns.get_loc('adeley_adebayo'):].values
        # Create a list of columns to drop if they are not in the provided player names list
        ll = [col for col in cols_to_drop_players if col not in players]
        # Remove the identified columns from the DataFrame
        df.drop(columns=ll, inplace=True)

    # Filter rows based on the specified game result label (WIN, DRAW, or LOSS)
    if result_label is not None:
        df.drop(df[df['result_label'] != result_label].index, inplace=True)

    # Filter rows based on betting criteria for wins or draws
    if result_label != 0:
        df = df[(((df['bet_team_home'] <= y) & (df['bet_team_home'] >= x)) & (df['team_home'] == "Hapoel Jerusalem")) |
                (((df['bet_team_away'] <= y) & (df['bet_team_away'] >= x)) & (df['team_away'] == "Hapoel Jerusalem"))]

    # Filter rows based on betting criteria for draws
    else:
        df = df[((df['bet_draw'] <= y) & (df['bet_draw'] >= x))]

    # Filter rows based on the specified date range
    if date is not None:
        # Convert the 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        # Sort the DataFrame by date
        df.sort_values(by='date')
        # Filter rows within the specified date range
        df = df[(df['date'] >= pd.to_datetime(date[0], dayfirst=True)) &
                (df['date'] <= pd.to_datetime(date[1], dayfirst=True))]

    return df


def create_matrices_combinations():
    """
    Create and compare transformation matrices for different combinations of game data.
    Returns:
    dict_matrices -> A dictionary containing pairs of transformation matrices to compare
    """

    # Define constants for game results
    WIN = 1
    DRAW = 0
    LOSS = -1

    # Initialize an empty dictionary to store pairs of transformation matrices
    dict_matrices = {}

    def create_matrices(title_a, title_b, rows_condition):
        """
        Create two transformation matrices and store them in the dictionary.

        Parameters:
        title_a -> Name of the first transformation matrix
        title_b -> Name of the second transformation matrix
        rows_condition -> DataFrame containing rows that meet specific criteria
        """
        # Create the first transformation matrix based on the provided rows_condition
        t1 = transformation_matrix(rows_condition, k=10, name_csv=title_a)
        df = pd.read_csv('data.csv')

        # Extract dates to delete from the rows_condition DataFrame
        dates_to_delete = rows_condition["date"].values

        # Extract all dates from the original DataFrame
        overall_dates = df["date"].values

        # Remove rows with dates that match those in dates_to_delete
        for date in overall_dates:
            if date in dates_to_delete:
                df = df.drop(df[df['date'] == date].index)

        # Create the second transformation matrix based on the updated DataFrame
        t2 = transformation_matrix(df, k=10, name_csv=title_b)

        # Store the pair of transformation matrices in the dictionary
        dict_matrices[f"{title_a} VS {title_b}"] = (t1, t2)

    def create_normal():
        # Create transformation matrices for the comparison of win vs. loss/draw
        rows_condition = filters(result_label=WIN)
        create_matrices(title_a='win',
                        title_b='draw_and_loss',
                        rows_condition=rows_condition)

        # Create transformation matrices for the comparison of loss vs. draw/win
        rows_condition = filters(result_label=LOSS)
        create_matrices(title_a='loss',
                        title_b='draw_and_win',
                        rows_condition=rows_condition)

    def create_matrices_win():
        # Create transformation matrices for the comparison of surprise win vs. no surprise win
        title_a = 'surprise_win'
        rows_condition_1 = filters(x=3, y=100, result_label=WIN)
        t1 = transformation_matrix(rows_condition_1, k=10, name_csv=title_a)

        title_b = 'no_surprise_win'
        rows_condition_2 = filters(x=0, y=3, result_label=WIN)
        t2 = transformation_matrix(rows_condition_2, k=10, name_csv=title_b)

        # Store the pair of transformation matrices in the dictionary
        dict_matrices[f"{title_a} VS {title_b}"] = (t1, t2)

    def create_matrices_loss():
        # Create transformation matrices for the comparison of surprise loss vs. no surprise loss
        title_a = 'surprise_loss'
        rows_condition_1 = filters(x=0, y=3, result_label=LOSS)
        t1 = transformation_matrix(rows_condition_1, k=10, name_csv=title_a)

        title_b = 'no_surprise_loss'
        rows_condition_2 = filters(x=3, y=100, result_label=LOSS)
        t2 = transformation_matrix(rows_condition_2, k=10, name_csv=title_b)

        # Store the pair of transformation matrices in the dictionary
        dict_matrices[f"{title_a} VS {title_b}"] = (t1, t2)

    # Call the functions to create transformation matrices and populate the dictionary
    create_normal()
    create_matrices_win()
    create_matrices_loss()

    # Return the dictionary containing pairs of transformation matrices
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


def compress(mat, num_categories):
    """
    Compress a transformation matrix into a specified number of categories.

    Parameters:
    mat -> The input transformation matrix to be compressed.
    num_categories -> The number of categories to compress the matrix into (2 or 3).

    Returns:
    compressed_matrix -> The compressed transformation matrix.
    """
    if num_categories == 2:
        # For 2 categories: L(1-5), H(6-10)
        n = 6

        df1 = mat.iloc[:n].sum()
        df2 = mat.iloc[n:11].sum()
        # Create a new DataFrame with 'L' and 'H' categories
        compressed_matrix = pd.DataFrame([df1, df2], index=['L', 'H']).T

        df1 = compressed_matrix.iloc[:n].sum()
        df2 = compressed_matrix.iloc[n:11].sum()
        # Return the final compressed matrix
        return pd.DataFrame([df1, df2], index=['L', 'H']).T

    if num_categories == 3:
        # For 3 categories: L(1-4), M(5-7), H(8-10)
        df1 = mat.iloc[:4].sum()
        df2 = mat.iloc[4:7].sum()
        df3 = mat.iloc[7:11].sum()

        # Create a new DataFrame with 'L', 'M', and 'H' categories
        compressed_matrix = pd.DataFrame([df1, df2, df3], index=['L', 'M', 'H']).T
        df1 = compressed_matrix.iloc[:4].sum()
        df2 = compressed_matrix.iloc[4:7].sum()
        df3 = compressed_matrix.iloc[7:11].sum()

        # Return the final compressed matrix
        return pd.DataFrame([df1, df2, df3], index=['L', 'M', 'H']).T


def wilks_likelihood_test(A, B, num_categories=10):
    """
    Perform a Wilks' likelihood ratio test to compare two transformation matrices.

    This function conducts a hypothesis test to determine whether two transformation
    matrices (A and B) significantly differ from each other using Wilks' likelihood
    ratio test. The test is performed for a specified number of categories.

    Parameters:
    A -> The first transformation matrix for comparison.
    B -> The second transformation matrix for comparison.
    num_categories -> The number of categories used for the test (default is 10).

    Returns:
    p_value -> The p-value from the likelihood ratio test.
    result -> A string indicating whether H0 or H1 is true ('YES' for H1, 'NO' for H0).
    """
    # Load the reference transformation matrix (ALL_GAMES)
    ALL_GAMES = pd.read_csv('transformation_matrix_examples/transformation_basic.csv', index_col=0)

    if num_categories == 10:
        # If 10 categories are used, no further processing is needed
        ALL_GAMES = ALL_GAMES.iloc[:11].T
        A = A.iloc[:11].T
        B = B.iloc[:11].T
    else:
        # Compress the matrices to the specified number of categories
        ALL_GAMES = compress(ALL_GAMES, num_categories=num_categories)
        A = compress(A, num_categories=num_categories)
        B = compress(B, num_categories=num_categories)

    T_TOTAL = 0
    for note in range(A.shape[0]):
        # Calculate likelihoods for H0 and H1
        p_H0 = ALL_GAMES.iloc[note].values / (ALL_GAMES.iloc[note].values.sum())

        if A.iloc[note].values.sum() == 0 or B.iloc[note].values.sum() == 0:
            continue

        p_A_H1 = A.iloc[note].values / (A.iloc[note].values.sum())
        p_B_H1 = B.iloc[note].values / (B.iloc[note].values.sum())

        L_H0, L_H1 = 1, 1
        for i in range(len(p_H0)):
            # Calculate likelihoods for H0 and H1
            L_H0 *= p_H0[i] ** (ALL_GAMES.iloc[note].values[i])
            L_H1 *= p_A_H1[i] ** (A.iloc[note].values[i])
            L_H1 *= p_B_H1[i] ** (B.iloc[note].values[i])

        # Calculate the test statistic T
        W = L_H1 / L_H0
        T = 2 * np.log(W)
        T_TOTAL += T

    # Find the Chi-Square critical value
    M = scipy.stats.chi2.ppf(1 - 0.05, df=(A.shape[1] - 1) * A.shape[0])

    # Calculate the p-value based on the chi-square statistic T
    # note: p value can be 0 sometimes because of the software numerical stability
    p_value = 1 - scipy.stats.chi2.cdf(T_TOTAL, df=(A.shape[1] - 1) * A.shape[0])

    if T_TOTAL >= M:
        print("H1 IS TRUE.")
        return p_value, 'YES'
    else:
        print("H0 IS TRUE.")
        return p_value, 'NO'


def table():
    """
    Generate and save a table of results from Wilks' likelihood ratio tests.

    Returns:
    df -> A pandas DataFrame containing the table of results.
    """
    # Generate combinations of transformation matrices
    matrices_combinations = create_matrices_combinations()

    # Initialize an empty DataFrame with column names
    df = pd.DataFrame(
        columns=['what do we check ?', 'num categories', 'p-value', 'H1 is true'])

    # Loop through different numbers of categories (2 and 3)
    for num_categories in [2, 3]:
        for name in matrices_combinations:
            parts = name.split('_')

            # Remove 'matrix' from the name parts
            if 'matrix' in parts:
                parts.remove('matrix')
            check_name = ' '.join(parts[:])

            # Perform Wilks' likelihood ratio test and obtain results
            p_value, result = wilks_likelihood_test(A=matrices_combinations[name][0],
                                                    B=matrices_combinations[name][1],
                                                    num_categories=num_categories)

            # Skip rows where the result is not significant
            if not result:
                continue

            # Create a new row with test results
            new_row = {
                'what do we check ?': [check_name],
                'num categories': [num_categories],
                'p-value': [p_value],
                'H1 is true': [result]
            }

            # Convert the new_row dictionary into a DataFrame
            new_row_df = pd.DataFrame(new_row, columns=df.columns)

            # Concatenate the existing DataFrame and the new row DataFrame
            df = pd.concat([df, new_row_df], ignore_index=True)

    # Write the DataFrame to a CSV file
    csv_filename = 'table_combinations_results.csv'
    df.to_csv(csv_filename, index=False)
    return df


if __name__ == "__main__":
    # create transformation matrix off all games
    # df = pd.read_csv('data.csv')
    # transformation_matrix(data=df, k=10, name_csv='transformation_basic')

    # Execute the table function when the script is run
    table()
