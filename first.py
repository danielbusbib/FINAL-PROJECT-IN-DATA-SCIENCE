# Imports necessary libraries/modules
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import csv
import re
import typing
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd

# Import configuration settings from the 'config' module
import config


def get_bets(day: int, month: int, year: int):
    """Fetches betting information from the telesport.il website for a specific date."""
    # Configure Selenium to run in headless mode
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    # Construct the URL for the specific date
    driver.get(
        f'https://www.telesport.co.il/%D7%90%D7%96%D7%95%D7%A8%20%D7%95%D7%95%D7%99%D7%A0%D7%A8#110299/198/'
        f'{year}-{month}-{day}T18:30:00')

    # Extract the HTML content of the page
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    all_text = soup.get_text()
    text_lines = all_text.split('\n')

    # Parsing lines to extract betting information
    flag = False
    for line in text_lines:
        if 'כדורסל' in line:
            flag = True
        if 'כדורגל' in line:
            flag = False
        if not flag and 'הפועל ירושלים' in line and 'כדורסל' not in line:
            line = line.replace(' ', '')
            bets = line[line.index('D') + 1:]
            bet_home = bets[:4]
            bet_draw = bets[4:8]
            bet_away = bets[8:12]
            # Return betting odds
            return bet_home, bet_draw, bet_away


def parse_game(season_id: int, game_id: int, home_game: bool):
    """Parses information about a football game, including player statistics."""
    url = f"https://www.football.org.il/leagues/games/game/?season_id={season_id}&game_id={game_id}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    coaches = list()
    # Extract information about home team's coach
    for coach_line in soup.find_all('div', {'aria-labelledby': 'GAME_COACH_HOME'}):
        coaches.append(coach_line.find('b', {'class': 'name'}).text)
    if len(coaches) == 0:
        coaches.append('')

    # Extract information about guest team's coach
    for coach_line in soup.find_all('div', {'aria-labelledby': 'GAME_COACH_GUEST'}):
        if coach_line.find('b', {'class': 'name'}).text:
            coaches.append(coach_line.find('b', {'class': 'name'}).text)
    if len(coaches) == 1:
        coaches.append('')

    # Initialize lists to store player data
    lst = list(config.PLAYERS.keys())
    lst_time = [0] * len(lst)

    title = 'קבוצה - ביתית' if home_game else 'קבוצה אורחת'
    bench = 'home Bench' if home_game else 'guest Bench'

    # get max time recorded
    max_time = 0
    for row in soup.find_all('div', {'class': 'node number'}):
        max_time = int(row.find('span', {'class': ''}).text)

    # Create a set to store players who did not play in the match
    not_played = set()
    for row in soup.find_all('div', {'class': bench}):
        for row1 in row.find_all('a', {'title': title}):
            name = row1.find('span', {'class': 'name'}).text
            if "-" in name:
                name = name[:name.find(' -')]
            not_played.add(name)

    # Extract data for players who played in the match
    for row_match in soup.find_all('a', {'title': title}):
        for row in row_match.find_all('div', {'class': 'player'}):
            if str(row.find('span', {'class': 'number'}).text).isalpha():  # Check for coaches or referees
                break
            number = re.findall(r'\d+', row.find('span', {'class': 'number'}).text)[0]
            name = row.find('span', {'class': 'name'}).text
            if "-" in name:  # captain case
                name = name[:name.find(' -')]
            if name in not_played:
                continue

            time_on_pitch = max_time
            if row.find('span', {'class': 'change-up'}):
                time_on_pitch = max_time - int(re.findall(r'\d+', row.find('span', {'class': 'change-up'}).text)[0])
            elif row.find('span', {'class': 'change-down'}):
                time_on_pitch = re.findall(r'\d+', row.find('span', {'class': 'change-down'}).text)[0]

            first_yellow_card_minute = second_yellow_card_minute = ''  # Check for yellow cards
            if row.find('span', {'class': 'yellow'}):
                first_yellow_card_minute = re.findall(r'\d+', row.find('span', {'class': 'yellow'}).text)[0]
                time_on_pitch = str(time_on_pitch) + 'y'
            if row.find('span', {'class': 'yellow2'}):
                second_yellow_card_minute = re.findall(r'\d+', row.find('span', {'class': 'yellow2'}).text)[1]
                time_on_pitch = str(time_on_pitch) + 'r'

            # print([number, name, first_yellow_card_minute, second_yellow_card_minute, time_on_pitch])

            lst_time[lst.index(name)] = time_on_pitch
    # Return coach information and player time data
    return coaches, lst_time


def write_bets(date, bets):
    """Appends betting data to 'betsData.csv' for a specific date."""
    with open('betsData.csv', mode='a', encoding='utf-8', newline='') as csv_file:
        # write columns titles
        writer = csv.writer(csv_file)
        writer.writerow([date, bets[0], bets[1], bets[2]])


def parse_seasons(seasons_id: typing.List[int]):
    """Parses data for multiple football seasons and writes it to 'data.csv'."""
    with open('data.csv', mode='w', encoding='utf-8', newline='') as csv_file:
        df_bets = pd.read_csv("betsData.csv")
        # Create a CSV writer and write column titles
        writer = csv.writer(csv_file)
        writer.writerow(
            ['date', 'coach_team_home', 'team_home', 'bet_team_home', 'bet_draw', 'bet_team_away', 'team_away',
             'coach_team_away', 'match_stadium', 'match_hour', 'result', 'result_label'] +
            list(config.PLAYERS.values()))

        # Process data for each season
        for season_id in seasons_id:
            url = f"https://www.football.org.il/team-details/team-games/?team_id=5981&season_id={season_id}"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")

            for row_match in soup.find_all('div', {'class': 'table_row_group'}):
                # Parse each line of match details
                for row in soup.find_all('a', {'class': 'table_row'}):
                    link_id = row['href'].split('game_id=')[1]
                    date = row.find_all('div', {'class': 'table_col'})[0].text.split('תאריך')[1]
                    groups = row.find_all('div', {'class': 'table_col'})[1].text.split('משחק')[1].split(' - ')
                    group_home = groups[0]
                    group_away = groups[1]
                    match_location = row.find_all('div', {'class': 'table_col'})[2].text.split('אצטדיון')[1]
                    match_hour = row.find_all('div', {'class': 'table_col'})[3].text.split('שעה')[1]
                    result = row.find_all('div', {'class': 'table_col'})[4].text.split('תוצאה')[1]

                    print(f'parsing match...')
                    print(f'date:{date}')

                    if result == 'טרם נקבעה':
                        continue

                    result = result[::-1]

                    if result[0] == result[2]:
                        result_label = 0
                    elif group_home == config.TEAM_NAME:
                        if result[0] > result[2]:
                            result_label = 1
                        else:
                            result_label = -1

                    elif result[0] < result[2]:
                        result_label = 1
                    else:
                        result_label = -1

                    dfd = date.split('/')

                    if date in df_bets['date'].values:
                        bets = df_bets[df_bets['date'] == date]['bet_team_home'].values[0], \
                               df_bets[df_bets['date'] == date]['bet_draw'].values[0], \
                               df_bets[df_bets['date'] == date]['bet_team_away'].values[0]
                    else:
                        bets = get_bets(dfd[0], dfd[1], dfd[2])
                        write_bets(date, bets)

                    result = result.replace("-", "--")

                    # write
                    coaches, lst = parse_game(season_id=season_id,
                                              game_id=link_id,
                                              home_game=True if group_home == config.TEAM_NAME else False)
                    # write match final row
                    writer.writerow(
                        [date,
                         config.COACHES[coaches[0]],
                         config.TEAMS[group_home],
                         bets[0], bets[1], bets[2],
                         config.TEAMS[group_away],
                         config.COACHES[coaches[1]],
                         config.STADIUMS[match_location],
                         match_hour,
                         result,
                         result_label]
                        + lst)


def write_rpe():
    """Writes RPE data to 'RPE_data.csv'."""
    lst = list(config.PLAYERS.values())
    vals = list()
    for name in lst:
        vals.append(name + '_hardness_rating')
        vals.append(name + '_overall_team_rating')
        vals.append(name + '_individual_rating')

    date_convert = lambda val, a, b: datetime.strptime(val, a).strftime(b)

    with open('RPE_data.csv', mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['date'] + vals)

        # RPE data for the 21_22 season
        df = pd.read_csv("RPE_MATCH_21_22.csv")
        dict_res = dict()
        dates = [d.split(' ')[0] for d in df['Timestamp'].values]
        df_time = pd.to_datetime(dates)
        cnvr_dates = dict()
        for i in range(len(df_time) - 1):
            if int((df_time[i + 1] - df_time[i]).days) == 1:
                cnvr_dates[date_convert(dates[i], "%m/%d/%Y", "%m/%d/%Y")] = date_convert(dates[i + 1], "%m/%d/%Y",
                                                                                          "%m/%d/%Y")
        for date in dates:
            df1 = df.loc[df['Timestamp'].str.contains(str(date), case=False)].fillna(0)
            d = date_convert(date, "%m/%d/%Y", "%d/%m/%Y")

            cur_l = [-1] * len(vals)
            if d in dict_res:
                continue
            date = date_convert(date, "%m/%d/%Y", "%m/%d/%Y")
            if date in cnvr_dates.values():
                continue

            if date in cnvr_dates:
                df1 = pd.concat(
                    [df1, df.loc[df['Timestamp'].str.contains(str(cnvr_dates[date]), case=False)].fillna(0)], axis=0)
                df1 = pd.concat(
                    [df1, df.loc[df['Timestamp'].str.contains(str(cnvr_dates[date])[:5].replace('0', '') +
                                                              str(cnvr_dates[date])[5:], case=False)].fillna(0)],
                    axis=0)

            for index, row in df1.iterrows():
                name = row["שם מלא - Full Name"].split('.')[1].replace(' ', '_').lower()
                if name[0] == '_':
                    name = name[1:]

                cur_l[3 * lst.index(name)] = int(row['כמה קשה היה המשחק ?   How hard was the  match'])
                cur_l[3 * lst.index(name) + 1] = int(row["ציון קבוצתי למשחק  -  Overall Team's rating"])
                cur_l[3 * lst.index(name) + 2] = int(row["ציון אישי למשחק  -  Individual Rating"])
            dict_res[d] = cur_l.copy()
            writer.writerow([d] + cur_l.copy())

        # RPE data for the 22_23 season
        df = pd.read_csv("RPE_MATCH_22_23.csv")
        dict_res = dict()
        dates = [d.split(' ')[0] for d in df['Timestamp'].values]
        df_time = pd.to_datetime(dates)
        cnvr_dates = dict()
        for i in range(len(df_time) - 1):
            if int((df_time[i + 1] - df_time[i]).days) == 1:
                cnvr_dates[date_convert(dates[i], "%m/%d/%Y", "%m/%d/%Y")] = date_convert(dates[i + 1], "%m/%d/%Y",
                                                                                          "%m/%d/%Y")
        for date in dates:
            df1 = df.loc[df['Timestamp'].str.contains(str(date), case=False)].fillna(0)
            d = date_convert(date, "%m/%d/%Y", "%d/%m/%Y")

            cur_l = [-1] * len(vals)
            if d in dict_res:
                continue
            date = date_convert(date, "%m/%d/%Y", "%m/%d/%Y")

            if date in cnvr_dates.values():
                continue

            if date in cnvr_dates:
                df1 = pd.concat(
                    [df1, df.loc[df['Timestamp'].str.contains(str(cnvr_dates[date]), case=False)].fillna(0)], axis=0)
                df1 = pd.concat(
                    [df1, df.loc[df['Timestamp'].str.contains(str(cnvr_dates[date])[0].replace('0', '') +
                                                              str(cnvr_dates[date])[1:3] +
                                                              str(cnvr_dates[date])[3].replace('0', '') +
                                                              str(cnvr_dates[date])[4:], case=False)].fillna(0)],
                    axis=0)

            for index, row in df1.iterrows():
                name = row["שם מלא - Full Name"].split('.')[1].replace(' ', '_').lower()
                if name[0] == '_':
                    name = name[1:]

                cur_l[3 * lst.index(name)] = int(row[
                                                     "כמה קשה היה האימון  ?   How hard was the training / Quelle a été"
                                                     " la difficulté de l'entraînement"])
                cur_l[3 * lst.index(name) + 1] = int(row["Team's rating"])
                cur_l[3 * lst.index(name) + 2] = int(row["Individual rating"])
            dict_res[d] = cur_l.copy()
            writer.writerow([d] + cur_l.copy())


if __name__ == "__main__":
    # Call the parse_seasons function for specific season IDs
    parse_seasons(seasons_id=[23, 24])
    # Call the write_rpe function to write RPE data to 'RPE_data.csv'
    write_rpe()
