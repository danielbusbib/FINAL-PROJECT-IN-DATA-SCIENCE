# imports
from bs4 import BeautifulSoup
import requests
import csv
import re
import typing
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# globals
import config


def get_bets(day: int, month: int, year: int):
    """bets from the site telesport.il"""
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.get(
        f'https://www.telesport.co.il/%D7%90%D7%96%D7%95%D7%A8%20%D7%95%D7%95%D7%99%D7%A0%D7%A8#110299/198/'
        f'{year}-{month}-{day}T18:30:00')
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    all_text = soup.get_text()
    text_lines = all_text.split('\n')
    # parse lines
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
            return bet_home, bet_draw, bet_away


def parse_game(season_id: int, game_id: int, home_game: bool):
    """return time on pitch for each player in game_id on season_id"""
    url = f"https://www.football.org.il/leagues/games/game/?season_id={season_id}&game_id={game_id}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    coaches = list()
    for coach_line in soup.find_all('div', {'aria-labelledby': 'GAME_COACH_HOME'}):
        coaches.append(coach_line.find('b', {'class': 'name'}).text)
    if len(coaches) == 0:
        coaches.append('')

    for coach_line in soup.find_all('div', {'aria-labelledby': 'GAME_COACH_GUEST'}):
        if coach_line.find('b', {'class': 'name'}).text:
            coaches.append(coach_line.find('b', {'class': 'name'}).text)
    if len(coaches) == 1:
        coaches.append('')

    lst = list(config.PLAYERS.keys())
    lst_time = [0] * len(lst)

    title = 'קבוצה - ביתית' if home_game else 'קבוצה אורחת'
    bench = 'home Bench' if home_game else 'guest Bench'

    # get max time recorded
    max_time = 0
    for row in soup.find_all('div', {'class': 'node number'}):
        max_time = int(row.find('span', {'class': ''}).text)

    # save not played players
    not_played = set()
    for row in soup.find_all('div', {'class': bench}):
        for row1 in row.find_all('a', {'title': title}):
            name = row1.find('span', {'class': 'name'}).text
            if "-" in name:
                name = name[:name.find(' -')]
            not_played.add(name)

    # save played
    for row_match in soup.find_all('a', {'title': title}):
        for row in row_match.find_all('div', {'class': 'player'}):
            if str(row.find('span', {'class': 'number'}).text).isalpha():  # coach ou arbitres
                break
            number = re.findall(r'\d+', row.find('span', {'class': 'number'}).text)[0]
            name = row.find('span', {'class': 'name'}).text  # Récupération du nom du joueur
            if "-" in name:  # cas ou capitaine ou poste
                name = name[:name.find(' -')]
            if name in not_played:
                continue

            time_on_pitch = max_time
            if row.find('span', {'class': 'change-up'}):
                time_on_pitch = max_time - int(re.findall(r'\d+', row.find('span', {'class': 'change-up'}).text)[0])
            elif row.find('span', {'class': 'change-down'}):
                time_on_pitch = re.findall(r'\d+', row.find('span', {'class': 'change-down'}).text)[0]

            first_yellow_card_minute = second_yellow_card_minute = ''  # Initialisation du nombre de cartons jaunes
            if row.find('span', {'class': 'yellow'}):  # Vérification de la présence de cartons jaunes
                first_yellow_card_minute = re.findall(r'\d+', row.find('span', {'class': 'yellow'}).text)[0]
                time_on_pitch = str(time_on_pitch) + 'y'
            if row.find('span', {'class': 'yellow2'}):
                second_yellow_card_minute = re.findall(r'\d+', row.find('span', {'class': 'yellow2'}).text)[1]
                time_on_pitch = str(time_on_pitch) + 'r'

            # print([number, name, first_yellow_card_minute, second_yellow_card_minute, time_on_pitch])

            lst_time[lst.index(name)] = time_on_pitch
    return coaches, lst_time


def parse_seasons(seasons_id: typing.List[int]):
    with open('data.csv', mode='w', encoding='utf-8', newline='') as csv_file:
        # write columns titles
        writer = csv.writer(csv_file)
        writer.writerow(
            ['date', 'coach_team_home', 'team_home', 'bet_team_home', 'bet_draw', 'bet_team_away', 'team_away',
             'coach_team_away', 'match_stadium', 'match_hour', 'result'] +
            list(config.PLAYERS.values()))

        # write data for each season
        for season_id in seasons_id:
            url = f"https://www.football.org.il/team-details/team-games/?team_id=5981&season_id={season_id}"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")

            for row_match in soup.find_all('div', {'class': 'table_row_group'}):
                # parse each line of match details
                for row in soup.find_all('a', {'class': 'table_row'}):
                    link_id = row['href'].split('game_id=')[1]
                    date = row.find_all('div', {'class': 'table_col'})[0].text.split('תאריך')[1]
                    groups = row.find_all('div', {'class': 'table_col'})[1].text.split('משחק')[1].split(' - ')
                    group_home = groups[0]
                    group_away = groups[1]
                    match_location = row.find_all('div', {'class': 'table_col'})[2].text.split('אצטדיון')[1]
                    match_hour = row.find_all('div', {'class': 'table_col'})[3].text.split('שעה')[1]
                    result = row.find_all('div', {'class': 'table_col'})[4].text.split('תוצאה')[1]
                    if result == 'טרם נקבעה':
                        continue
                    dfd = date.split('/')
                    print(dfd)
                    bets = get_bets(dfd[0], dfd[1], dfd[2])
                    result = result.replace("-", "--")
                    # write
                    coaches, lst = parse_game(season_id=season_id, game_id=link_id,
                                              home_game=True if group_home == config.TEAM_NAME else False)
                    writer.writerow(
                        [date, coaches[0], group_home, bets[0], bets[1], bets[2], group_away, coaches[1],
                         match_location, match_hour, result] + lst)


if __name__ == "__main__":
    parse_seasons(seasons_id=[23, 24])
