from bs4 import BeautifulSoup
import requests
import csv
import re
import config


def parse_season(season_id: int):
    url = f"https://www.football.org.il/team-details/team-games/?team_id=5981&season_id={season_id}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    with open('data.csv', mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['date', 'group_home', 'group_away', 'match_stadium', 'match_hour', 'result'] +
                        list(config.PLAYERS.values()))
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
                result = result.replace("-", "--")
                # write
                print(result)
                lst = parse_game(season_id=season_id, game_id=link_id,
                                 home_game=True if group_home == config.TEAM_NAME else False)
                writer.writerow([date, group_home, group_away, match_location, match_hour, result] + lst)


def parse_game(season_id: int, game_id: int, home_game: bool):
    url = f"https://www.football.org.il/leagues/games/game/?season_id={season_id}&game_id={game_id}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    lst = list(config.PLAYERS.keys())
    lst_time = [0] * len(lst)

    title = 'קבוצה - ביתית' if home_game else 'קבוצה אורחת'
    bench = 'home Bench' if home_game else 'guest Bench'

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
            first_yellow_card_minute = second_yellow_card_minute = ''  # Initialisation du nombre de cartons jaunes
            if row.find('span', {'class': 'yellow'}):  # Vérification de la présence de cartons jaunes
                first_yellow_card_minute = re.findall(r'\d+', row.find('span', {'class': 'yellow'}).text)[0]
            if row.find('span', {'class': 'yellow2'}):
                second_yellow_card_minute = re.findall(r'\d+', row.find('span', {'class': 'yellow2'}).text)[1]
            time_on_pitch = 90
            if row.find('span', {'class': 'change-up'}):
                time_on_pitch = 90 - int(re.findall(r'\d+', row.find('span', {'class': 'change-up'}).text)[0])
            elif row.find('span', {'class': 'change-down'}):
                time_on_pitch = re.findall(r'\d+', row.find('span', {'class': 'change-down'}).text)[0]

            # print([number, name, first_yellow_card_minute, second_yellow_card_minute, time_on_pitch])

            lst_time[lst.index(name)] = time_on_pitch
    return lst_time


if __name__ == "__main__":
    parse_season(season_id=23)
