import requests
from bs4 import BeautifulSoup
import pandas as pd
import argparse
from datetime import datetime
"""
This function will scrape the player stats from the player's page on hockey-reference.com
it'll run thru the top 100 players from the last season and get their stats from all of their previous seasons (not including playoffs)
then it'll save the data to a csv file for the learning model to parse later
"""


def scrape_player(url):
    # Fetch the webpage
    response = requests.get(url)

    # Parse the page using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    player_name = soup.find('div', {'class' : 'players'}).find('h1').text
    print("Scraping player:", player_name)
    table = soup.find('table', {'class': 'stats_table'})  # The table ID is 'stats_table' for regular season player stats
    if table is None:
        print(f"Could not find stats table for player: {player_name}")
        return None
    # Extract data rows
    rows = table.find('tbody').find_all('tr')
    player_data = []
    for row in rows:
        if row.find('th', {"scope": "row"}) is not None:
            season_link = row.find('th')
            season_year = season_link.text
            season_stats = [td.text for td in row.find_all('td')]
            season_statline = [player_name] + [season_year] + season_stats
            player_data.append(season_statline)
    print(f"Finished scraping player: {player_name}\n")
    return player_data

def main(year):
    main_url = f'https://www.hockey-reference.com/leagues/NHL_{year}_skaters.html'
    print(f"Scraping player stats from {main_url}\n")
    response = requests.get(main_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    training_data = []
    table = soup.find('table', {'class': 'stats_table'})
    headers = ['Player Name'] + [th.text for th in table.find('thead').find_all('th')][10:]  # Skipping the over headers

    players = table.find('tbody').find_all('tr')
    prev_player_page = ''
    for player in players:
        if player.find('th', {"scope": "row"}) is not None:
            player_link = player.find('td').find('a')
            current_player_page = 'https://www.hockey-reference.com' + player_link['href']
            if current_player_page != prev_player_page:
                # print(f"Scraping player page: {current_player_page} | previous player page: {prev_player_page}\n")
                player_data = scrape_player(current_player_page)
                training_data.append(player_data)
                prev_player_page = current_player_page
            else:
                print("Skipping duplicate player page")

    df = pd.DataFrame(training_data, columns=headers)
    print(df.head())


    # df.to_csv('player_stats.csv', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selects the year of the players\' latest season to compare')
    parser.add_argument('--year', type=int, default=None, help='Year of the players\' latest season')
    args = parser.parse_args()
    if args.year is None:
        main(datetime.now().year - 1) # Default to the previous season
    main(args.year)
