import requests
from bs4 import BeautifulSoup
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
import time

''' This can be done at a later time, just use time.sleep() for now '''
# proxies = open("proxies.txt", "r").read().strip().split("\n")
#
# def get_proxy():
#     return {'http': 'http://' + proxies.pop()}

######################################################################
################# SINGLE SEASON PLAYER STATS SCRAPER #################
######################################################################

def scrape_season(year):
    """
This function will scrape season stats for the requested season
This will mainly be used to check for accuracy in the model
"""

    main_url = f'https://www.hockey-reference.com/leagues/NHL_{year}_skaters.html'
    print(f"Scraping season stats from {main_url}\n")
    response = requests.get(main_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'class': 'stats_table'})

    headers = [th.text for th in table.find('thead').find_all('th')][10:]
    headers = headers[:-1] # remove the 'Awards' column
    players = table.find('tbody').find_all('tr')
    player_data = []

    #test_players = players[:3]  # For testing purposes

    for player in players:
        if player.find('th', {"scope": "row"}) is not None:
            season_stats = [td.text for td in player.find_all('td')]
            season_stats = season_stats[:-1] # remove the 'Awards' column 
            player_data.append(season_stats)
    df = pd.DataFrame(player_data, columns=headers[1:])
    df.to_csv(f'data/season_stats_{year}.csv', index=False)
    print(df.head())


#####################################################################
################# CAREER PLAYER STATS SCRAPER #######################
#####################################################################

def scrape_player(url, year, training_data):
    """
This function will scrape the player stats from the player's page on hockey-reference.com
it'll run thru the top 100 players from the last season and get their stats from all of their previous seasons (not including playoffs)
then it'll save the data to a csv file for the learning model to parse later
"""
    # Fetch the webpage
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
        return 0
    
    # Parse the page using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    player_name = soup.find('div', {'class' : 'players'}, ).find('h1').text
    print("Scraping player:", player_name)
    table = soup.find('table', {'id': 'player_stats'})  # The table ID is 'stats_table' for regular season player stats
    if table is None:
        print(f"Could not find stats table for player: {player_name}")
        return
    # Extract data rows
    rows = table.find('tbody').find_all('tr')
    player_data = []
    for row in rows:
        # if the season is greater than the year argument, skip it
        if row.find('th', {"data-stat": "year_id"}).text[:4] >= str(year):
            continue
        # else, scrape the data
        if row.find('th', {"scope": "row"}) is not None:
            season_link = row.find('th')
            season_year = season_link.text

            season_stats = [td.text for td in row.find_all('td')]
            season_stats = season_stats[:-1] #remove the "Awards" column

            if season_stats[4] != "":
                # print("#" * 50)
                # print(f"Scraped season: {season_year}")
                # print(season_stats)
                # print("#" * 50)
                season_statline = [player_name.strip()] + [season_year] + season_stats
                player_data.append(season_statline)
            else:
                print(f"Could not find season stats for player: {player_name}")
    print(f"Finished scraping player: {player_name}\n")
    training_data.extend(player_data)

def get_headers(url):
    """
    This function will scrape the headers from the player's page on hockey-reference.com
    so that the data can be properly matched and formatted
"""
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
        return 0

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'player_stats'})
    headers = [th.text for th in table.find('thead').find_all('th')][10:]
    headers = headers[:-1] # remove the 'Awards' column
    return headers


def main(year):
    """
The main function will scrape the table from the season requested
and then scrape the player pages for the players that played in that season via scrape_player()
"""
    #TODO
    '''
    1. Get the proxy to work to prevent rate limiting
    2. add a check to ensure that the seasons after the argumented year are not scraped
    '''

    
    main_url = f'https://www.hockey-reference.com/leagues/NHL_{year}_skaters.html'
    print(f"Scraping player stats from {main_url}\n")
    response = requests.get(main_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    training_data = []
    table = soup.find('table', {'class': 'stats_table'})
    
    headers = ['Player Name'] + get_headers("https://www.hockey-reference.com/players/b/brownco02.html")

    players = table.find('tbody').find_all('tr')
    prev_player_page = ''

    start_time = time.time()

    for player in players:
    # test_players = players[:3]  # For testing purposes
    # for player in test_players:
        if player.find('th', {"scope": "row"}) is not None:

            try:
                player_link = player.find('td').find('a')
                current_player_page = 'https://www.hockey-reference.com' + player_link['href']
            except TypeError:
                print("Could not find player link")
                continue

            if current_player_page != prev_player_page:
                # print(f"Scraping player page: {current_player_page} | previous player page: {prev_player_page}\n")
                player_data = scrape_player(current_player_page, year, training_data)
                if player_data is not None or player_data != 0:
                    training_data.append(player_data)
                    prev_player_page = current_player_page
                    time.sleep(2.5) # Sleep for 2.5 seconds to prevent rate limiting (30 requests / minute) (TODO: Implement proxies)
                else:
                    print(f"Error scraping player page: {current_player_page}")
                    time.sleep(1) # Sleep for 1 second to prevent rate limiting
            else:
                print("Skipping duplicate player page")

    end_time = time.time()
    
    print("#" * 50)
    print(f"Scraped {len(training_data)} players in {end_time - start_time} seconds")
    print(f"Length of training data: {len(training_data)}")
    print("#" * 50)

    filtered_data = [data for data in training_data if data is not None] #filter out the None values, idk where theyre coming from
    df = pd.DataFrame(filtered_data, columns=headers)
    print(df.head())
    df.to_csv(f'data/player_stats_{year}.csv', index=False)

#####################################################################
####################### ARGUMENT PARSER #############################
#####################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selects the year of the players\' latest season to compare')
    parser.add_argument('--year', type=int, default=None, help='Year of the players\' latest season')
    parser.add_argument("--single", action="store_true", help="Scrape a single season")
    args = parser.parse_args()
    if args.year is None:
        main(datetime.now().year - 1) # Default to the previous season
    if args.single:
        print("Scraping single season")
        scrape_season(args.year)
    else:
        main(args.year)
