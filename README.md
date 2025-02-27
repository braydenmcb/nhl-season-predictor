# NHL SEASON STATS PREDICTOR

## This program will:
- Scrape the data of all players of a given season via [Hockey-Reference.com](https://www.hockey-reference.com/leagues/NHL_2024_skaters.html)
- it will then use a linear/polynomial model to predict the stats and then output a .csv file to show the ranked stats of each player
> [!NOTE]
> It will only scrape players that played in the specified season <br>
> Ex. Connor Bedard will be in the training set if the season is at least the 2023-24 season

# How to use:
## Setting up the program(s):
Create a virtual environment via the commands:
  - for Mac:
  ```
  python -m venv env
  source env/bin/activate
  ```
  - for Windows:
  ```
  python venv env
  .\venv\Scripts\activate
  ```
Once the environment is made, install the needed libraries via pip
  - Mac: `pip install requests beautifulsoup4 pandas numpy`
  - Windows: `pip3 install requests beautifulsoup4 pandas numpy`
    
## Running the program(s):
### For the scraper and predictor
use `python predictor.py --year "year"` to predict the next season after the given year, it will be based on the latter year in the season
  - Ex. if you want the 2023-24 season, use `--year 2024`
  - The default year is the previous completed season (right now the 2022-23 season)
  - The program will call `scraper.py` if the seasons are not in the data folder

### For just the scraper
use `python scraper.py --year "year"` to scrape the data from that season. <br/>
you can also use `python scraper.py --year "year" --single` to scrape a single season.
  - This is mainly used to gather the results / expected data
  - leaving out the `--single` will scrape each players' career stats (given they played in the given year).

## Road Map
- [ ] Make the scraper track up to the specified season at compilation and output to a csv file
- [ ] Make the prediciton model and output to csv
- [ ] make a way to ensure if the season is already completed \(in order to be able to use loss funcs. to determine accuracy)
