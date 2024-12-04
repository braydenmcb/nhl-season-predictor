# NHL SEASON STATS PREDICTOR

## This program will:
- Scrape the data of all players of a given season via [Hockey-Reference.com](https://www.hockey-reference.com/leagues/NHL_2024_skaters.html)
- it will then use a linear/polynomial model to predict the stats and then output a .csv file to show the ranked stats of each player
> [!NOTE]
> It will only scrape players that played in the specified season <br>
> Ex. Connor Bedard will be in the training set if the season is at least the 2023-24 season

## How to use:
### As of right now:
use `python scraper.py --year "year"` to scrape the data from that season, it will be based on the latter year in the season
  - Ex. if you want the 2023-24 season, use `--year 2024`
  - The default year is the previous season (right now the 2022-23 season)
you can also use `python scraper.py --year "year" --single` to scrape a single season
  - This is mainly used to gather the results / expected data
  - leaving out the `--single` will scrape each players career stats by season

## Road Map
- [ ] Make the scraper track up to the specified season at compilation and output to a csv file
- [ ] Make the prediciton model and output to csv
- [ ] make a way to ensure if the season is already completed \(in order to be able to use loss funcs. to determine accuracy)
