# Football Match Prediction Model

*README based on a summary of my audio notes while building this project. Provided by OpenAI's ChatGPT 5*

## Overview

This project predicts the outcome of football matches between teams in the top 5 European leagues (as defined by Opta Aug. 2025) using **expected goals (xG)** data from FBRef. It scrapes data through the Brave Search API, runs thousands of Poisson-based simulations, and outputs win/draw/loss probabilities.

The goal was to get predictions close to bookmaker odds, since they have extremely advanced models, while still building my own lambda calculation. I can’t fully calibrate it to actual results, so the lambda values are based on my own reasoning, not perfect, but fine for this project. The idea to ground the model in the betting odds came after realizing that I couldn't calibrate it.

For a more serious production build, I would store scraped data in a file (preferably JSON, but CSV is fine) so the program doesn’t waste resources scraping every time it runs. Right now, everything is freshly scraped on each run.

---

## Key Features

The model includes dynamic home and away advantage factors based on real xG over and underperformance, fuzzy matching to handle typos and inconsistent team names, exception handling for missing or outdated data, league strength coefficients for cross-league matchups based on Opta’s rankings, and a modern GUI with logos and proper data credits.

By using Brave Search API, the code allows a variety of different names for teams. The API searches on FBref using the given input and allows them to find the full/proper name for the club, thus making it easier for me.

---

## How It Works

The model uses a **Poisson distribution**, where the mean equals the variance, to simulate goals scored. This works well for most games, but underestimates extreme scorelines.

Lambda values, representing the average xG per team per match, are based on a weighted mix of season averages and other factors. This approach is inspired by Lee & Smith (2002), who showed that performance tends to **regress to the mean**, making season averages more predictive than short-term form.

---

## Limitations

The model currently uses data from 2024-2025, as the 2025-2026 season is yet to begin (as of August 15th, 2025). Therefore, the model will not be able to predict any games until at least 6 games have been played, and will get more accurate as more games are played.

The model uses subjective lambda values with no calibration to actual results, does not account for public-perception bias in bookmaker odds, uses xG data from all competitions (which can inflate stats for Europa League or Conference League teams compared to Champions League sides), does not generate predictions for newly promoted teams until they have played at least six matches in all competitions, and has some messy variable names such as `team1Name` and `team2-name`. The model does not take player injuries into account. This means that if an important player is to get hurt, e.g., Rodri for Manchester City last year, the drop in team performance will be visible after it is shown in the decrease in xG and/or increase in xGa. The inverse also applies; new signings late in the transfer window, or in January, will not directly affect the model.

Fbref does ban IP addresses that send too many requests, therefore be wary: 
- Max 10 requests/min to FBref & Stathead
- Max 20 requests/min to other sites
- Violations result in a 1-day IP block

---

## Models Considered (Not Implemented)

I considered a Negative Binomial model, which handles rare high scores better, but decided the complexity outweighed the benefit for this xG-based approach. I also looked at a Bivariate Poisson model, which accounts for correlation between teams’ scoring, but it requires more data and is harder to implement. Both could be explored in future versions to improve scoreline realism.

---

## Development Notes

The project began by scraping FBRef, but I ran into dynamic URL issues with team IDs. I tried using Gemini AI for ID lookup and typo handling, but replaced it with the Brave Search API to avoid hallucinations. I fixed inconsistent FBRef team names using fuzzy matching, refactored the code to avoid circular imports for the GUI, added a boost to the lambda values of favorites after noticing underdogs and draws were being overvalued, and realized that the biggest challenge wasn’t the statistical modeling but making the scraping reliable, handling edge cases, and keeping the program user-friendly.

---

## Reference

Lee, M., & Smith, G. (2002). Regression to the mean and football wagers. *Journal of Behavioral Decision Making*, 15, 329–342. [https://doi.org/10.1002/BDM.418](https://doi.org/10.1002/BDM.418)

---

## Known glitches

I have tried this code on 3 different machines, and have encountered a glitch where the probability bar ends up in a random place in the GUI instead of under the output box. There seems to be an issue when using customTkinter and Tkinter on some devices. The easiest solution is to remove the probability bar, as it is purely visual.

**Note (17.09.2025):** The website is currently unavailable (403 error), meaning that stats cannot be scraped and used. This seems to be related to Cloudflare restrictions, and I’ll be working on a proper fix once I have more time. 


---

## Try the code yourself!
As I have been advised against sharing my API key for Brave Search API with you, you will need to create one yourself. It is free of charge and can be done at [https://brave.com/search/api/](https://brave.com/search/api/). You will need to do the following:

* Clone the repository
* Install the required libraries by writing "pip install -r requirements.txt" in the terminal.
* Create your own .env file with your API key (or directly write it in FBrefIDFetch.py).
* Run the gui.py file.
Please send me a message if you have ideas to improve the code or encounter errors.
IKKE BETTE STORSTIPENDET MED TIPS FRA MODELLEN
