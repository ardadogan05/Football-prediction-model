import requests #be able to send request to http (get data from website)
import pandas as pd #handle files
from io import StringIO #read html as file-like object
import time #to avoid getting blocked from fbref from scraping too often
import difflib #to match string like "west ham utd" to "west ham"
from FBrefIDFetch import fetchID
import numpy as np

#Team 1 is home, 2 away

def dataFetch(userInput1, userInput2, stats):
    query = " stats site:fbref.com"
    team1Query = userInput1 + query
    team2Query = userInput2 + query



    team1ID, team1 = fetchID(team1Query)
    time.sleep(1.25)
    team2ID, team2 = fetchID(team2Query)
    print(team1,team2)


    # the names are used to find xG on H2H page, therefore it must have the format "manchster city", instead of "manchester-city", which is what the fetchID returns.

    def teamNameFromID(team): #allows input like man city and man utd to get turned into "manchester city" and "manchester united". Is derived from the URL from search API
        if len(team.split('-')) == 3:
            return (team.split('-')[0] + " " + team.split('-')[1] + " " + team.split('-')[2]).lower()
        
        elif len(team.split('-')) == 2:
            return (team.split('-')[0] + " " + team.split('-')[1]).lower()
        
        elif len(team.split('-')) == 4:
            return (team.split('-')[0] + " " + team.split('-')[1] + " " + team.split('-')[2] + " " + team.split('-')[3]).lower()

        else:
            return team.lower()

    team1_name = teamNameFromID(team1)
    team2_name = teamNameFromID(team2)

    stats["team1 name"] = team1_name
    stats["team2 name"] = team2_name


    season = "2024-2025"
    form_sample_size = 6


    #handling files
    def build_url(teamID, team, season):
        return "https://fbref.com/en/squads/" + teamID + "/"+ season + "/" + team + "-Stats"
    def build_urlh2h(team1ID, team2ID, team1, team2, season):
        return "https://fbref.com/en/stathead/matchup/teams/"+ team1ID +"/"+ team2ID+ "/"+ team1 +"-vs-"+ team2+ "-History"

    team1_url = build_url(team1ID, team1, season)
    team2_url = build_url(team2ID, team2, season)
    h2h_url = build_urlh2h(team1ID, team2ID, team1, team2, season)

    #Timeout if website doesn't load in 10s, sleep to not get ip banned for scraping
    html_content1  = requests.get(team1_url, timeout=10)
    time.sleep(1.25)
    html_content2  = requests.get(team2_url,timeout=10)
    time.sleep(1.25)

    try:
        h2h_content =  requests.get(h2h_url, timeout=10)
        h2h_content.raise_for_status() #raises error if unable to scrape

    except requests.exceptions.RequestException: #handles all forms for errors
        print("H2H data is unavailabe or there was an error loading the data")
        h2h_content = None


    responses = [html_content1, html_content2, h2h_content]
    for response in responses:
        if response.status_code == 200:
            pass
        else:
            print(f'There was an error retrieving data for {response.url}: {response.status_code}')


    df1 = pd.read_html(StringIO(html_content1.text))
    df2 = pd.read_html(StringIO(html_content2.text))
    try:
        df_h2h = pd.read_html(StringIO(h2h_content.text))
    except:
        print("There is no H2H data available, the prediction will be less accurate")
        h2h_url = None


    if h2h_url is not None:
        h2h_xg_teamH_column = df_h2h[0].iloc[:, 6] # all rows, 6th column, means all xG values for home team
        h2h_xg_teamA_column = df_h2h[0].iloc[:, 8]
        h2h_H_column = df_h2h[0]["Home"]
        h2h_A_column = df_h2h[0]["Away"]

        

        #calculate h2h xG: NaN frequently present, need to skip those to find average of last 3
        total_xG_team1 = 0
        total_xG_team2 = 0
        count = 0
        i = 0
        #Doesn't feel clean, but is functional. Fix if deemed neccessary
        lastGameDate = df_h2h[0]["Date"][0]

        if int(lastGameDate.split('-')[0]) > 2022: #Not doing H2H xG stats if game is played in 2022 or ealier. Avoids old games and lack of xG, and scraping doesn't work
            def match_teams(originalNames, scrapedNames): #Names used for teams in the H2H columns are inconsistent, this part takes both names, and matches the closest ones
                matched = {}
                for orig in originalNames:
                    best_match = difflib.get_close_matches(orig, scrapedNames, n = 1, cutoff= 0.3)
                    if best_match:
                        matched[orig] = best_match[0]
                    else:
                        matched[orig] = None
                return matched

            scrapedNames = [h2h_H_column[0], h2h_A_column[0]]
            originalNames = [team1_name, team2_name]

            matchDict = match_teams(originalNames, scrapedNames)

            while (count < 3) and (i < len(h2h_H_column)):
                if not pd.isna(h2h_xg_teamH_column[i]): #pd.isna() checks if it is a number ( avoid NaN), if you have xG for home team, you have away as well
                    if h2h_H_column[i].lower() == (matchDict[team1_name]).lower():
                        total_xG_team1 += float(h2h_xg_teamH_column[i])
                        total_xG_team2 += float(h2h_xg_teamA_column[i])
                        count += 1
                        i +=1
                    elif h2h_H_column[i].lower() == (matchDict[team2_name]).lower():
                        total_xG_team1 += float(h2h_xg_teamA_column[i])
                        total_xG_team2 += float(h2h_xg_teamH_column[i])
                        count += 1
                        i +=1
                else:
                    i+=1
                    
            team1_h2h_xg = total_xG_team1/count
            team2_h2h_xg = total_xG_team2/count
        else:
            team1_h2h_xg = 0
            team2_h2h_xg = 0
    else: 
        team1_h2h_xg = 0
        team2_h2h_xg = 0


    def findLeague(dataFrame):
        comps = dataFrame[1]["Comp"]
        top5l = ["Premier League", "Serie A", "Bundesliga", "Ligue 1", "La Liga"]
        for i in range(len(comps)):
            if comps[i] in top5l:
                return comps[i]
        print("Could not retrieve league")
        raise ValueError("Invalid input: must be a top 5 league team as defined by Opta")

    team1League = findLeague(df1)
    team2League = findLeague(df2)

    xg_column1 = df1[1]["xG"] #column with xG home team
    xga_column1 = df1[1]["xGA"] #column with xGa h team

    xg_column2 = df2[1]["xG"] #column with xG away
    xga_column2 = df2[1]["xGA"] #away

    #xG and xGa for both home and away have been extracted as columns
    xg_seasonavg1 = xg_column1.mean() #mean skips NaN values
    xga_seasonavg1 = xga_column1.mean()

    xg_seasonavg2 = xg_column2.mean() #mean skips NaN values
    xga_seasonavg2 = xga_column2.mean()

    xg_form1 = xg_column1.tail(form_sample_size).mean() #tail() will include NaN values, shaky form calculator. rare to have 1+ games without xG in last 6, 
    xga_form1 = xga_column1.tail(form_sample_size).mean()

    xg_form2 = xg_column2.tail(form_sample_size).mean()
    xga_form2 = xga_column2.tail(form_sample_size).mean()

    leagueCoefficient = {"Premier League" : 1, "Serie A": 0.9395, "La Liga" : 0.9395, "Bundesliga" : 0.9320, "Ligue 1" : 0.9233}
    team1LeagueCoef = leagueCoefficient[team1League]
    team2LeagueCoef = leagueCoefficient[team2League]

    #For dynamic home and away boost: Idea is to look at over/underperformance of xG home and away, and give boost/decrease in xG accordingly
    #Home team
    team1Venue = df1[1]["Venue"]
    home_count = 0
    team1_home_xG = 0
    xG_home_list = []
    for i, venue in enumerate(team1Venue):
        if venue == "Home" and not pd.isna(xg_column1[i]):
            team1_home_xG += xg_column1[i]
            xG_home_list.append(xg_column1[i])
            home_count += 1
    homeAdvantage = (team1_home_xG/home_count)/xg_seasonavg1

    #Away team
    team2Venue = df2[1]["Venue"]
    away_count = 0
    team2_away_xG = 0
    for i, venue in enumerate(team2Venue):
        if venue == "Away" and not pd.isna(xg_column2[i]):
            team2_away_xG += xg_column2[i]
            away_count += 1
        else:
            continue
    awayDisadvantage = (team2_away_xG/away_count)/xg_seasonavg2

    #limit the dynamic home and away advantages, avoid making home team favorites every time.
    #also dampen the effect so it isn't 1.1 and 0.9 every time.
    dampen = 0.95
    advantage_factor = (homeAdvantage - 1) * dampen
    disadvantage_factor = (awayDisadvantage - 1) * dampen

    homeAdvantage = 1 + advantage_factor
    awayDisadvantage = 1 + disadvantage_factor


    homeAdvantage = np.clip(1, homeAdvantage, 1.10) #Arsenal actually underperform their xG at home. Counterintuitive that they get worse at home, therefore no nerf for home team.
    awayDisadvantage = np.clip(0.90, awayDisadvantage, 1.00)

    
    stats["homeAdvantage"] = homeAdvantage
    stats["awayDisadvantage"] = awayDisadvantage

    stats["team 1 league"] = team1League
    stats["team 2 league"] = team2League

    stats["h2h url"] = h2h_url

    stats["xG szn avg 1"] = xg_seasonavg1
    stats["xGa szn avg 1"] = xga_seasonavg1

    stats["xG szn avg 2"] = xg_seasonavg2
    stats["xGa szn avg 2"] = xga_seasonavg2

    stats["xG form 1"] = xg_form1
    stats["xGa form 1"] = xga_form1

    stats["xG form 2"] = xg_form2
    stats["xGa form 2"] = xga_form2

    stats["xG h2h team 1"] = team1_h2h_xg
    stats["xG h2h team 2"] = team2_h2h_xg

    stats["team 1 league coef"] = leagueCoefficient[team1League]
    stats["team 2 league coef"] = leagueCoefficient[team2League]

    print("xG per game home:", (team1_home_xG/home_count))
    print("xG per game home list: ", np.nanmean(xG_home_list))
    print("xG per game szn avg: ", xg_column1.mean())
    
    
    return stats


















