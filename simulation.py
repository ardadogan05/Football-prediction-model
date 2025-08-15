import numpy as np


N = 10000 #simulations

def calculateSimulation(stats):
    #determine lambda for team 1 and 2
    #try to get probability to match opta's, 
    team1LeagueCoef = stats["team 1 league coef"]
    team2LeagueCoef = stats["team 2 league coef"]

    team1League = stats["team 1 league"]
    team2League = stats["team 2 league"]


    homeAdvantage = stats["homeAdvantage"] # boost for home team, based on: https://www.premierleague.com/en/news/4032415, 1.1(very low) - 1.25 (average is 1.3)
    awayDisadvantage = stats["awayDisadvantage"]

    #Different lambda calculations in case H2h data isn't availabe. Important that xG and xGa are equal to avoid punishing defensive teams.
    if stats["xG h2h team 1"] != 0 or stats["xG h2h team 2"] != 0:
        lambda1 = (0.1875*stats["xG form 1"] + 
                   0.2626*stats["xG szn avg 1"] + 
                   0.1875*stats["xGa form 2"] + 
                   0.2626*stats["xGa szn avg 2"] + 
                   0.10*stats["xG h2h team 1"])
          
        lambda2 = (0.1875*stats["xG form 2"] + 
                   0.2625*stats["xG szn avg 2"] + 
                   0.1875*stats["xGa form 1"] + 
                   0.2625*stats["xGa szn avg 1"] + 
                   0.10*stats["xG h2h team 2"]) 
        
    else:
        lambda1 = (0.20*stats["xG form 1"] + 
                   0.30*stats["xG szn avg 1"] + 
                   0.30*stats["xGa form 2"] + 
                   0.20*stats["xGa szn avg 2"])
        
        lambda2 = (0.20*stats["xG form 2"] + 
                   0.30*stats["xG szn avg 2"] +
                   0.30*stats["xGa form 1"] + 
                   0.20*stats["xGa szn avg 1"])

    lambda1 *= homeAdvantage 
    lambda2 *= awayDisadvantage 

#Inital planned to print out xG, but tuning the model to match the betmakers causes the xG to not be representative. 
    if (team1LeagueCoef != team2LeagueCoef): #if could be removed, but doesn't matter
        lambda1 *=  team1LeagueCoef
        lambda2 *=  team2LeagueCoef

    dlambda = lambda1-lambda2
    beta_close_gap = 1.8
    beta_medium_gap = 1.7
    beta_large_gap = 0.75
    max_boost = 0.8


    if dlambda >= 0:
        if dlambda <= 0.15:
            boost = min(dlambda * beta_close_gap, max_boost)
            lambda1 *= (1+boost/2)
            lambda2 /= (1+boost/2)
        elif dlambda <= 0.45:
            boost = min(dlambda*beta_medium_gap, max_boost)
            lambda1 *= (1+boost/2)
            lambda2 /= (1+boost/2)
        else:
            boost = min(dlambda*beta_large_gap, max_boost)
            lambda1 *= (1+boost/2)
            lambda2 /= (1+boost/2)

    elif dlambda < 0:
        if abs(dlambda) <= 0.15:
            boost = min(abs(dlambda)*beta_close_gap, max_boost)
            lambda1 /= (1+boost/2)
            lambda2 *= (1+boost/2)

        elif abs(dlambda) <= 0.45:
            boost = min(abs(dlambda)*beta_medium_gap, max_boost)
            lambda1 /= (1+boost/2)
            lambda2 *= (1+boost/2)

        else:
            boost = min(abs(dlambda)*beta_large_gap, max_boost)
            lambda1 /= (1+boost/2)
            lambda2 *= (1+boost/2)



    print(lambda1, lambda2)
    
    stats["xG 1"] = lambda1
    stats["xG 2"] = lambda2



    t1_win = 0
    t2_win = 0
    draw = 0

    sim1 = np.random.poisson(lambda1,N)
    sim2 = np.random.poisson(lambda2,N)

    for i in range(N):
        if sim1[i]>sim2[i]:
            t1_win += 1
        elif sim2[i]>sim1[i]:
            t2_win += 1
        else: 
            draw += 1

    print(f'H: {t1_win/N}, D: {draw/N}, A: {t2_win/N}')
    team1WinRate = t1_win/N
    team2WinRate = t2_win/N
    drawRate = draw/N
    return team1WinRate, drawRate, team2WinRate, team1League, team2League
        



