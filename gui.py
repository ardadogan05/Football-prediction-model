import customtkinter
from data_fetch import dataFetch
from simulation import calculateSimulation
from PIL import Image
import tkinter as tk


customtkinter.set_appearance_mode("system default")
customtkinter.set_default_color_theme("blue")


app = customtkinter.CTk()
app.title("Football match prediction")
app.geometry("1200x720")

widthFrame = 1200
heightFrame = 720

app.resizable(False,False)


#Font
smallText = customtkinter.CTkFont(family="Times New Roman", size=12, )
boldFont = customtkinter.CTkFont(family="Montserrat", size=25, weight="bold")

#Watermark

watermark = customtkinter.CTkLabel(app, text = "Created by Arda Dogan, MSc Student in Cybernetics and Robotics", font=smallText)
watermark.place(x = 30, y = 20)


#Team 1 (home)

label1 = customtkinter.CTkLabel(app, text="Home team:", font = boldFont, text_color= "#4a6fa5")
label1.place(x = 50, y = 175)

entry1 = customtkinter.CTkEntry(app, width=250)
entry1.place(x = 50, y = 220)
# team 2 

label2 = customtkinter.CTkLabel(app, text="Away team:", font = boldFont, text_color="#b55a4e")
label2.place(x = widthFrame - 250 - 50, y = 175)

entry2 = customtkinter.CTkEntry(app, width = 250)
entry2.place(x = widthFrame - 250 - 50, y  = 220)

#function for switch button for home and away
def switch_teams():
    team1 = entry1.get()
    team2 = entry2.get()

    entry1.delete(0,"end")
    entry1.insert(0,team2)

    entry2.delete(0,"end")
    entry2.insert(0,team1)

#Textbox to print out results

output = customtkinter.CTkTextbox(app, width = 400, height = 200)
output.place(x = 400, y = 300)

#Images
fbrefImage = customtkinter.CTkImage(light_image = Image.open("Pictures/fbref.png"), size = (150, 150))
fbrefImageLabel = customtkinter.CTkLabel(app, image=fbrefImage, text="")
fbrefImageLabel.place(x = 25, y = 550)

optaImage = customtkinter.CTkImage( dark_image= Image.open("Pictures/OptaDark.jpg"), size = (300, 90))
optaImageLabel = customtkinter.CTkLabel(app, image=optaImage, text="")
optaImageLabel.place(x = widthFrame - 300, y = 632)

infoData = customtkinter.CTkLabel(app, text= "Match statistics obtained from FBref, aggregator of official Opta statistics.")
infoData.place(x = 400, y = 680)

#Probability bar
barSize = 400
canvas = tk.Canvas(app, width= 400, height= 15, highlightthickness=0, highlightcolor= "blue", bg= "#1f538d")
canvas.place(x = 400, y = 500)


#simulation function: 
def runSimulation():
    canvas.delete("all")
    output.delete("0.0", "end")
    userInput1 = entry1.get()
    userInput2 = entry2.get()
    if not userInput1 or not userInput2:
        raise ValueError("Please enter a valid teamname in both text boxes.")

    statsDict = {}
    dataFetch(userInput1, userInput2, statsDict)
    team1WinProb, drawProb, team2WinProb, team1League, team2League =  calculateSimulation(statsDict)

    #Competition photo
    if team1League == team2League:
        compPhoto = customtkinter.CTkImage(light_image = Image.open("Pictures/" + statsDict["team 1 league"] + ".png"), size = (150, 150))
    else:
        compPhoto = customtkinter.CTkImage(light_image = Image.open("Pictures/UEFA_Logo.png"), size = (150, 150))
    compPhotoLabel = customtkinter.CTkLabel(app, image=compPhoto, text="")
    compPhotoLabel.place(x = 525, y = 50)


    canvas.create_rectangle(0, 0, 400*team1WinProb, 15, fill='#4a6fa5', outline='dark gray')
    canvas.create_rectangle(400*team1WinProb, 0, 400*team1WinProb + 400*drawProb, 15, fill='gray', outline='dark gray')
    canvas.create_rectangle(400 * team1WinProb + 400 * drawProb, 0, 400 * team1WinProb + 400 * drawProb + 400*team2WinProb, 15, fill='#b55a4e', outline='dark gray')

    output.insert("15.0", f"The predicted result for the game between {statsDict['team1 name']} (H) and {statsDict['team2 name']} (A) is:\n\n"
        f"{statsDict['team1 name']} win: {team1WinProb:.1%}\n"
        f"Draw: {drawProb:.1%}\n"
        f"{statsDict['team2 name']} win: {team2WinProb:.1%}\n\n")
    



#"Simulate button"

button = customtkinter.CTkButton(app, text = "Calculate probabilities", command = runSimulation)
button.place(x = 530, y = 600)

swap_icon = customtkinter.CTkImage(Image.open("Pictures/swap_icon.png"), size = (30,30))
switch_button = customtkinter.CTkButton(app, image = swap_icon, text = "", command=switch_teams, width= 40, height = 50)
switch_button.place(x = 580, y = 220)

app.mainloop()

