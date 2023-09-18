# CHAT B0T
A simple "manual" like Chat Bot using ``intents.json`` for training and tensorflow Sequential Model

### HOW TO RUN 

#### PIP PACKAGES
Run ``pip install -r requirements.txt`` to install all the required packages

#### Training Data
For the BOT to work well you need to add more training data in `intents.json` based on the field you want the bot to focus on.
#### CODE
1. Run `training.py` to generate the .pkl files for both **words** and **classes** with the **.h5** Tensorflow trained Sequential Model

2. Finally Run `chatbot.py` to run the actual Chatbot 

