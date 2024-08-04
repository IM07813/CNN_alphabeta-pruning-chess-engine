This document describes a Python program for a chess AI that utilizes a Convolutional Neural Network (CNN) for position evaluation combined with a traditional piece value, alpha-beta search, and position scoring approach.

# Project Structure:
preprocessed_data.zip has all the preprocessing scripts and datasets we used to train  the model 
gameState.py: This file defines a GameState class representing the chessboard state, including board representation, piece locations, move generation, and evaluation functions.
model.py: This file handles training and deploying the CNN model for chess position evaluation. It reads FEN strings representing chess positions, trains the model on a dataset with material-based evaluation labels, and saves the trained model for later use.
find_move.py: This file contains the core logic for finding the best move for the AI. It uses NegaMax pruning with alpha-beta search for efficient move exploration. The evaluation function combines the CNN prediction with a traditional piece value and position scoring approach.
chessmain.py: Chess logic and board representation which It manages with the chess game logic, keeps track of the board state, and validates legal moves.
Graphical User Interface (GUI): It creates the chessboard and piece graphics, handles user input (mouse clicks), and displays move animations and game information.
AI Integration : It allows the computer to play as black using a separate process to find the best move and make decisions.
cnn_scorer.py: FEN to board matrix: It converts a standard chess notation (FEN) into a numerical board representation for the CNN model.

Key Functionalities:

### Board Representation: 
The chessboard is represented as a 2D numpy array with piece codes (e.g., "wP" for white pawn, "bR" for black rook).
### Move Generation: 
The GameState class provides functions for generating valid moves for the current player considering piece types, movement rules, and checks.
### CNN Model: 
The model.py script trains a CNN model on FEN strings and corresponding material-based evaluation labels. The model predicts a score for a given chess position.
### Evaluation Function:
The scoreBoard function combines the CNN prediction with a traditional piece value and position scoring approach for a more comprehensive evaluation.
### Move Search:
The findBestMove function utilizes NegaMax pruning with alpha-beta search to explore possible moves and find the one leading to the best evaluation score for the AI player's color.


# Prerequisites:
Python 3 with necessary libraries (numpy, tensorflow, scikit learn and pygame etc.) installed.
A pre-trained chess_cnn_model.h5 file.


# Further Improvements:
    CNN, transformer or any other deep neural network is not sufficient enough to tackle and select write moves only by itself, It doesn't matter how huge of a dataset the model is trained on as the legal moves in chess surpasses any number we can process in sufficient amount of time and its not an approach one can use solely to win a chess against a Human opponent. 
    All the other algorithms help this model to acheive the performance you see and for further improvements, data would be much refined, with better architecture along with other algorithms combined. but its something uncharted and further integration may take alot of time, depending on how good you want this to become. 
    

