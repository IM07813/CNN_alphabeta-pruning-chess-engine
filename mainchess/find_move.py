import random
import tensorflow as tf
import numpy as np
from cEngine import GameState
from model import fen_to_matrix
from cEngine import GameState



# Piece scores
pieceScore = {"K": 0, "Q": 9, "R": 5, "B": 3, "N": 3, "p": 1}

# Function to create score arrays
def create_score_array(base_pattern):
    return [base_pattern[min(i, len(base_pattern) - 1)] for i in range(8)]

# Create score arrays
knightScores = [
    create_score_array([0.0, 0.1, 0.2]),
    create_score_array([0.1, 0.3, 0.5]),
    create_score_array([0.2, 0.5, 0.6, 0.65]),
    create_score_array([0.2, 0.55, 0.65, 0.7]),
    create_score_array([0.2, 0.5, 0.65, 0.7])[::-1],
    create_score_array([0.2, 0.55, 0.6, 0.65])[::-1],
    create_score_array([0.1, 0.3, 0.5, 0.55])[::-1],
    create_score_array([0.0, 0.1, 0.2])
]

bishopScores = [
    create_score_array([0.0, 0.2]),
    create_score_array([0.2, 0.4]),
    create_score_array([0.2, 0.4, 0.5, 0.6]),
    create_score_array([0.2, 0.5, 0.6]),
    create_score_array([0.2, 0.4, 0.6]),
    create_score_array([0.2, 0.6]),
    create_score_array([0.2, 0.5, 0.4]),
    create_score_array([0.0, 0.2])
]

rookScores = [
    create_score_array([0.25]),
    create_score_array([0.5, 0.75]),
    *[create_score_array([0.0, 0.25])] * 5,
    create_score_array([0.25, 0.5, 0.25])
]

queenScores = [
    create_score_array([0.0, 0.2, 0.3]),
    create_score_array([0.2, 0.4]),
    create_score_array([0.2, 0.4, 0.5]),
    create_score_array([0.3, 0.4, 0.5]),
    create_score_array([0.4, 0.5, 0.4, 0.3])[::-1],
    create_score_array([0.2, 0.5, 0.4]),
    create_score_array([0.2, 0.4, 0.5, 0.4]),
    create_score_array([0.0, 0.2, 0.3])
]

pawnScores = [
    create_score_array([0.8]),
    create_score_array([0.7]),
    create_score_array([0.3, 0.4, 0.5, 0.4, 0.3]),
    create_score_array([0.25, 0.3, 0.45, 0.3, 0.25]),
    create_score_array([0.2, 0.4, 0.2]),
    create_score_array([0.25, 0.15, 0.1, 0.2, 0.1, 0.15, 0.25]),
    create_score_array([0.25, 0.3, 0.0, 0.3, 0.25]),
    create_score_array([0.2])
]

piecePositionScores = {"wN": knightScores,
                       "bN": knightScores[::-1],
                       "wB": bishopScores,
                       "bB": bishopScores[::-1],
                       "wQ": queenScores,
                       "bQ": queenScores[::-1],
                       "wR": rookScores,
                       "bR": rookScores[::-1],
                       "wp": pawnScores,
                       "bp": pawnScores[::-1]}

CHECKMATE = 1000
STALEMATE = 0
DEPTH = 4

cache = {}


     
def scoreBoard(gs):
    if gs.checkmate:
        return -CHECKMATE if gs.whiteToMove else CHECKMATE
    elif gs.stalemate:
        return STALEMATE
    
    traditional_score = calculateTraditionalScore(gs)

    return traditional_score

def load_model():
  """Loads the chess CNN model from an H5 file.

  Returns:
    The loaded TensorFlow Keras model, or None if an error occurs.
  """

  try:
    model = tf.keras.models.load_model("chess_cnn_model.h5")
    return model
  except Exception as e:
    print(f"Error loading model: {e}")
    return None


model = load_model()

# Create a chess board and get its FEN
chess_board = GameState()
fen = chess_board.getFen()



def get_model_evaluation(model, fen):
    board_matrix = fen_to_matrix(fen)
    board_matrix = np.expand_dims(board_matrix, axis=(0, -1)) 
    return model.predict(board_matrix, verbose=0)[0][0]

def calculateTraditionalScore(gs):
    score = 0
    for row in range(len(gs.board)):
        for col in range(len(gs.board[row])):
            piece = gs.board[row][col]
            if piece != "--":
                piecePositionScore = 0
                if piece[1] != "K":
                    piecePositionScore = piecePositionScores[piece][row][col]
                if piece[0] == "w":
                    score += pieceScore[piece[1]] + piecePositionScore
                if piece[0] == "b":
                    score -= pieceScore[piece[1]] + piecePositionScore
    
    opponentKingRow, opponentKingCol = gs.blackKingLocation if gs.whiteToMove else gs.whiteKingLocation
    
    # Check if game is in endgame
    whitePieces, blackPieces, endgame_phase = isEndgame(gs)

    if endgame_phase:
        # calculate distance to the closest edge
        min_dist = min(opponentKingRow, 7 - opponentKingRow, opponentKingCol, 7 - opponentKingCol)
        score += min_dist * 0.1

        if whitePieces <= 7:
            score -= 50
        if blackPieces <= 7:
            score += 50
    return score

def isEndgame(gs):
    #Checks if the game is in EndGame
    whitePieces = sum(1 for row in gs.board for piece in row if piece[0] == "w")
    blackPieces = sum(1 for row in gs.board for piece in row if piece[0] == "b")

    return whitePieces, blackPieces, (whitePieces <= 7 or blackPieces <= 7 or (whitePieces + blackPieces <= 14))

def findBestMove(gs, validMoves, returnQueue):
    global nextMove
    nextMove = None
    random.shuffle(validMoves)
    findMoveNegaMaxAlphaBeta(gs, validMoves, depth=DEPTH, alpha=-CHECKMATE, beta=CHECKMATE, turnMultiplier = 1 if gs.whiteToMove else -1)
    returnQueue.put(nextMove)


def findMoveNegaMaxAlphaBeta(gs, validMoves, depth, alpha, beta, turnMultiplier):
    global nextMove, cache

    if depth == 0:
        return turnMultiplier * scoreBoard(gs)

    position_key = hash((str(gs.board), gs.whiteToMove))
    if position_key in cache:
        return cache[position_key]

    maxScore = -CHECKMATE
    if depth == DEPTH:
        validMoves = sorted(validMoves, key=moveScore, reverse=True)

    for move in validMoves:
        gs.makeMove(move)
        score = -findMoveNegaMaxAlphaBeta(gs, gs.getValidMoves(), depth - 1, -beta, -alpha, -turnMultiplier)
        gs.undoMove()

        if score > maxScore:
            maxScore = score
            if depth == DEPTH:
                nextMove = move

        alpha = max(alpha, maxScore)
        if alpha >= beta:
            break

    cache[position_key] = maxScore
    return maxScore

def moveScore(move):
    if move.pieceCaptured != '--':
        return 10 + piece_values[move.pieceCaptured[1]] - piece_values[move.pieceMoved[1]] / 10
    elif move.isPawnPromotion:
        return 9
    elif move.isCastleMove:
        return 3
    else:
        return piece_values[move.pieceMoved[1]] / 100 + (move.endRow * 8 + move.endCol) / 1000

piece_values = {'p': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}

def findRandomMove(validMoves):
    return random.choice(validMoves)
