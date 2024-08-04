import pygame as p
import sys
from cEngine import GameState, Move
from find_move import findBestMove, findRandomMove,get_model_evaluation, load_model 
from model import fen_to_matrix
from multiprocessing import Process, Queue
import time
import tensorflow as tf
import traceback
import numpy as np 



# Constants
MIN_WIDTH = MIN_HEIGHT = 400
MAX_WIDTH = MAX_HEIGHT = 1000
INITIAL_WIDTH = INITIAL_HEIGHT = 720
DIMENSION = 8
MAX_FPS = 60
IMAGES = {}   

# Colors
BOARD_COLORS = [p.Color("white"), p.Color("gray")]
HIGHLIGHT_COLOR = p.Color(255, 255, 0, 100)  # Yellow highlight
MOVE_DOT_COLOR = p.Color(0, 255, 0, 150)  # Green dot
CHECK_COLOR = p.Color(255, 0, 0, 100)  # Red for check
INFO_PANEL_COLOR = p.Color(40, 44, 52)  # Dark blue-gray
TEXT_COLOR = p.Color(171, 178, 191)  # Light gray
HEADER_COLOR = p.Color(97, 175, 239)  # Light blue
MOVE_LOG_BG = p.Color(33, 37, 43)  # Slightly lighter than sidebar
SCROLLBAR_COLOR = p.Color(78, 82, 93)  # Gray

# Sidebar constants
SIDEBAR_WIDTH = 250

def loadImages():
    pieces = ["wp", "wR", "wN", "wB", "wQ", "wK", "bp", "bR", "bN", "bB", "bQ", "bK"]
    for piece in pieces:
        IMAGES[piece] = p.image.load(f"images/{piece}.png")

def drawSplashScreen(screen):
    splash_image = p.image.load("vecteezy_chess-pieces-clipart-flat-design-black-chess-pieces-vector_20513754.jpg")
    splash_image = p.transform.scale(splash_image, (screen.get_width(), screen.get_height()))
    screen.blit(splash_image, (0, 0))
    p.display.flip()
    time.sleep(3)  # Display splash screen for 3 seconds

def drawGameState(screen, gs, validMoves, sqSelected, sq_size):
    drawBoard(screen, sq_size)
    highlightSquares(screen, gs, validMoves, sqSelected, sq_size)
    drawPieces(screen, gs.board, sq_size)

def drawBoard(screen, sq_size):
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            color = BOARD_COLORS[(row + col) % 2]
            p.draw.rect(screen, color, p.Rect(col * sq_size, row * sq_size, sq_size, sq_size))
    
    # Draw coordinate labels
    font = p.font.SysFont("Arial", int(sq_size / 5), True)
    for i in range(DIMENSION):
        # Draw file labels (a-h)
        file_label = font.render(chr(ord('a') + i), True, TEXT_COLOR)
        screen.blit(file_label, (i * sq_size + sq_size - sq_size / 4, screen.get_height() - sq_size / 4))
        # Draw rank labels (1-8)
        rank_label = font.render(str(DIMENSION - i), True, TEXT_COLOR)
        screen.blit(rank_label, (sq_size / 10, i * sq_size + sq_size / 10))

def drawPieces(screen, board, sq_size):
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            piece = board[row][col]
            if piece != "--":
                piece_image = p.transform.scale(IMAGES[piece], (sq_size, sq_size))
                screen.blit(piece_image, p.Rect(col * sq_size, row * sq_size, sq_size, sq_size))

def highlightSquares(screen, gs, validMoves, sqSelected, sq_size):
    if sqSelected != ():
        r, c = sqSelected
        if gs.board[r][c][0] == ('w' if gs.whiteToMove else 'b'):
            s = p.Surface((sq_size, sq_size))
            s.set_alpha(100)
            s.fill(HIGHLIGHT_COLOR)
            screen.blit(s, (c * sq_size, r * sq_size))
            for move in validMoves:
                if move.startRow == r and move.startCol == c:
                    p.draw.circle(screen, MOVE_DOT_COLOR, 
                                  (move.endCol * sq_size + sq_size // 2, move.endRow * sq_size + sq_size // 2), 
                                  sq_size // 6)

    if gs.inCheck:
        kingRow, kingCol = gs.whiteKingLocation if gs.whiteToMove else gs.blackKingLocation
        s = p.Surface((sq_size, sq_size))
        s.set_alpha(150)
        s.fill(CHECK_COLOR)
        screen.blit(s, (kingCol * sq_size, kingRow * sq_size))

def animateMove(move, screen, board, clock, sq_size):
    dR = move.endRow - move.startRow
    dC = move.endCol - move.startCol
    framesPerSquare = 10
    frameCount = (abs(dR) + abs(dC)) * framesPerSquare
    for frame in range(frameCount + 1):
        r, c = (move.startRow + dR * frame / frameCount, move.startCol + dC * frame / frameCount)
        drawBoard(screen, sq_size)
        drawPieces(screen, board, sq_size)
        color = BOARD_COLORS[(move.endRow + move.endCol) % 2]
        endSquare = p.Rect(move.endCol * sq_size, move.endRow * sq_size, sq_size, sq_size)
        p.draw.rect(screen, color, endSquare)
        if move.pieceCaptured != '--':
            if move.isenPassantMove:
                enPassantRow = move.endRow + 1 if move.pieceCaptured[0] == 'b' else move.endRow - 1
                endSquare = p.Rect(move.endCol * sq_size, enPassantRow * sq_size, sq_size, sq_size)
            screen.blit(p.transform.scale(IMAGES[move.pieceCaptured], (sq_size, sq_size)), endSquare)
        screen.blit(p.transform.scale(IMAGES[move.pieceMoved], (sq_size, sq_size)), p.Rect(c * sq_size, r * sq_size, sq_size, sq_size))
        p.display.flip()
        clock.tick(60)

def drawEndGameText(screen, text):
    font = p.font.SysFont("Helvetica", int(screen.get_width() / 20), True, False)
    textObject = font.render(text, 0, p.Color("Gray"))
    textLocation = p.Rect(0, 0, screen.get_width(), screen.get_height()).move(screen.get_width()/2 - textObject.get_width()/2, screen.get_height()/2 - textObject.get_height()/2)
    screen.blit(textObject, textLocation)
    textObject = font.render(text, 0, p.Color("Black"))
    screen.blit(textObject, textLocation.move(2, 2))

def drawGameInfo(screen, gs, moveLog, scroll_y, sidebar_width):
    sidebar = p.Rect(screen.get_width() - sidebar_width, 0, sidebar_width, screen.get_height())
    p.draw.rect(screen, INFO_PANEL_COLOR, sidebar)
    
    font = p.font.SysFont("Arial", int(sidebar_width / 10), True, False)
    small_font = p.font.SysFont("Arial", int(sidebar_width / 12), False, False)
    
    # Game Info
    info_texts = [
        ("Game Info", HEADER_COLOR, font),
        (f"{'White' if gs.whiteToMove else 'Black'} to move", TEXT_COLOR, small_font),
        (f"Check: {'Yes' if gs.inCheck else 'No'}", TEXT_COLOR, small_font),
        (f"Checkmate: {'Yes' if gs.checkmate else 'No'}", TEXT_COLOR, small_font),
        (f"Stalemate: {'Yes' if gs.stalemate else 'No'}", TEXT_COLOR, small_font),
    ]
    
    for i, (text, color, font_obj) in enumerate(info_texts):
        text_surface = font_obj.render(text, True, color)
        screen.blit(text_surface, (screen.get_width() - sidebar_width + 10, 10 + i * 30))
    
    # Move Log
    move_log_rect = p.Rect(screen.get_width() - sidebar_width + 10, 180, sidebar_width - 20, screen.get_height() - 200)
    p.draw.rect(screen, MOVE_LOG_BG, move_log_rect)
    
    move_log_surface = p.Surface((move_log_rect.width - 10, max(move_log_rect.height, len(moveLog) * 25)))
    move_log_surface.fill(MOVE_LOG_BG)
    
    for i, move in enumerate(moveLog):
        text = f"{i//2 + 1}. {move.getChessNotation()}" if i % 2 == 0 else f"    {move.getChessNotation()}"
        text_surface = small_font.render(text, True, TEXT_COLOR)
        move_log_surface.blit(text_surface, (5, i * 25))
    
    # Scrolling
    max_scroll = max(0, move_log_surface.get_height() - move_log_rect.height)
    scroll_y = min(max_scroll, max(0, scroll_y))
    
    screen.blit(move_log_surface, move_log_rect, (0, scroll_y, move_log_rect.width - 10, move_log_rect.height))
    
    # Scrollbar
    if max_scroll > 0:
        scrollbar_height = int((move_log_rect.height / move_log_surface.get_height()) * move_log_rect.height)
        scrollbar_pos = int((scroll_y / max_scroll) * (move_log_rect.height - scrollbar_height))
        p.draw.rect(screen, SCROLLBAR_COLOR, (screen.get_width() - 20, move_log_rect.top + scrollbar_pos, 10, scrollbar_height))

    return scroll_y


model = load_model()
def apply_cnn_bias(move, gs, model, threshold=0.08):
    """
    Get the CNN evaluation and print the probability.
    Returns the CNN score and a boolean indicating if the threshold was reached.
    """
    fen_after_move = gs.get_fen_after_move(move)
    cnn_score = get_model_evaluation(model, fen_after_move)
    
    threshold_reached = cnn_score >= threshold
    
    print(f"CNN evaluation: {cnn_score}")
    print(threshold_reached) 
    
    return cnn_score, threshold_reached




def main():
    p.init()
    screen = p.display.set_mode((INITIAL_WIDTH + SIDEBAR_WIDTH, INITIAL_HEIGHT), p.RESIZABLE)
    p.display.set_caption('CHESS')
    clock = p.time.Clock()

    
    
    # Load CNN model
    cnn_model = model
    
    # Display splash screen
    drawSplashScreen(screen)
    
    screen.fill(p.Color("white"))
    gs = GameState()
    validMoves = gs.getValidMoves()
    moveMade = False
    animate = False
    loadImages()
    moveSound = p.mixer.Sound('assets/move.mp3')
    p.display.set_icon(IMAGES['bK'])
    running = True
    sqSelected = ()
    playerClicks = []
    gameOver = False
    playerOne = True  # If a human is playing white, then this will be True. If an AI is playing, then False.
    playerTwo = False  # Same as above, but for black
    AIThinking = False
    moveFinderProcess = None
    moveUndone = False
    scroll_y = 0
    
    minimize_button = p.Rect(screen.get_width() - 60, 10, 20, 20)
    maximize_button = p.Rect(screen.get_width() - 30, 10, 20, 20)

    while running:
        current_w, current_h = screen.get_width(), screen.get_height()
        board_size = min(current_w - SIDEBAR_WIDTH, current_h)
        sq_size = board_size // DIMENSION
        
        humanTurn = (gs.whiteToMove and playerOne) or (not gs.whiteToMove and playerTwo)
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            elif e.type == p.VIDEORESIZE:
                if not p.display.get_surface().get_flags() & p.FULLSCREEN:
                    new_w = max(MIN_WIDTH, min(e.w, MAX_WIDTH))
                    new_h = max(MIN_HEIGHT, min(e.h, MAX_HEIGHT))
                    screen = p.display.set_mode((new_w, new_h), p.RESIZABLE)
                    board_size = min(new_w - SIDEBAR_WIDTH, new_h)
                    sq_size = board_size // DIMENSION
            elif e.type == p.MOUSEBUTTONDOWN:
                if e.button == 1:  # Left mouse button
                    location = p.mouse.get_pos()
                    if minimize_button.collidepoint(location):
                        p.display.iconify()
                    elif maximize_button.collidepoint(location):
                        if screen.get_flags() & p.FULLSCREEN:
                            screen = p.display.set_mode((INITIAL_WIDTH + SIDEBAR_WIDTH, INITIAL_HEIGHT), p.RESIZABLE)
                        else:
                            screen = p.display.set_mode((0, 0), p.FULLSCREEN)
                        board_size = min(screen.get_width() - SIDEBAR_WIDTH, screen.get_height())
                        sq_size = board_size // DIMENSION
                    elif not gameOver and humanTurn:
                        col = location[0] // sq_size
                        row = location[1] // sq_size
                        if col < 8: 
                            if sqSelected == (row, col):
                                sqSelected = ()
                                playerClicks = []
                            else:
                                sqSelected = (row, col)
                                playerClicks.append(sqSelected)
                            if len(playerClicks) == 2:
                                move = Move(playerClicks[0], playerClicks[1], gs.board)
                                for i in range(len(validMoves)):
                                    if move == validMoves[i]:
                                        gs.makeMove(validMoves[i])
                                        moveSound.play()
                                        moveMade = True
                                        animate = True
                                        sqSelected = ()
                                        playerClicks = []
                                if not moveMade:
                                    playerClicks = [sqSelected]
                elif e.button == 4:  # Scroll up
                    scroll_y = max(0, scroll_y - 20)
                elif e.button == 5:  # Scroll down
                    scroll_y = min(scroll_y + 20, len(gs.moveLog) * 25 - (current_h - 200))
            elif e.type == p.KEYDOWN:
                if e.key == p.K_z:
                    gs.undoMove()
                    moveMade = True
                    animate = False
                    gameOver = False
                    if AIThinking:
                        moveFinderProcess.terminate()
                        AIThinking = False
                    moveUndone = True
                if e.key == p.K_r:
                    gs = GameState()
                    validMoves = gs.getValidMoves()
                    sqSelected = ()
                    playerClicks = []
                    moveMade = False
                    animate = False
                    gameOver = False
                    if AIThinking:
                        moveFinderProcess.terminate()
                        AIThinking = False
                    moveUndone = True

        if not gameOver and not humanTurn and not moveUndone:
            if not AIThinking:

                AIThinking = True
                print("AI is thinking...")

                
                returnQueue = Queue()
                moveFinderProcess = Process(target=findBestMove, args=(gs, validMoves, returnQueue))
                moveFinderProcess.start()

            
            if not moveFinderProcess.is_alive():
                print('AI has calculated a move.') 
                potentialMove = returnQueue.get()
                cs, th = apply_cnn_bias(potentialMove, gs, model, threshold=0.08)
                print("CNN evaluation:", cs)
                
                if cs <= 0.08:
                    print("Confidence score too low. Reconsidering move...")
                    print("AI is thinking...")
                    
                    returnQueue = Queue()
                    moveFinderProcess = Process(target=findBestMove, args=(gs, validMoves, returnQueue))
                    moveFinderProcess.start()
                    
                    while moveFinderProcess.is_alive():
                        pass
                    
                    print('AI has calculated a new move.') 
                    potentialMove = returnQueue.get()
                    cs, th = apply_cnn_bias(potentialMove, gs, model, threshold=0.08)
                    print("CNN evaluation:", cs)
                
                AIMove = potentialMove
                
                if cs <= -4.0:
                    print("High chance of winning for white")
                elif -2.0 < cs <= 0.9:
                    print("Medium chance of winning for white")
                elif 0.9 < cs <= 4.0:
                    print("Even game or slight advantage for black")
                else:
                    print("High chance of winning for black")
                if AIMove is None:
                    AIMove = findRandomMove(validMoves)
                    
                    if not th:
                        print("Threshold not reached:", th)
                        print("Reconsidering another move")
                        AIMove = findRandomMove(validMoves) 
                
                print("AI has decided on its move.")
                gs.makeMove(AIMove)
                moveSound.play()
                moveMade = True
                animate = True
                AIThinking = False
                        


        if moveMade:
            if animate:
                animateMove(gs.moveLog[-1], screen, gs.board, clock, sq_size)
            validMoves = gs.getValidMoves()
            moveMade = False
            animate = False
            moveUndone = False

        drawGameState(screen, gs, validMoves, sqSelected, sq_size)
        scroll_y = drawGameInfo(screen, gs, gs.moveLog, scroll_y, SIDEBAR_WIDTH)
        
        if gs.checkmate:
            gameOver = True
            if gs.whiteToMove:
                drawEndGameText(screen, "checkmate")
            else:
                drawEndGameText(screen, "checkmate")
        elif gs.stalemate:
            gameOver = True
            drawEndGameText(screen, "Stalemate")

        # Draw minimize and maximize buttons
        p.draw.rect(screen, p.Color("lightgray"), minimize_button)
        p.draw.rect(screen, p.Color("lightgray"), maximize_button)
        p.draw.line(screen, p.Color("black"), (minimize_button.left + 2, minimize_button.centery), (minimize_button.right - 2, minimize_button.centery), 2)
        p.draw.rect(screen, p.Color("black"), maximize_button.inflate(-4, -4), 2)

        clock.tick(MAX_FPS)
        p.display.flip()

    p.quit()
    sys.exit()

if __name__ == "__main__":
    main()