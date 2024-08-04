import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Check for GPU availability and configure TensorFlow to use it
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)
        
        
def process_and_train(data_file):
    with open(data_file, 'r') as f:
        data = f.read()
    
    games = data.split('=== Game Start ===')[1:]  # Split into individual games
    X, y = prepare_data(games)
    
    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    model = create_cnn_model()
    model = train_model(X_train, y_train, X_val, y_val, model)
    return model

def fen_to_matrix(fen):
    piece_dict = {
        'p': 1, 'P': -1, 'n': 2, 'N': -2, 'b': 3, 'B': -3,
        'r': 4, 'R': -4, 'q': 5, 'Q': -5, 'k': 6, 'K': -6
    }
    matrix = np.zeros((8, 8), dtype=np.int8)
    fen = fen.split()[0]  
    row, col = 0, 0
    for char in fen:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        elif char in piece_dict:
            matrix[row, col] = piece_dict[char]
            col += 1
    return matrix

def prepare_data(games):
    X, y = [], []
    for game in games:
        positions = game.split('\n')
        for pos in positions:
            parts = pos.split()
            if len(parts) >= 1:
                fen = parts[0]
                matrix = fen_to_matrix(fen)
                X.append(matrix)
                
                
                evaluation = np.sum(matrix)
                y.append(evaluation)
    return np.array(X), np.array(y)

def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 1), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_model(X_train, y_train, X_val, y_val, model, epochs=200, batch_size=64):
  X_train = X_train.reshape(-1, 8, 8, 1)
  X_val = X_val.reshape(-1, 8, 8, 1)
  
  train_loss, train_mae, val_loss, val_mae = [], [], [], []
  
  for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    history = model.fit(X_train, y_train,  
                        validation_data=(X_val, y_val),
                        epochs=1,  
                        batch_size=batch_size,  
                        verbose=0)
    
    train_loss.append(history.history['loss'][0])
    train_mae.append(history.history['mae'][0])
    val_loss.append(history.history['val_loss'][0])
    val_mae.append(history.history['val_mae'][0])
    
    print(f"Training Loss: {train_loss[-1]:.4f}")
    print(f"Training MAE: {train_mae[-1]:.4f}")
    print(f"Validation Loss: {val_loss[-1]:.4f}")
    print(f"Validation MAE: {val_mae[-1]:.4f}")
  
  # Plot losses and MAE
  epochs_range = np.arange(epochs) + 1
  plt.figure(figsize=(10, 6))
  
  plt.plot(epochs_range, train_loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)
  plt.show()
  
  plt.figure(figsize=(10, 6))
  
  plt.plot(epochs_range, train_mae, label='Training MAE')
  plt.plot(epochs_range, val_mae, label='Validation MAE')
  plt.title('Training and Validation MAE')
  plt.xlabel('Epoch')
  plt.ylabel('MAE')
  plt.legend()
  plt.grid(True)
  plt.show()
  
  return model

def get_model_evaluation(model, fen):
    matrix = fen_to_matrix(fen)
    matrix = matrix.reshape(1, 8, 8, 1)  # Reshape for model input
    return model.predict(matrix)[0][0]

if __name__ == "__main__":
    try:
        model = process_and_train("fen_file.txt")
        model.save("chess_cnn_model.h5")
        print("\nModel trained and saved as chess_cnn_model.h5")
        
        # Test the model with a sample FEN
        sample_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        evaluation = get_model_evaluation(model, sample_fen)
        print(f"Evaluation for starting position: {evaluation:.4f}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

