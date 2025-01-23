import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import *
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3),activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(81*9))
model.add(Reshape((-1,9)))
model.add(Activation('softmax'))
model.load_weights('Change to Your Location')

def solve_sudoku_with_nn(model, puzzle):
    # Preprocess the input Sudoku puzzle
    puzzle = puzzle.replace('\n','').replace(' ', '')
    initial_board = np.array([int(j) for j in puzzle]).reshape((9,9,1))
    initial_board = (initial_board / 9)-0.5

    while True:
        #Use the neural network to predict values for empty cells
        predictions = model.predict(initial_board.reshape((1,9,9,1))).squeeze()
        pred = np.argmax(predictions, axis=1).reshape((9,9))+1
        prob = np.around(np.max(predictions, axis=1).reshape((9,9)),2)

        initial_board = ((initial_board+0.5)*9).reshape((9,9))
        mask = (initial_board == 0)

        if mask.sum() == 0:
            # Puzzle is solved
            break

        prob_new = prob *mask
        ind = np.argmax(prob_new)
        x,y = (ind//9),(ind%9)
        val = pred[x][y]
        initial_board[x][y] = val
        initial_board = (initial_board/9)-0.5
    
    # Convert the solved puzzle back to a string representation
    solved_puzzle = ''.join(map(str, initial_board.flatten().astype(int)))

    return solved_puzzle

def print_sudoku_grid(puzzle):
    puzzle = puzzle.replace('\n', '').replace(' ', '')
    result = []
    for i in range(9):
        if i % 3 == 0 and i != 0:
            result.append("-"*21)
        row = []
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row.append("|")
            row.append(puzzle[i*9 + j])
        result.append(" ".join(row))
    return "\n".join(result)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        puzzle = request.form['puzzle']
        solved_puzzle = solve_sudoku_with_nn(model, puzzle)
        formatted_solution = print_sudoku_grid(solved_puzzle)
        return jsonify({'solution': formatted_solution})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
