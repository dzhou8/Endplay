import sys
sys.path.insert(1, '../../preprocessing/')
import utils
import torch
import chess
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib.pyplot as plt

def load_dataset(paths, test_size=0.1, random_state=42):
    all_X = []
    all_Y = []

    for path in paths:
        data = np.load(path)
        all_X.append(data["inputs"])
        all_Y.append(data["outputs"])

    # Combine all data
    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return (X_train, X_test, Y_train, Y_test)


# given a tensor of solutions/answers, and a prediction function, outputs the top1 and top5 accuracy
def evaluate(X_test, Y_test, predict_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top1_correct = 0
    top5_correct = 0

    num_samples = 50
    for i in tqdm(range(num_samples)):
        x = torch.tensor(X_test[i], dtype=torch.float32).to(device)
        board = utils.tensor_to_board(x)

        y_true = Y_test[i]
        start_idx = np.argmax(y_true[0].flatten())
        end_idx = np.argmax(y_true[1].flatten())
        start_square = chess.square_name(start_idx)
        end_square = chess.square_name(end_idx)

        top5_moves = predict_fn(board)
        top1_move = top5_moves[0]
        # print(chess.square_name(top1_move.from_square), chess.square_name(top1_move.to_square))
        # print(start_square, end_square)
        if (chess.square_name(top1_move.from_square) == start_square) and (chess.square_name(top1_move.to_square) == end_square):
            top1_correct += 1
        
        for move in top5_moves:
            if (chess.square_name(move.from_square) == start_square) and (chess.square_name(move.to_square) == end_square):
                top5_correct += 1
                break

    top1_accuracy = top1_correct / num_samples
    top5_accuracy = top5_correct / num_samples

    print(f"Top-1 Accuracy: {top1_accuracy:.3f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.3f}")

    # draw a bar chart
    plt.bar(["Top1", "Top5"], [top1_accuracy, top5_accuracy])
    plt.ylim(0, 1)
    plt.title("Move Prediction Accuracy")
    plt.show()

    return top1_accuracy, top5_accuracy