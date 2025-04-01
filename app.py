from flask import Flask, request, render_template, jsonify
import chess
from stockfish import Stockfish

import torch
from model.ChessMoveCNN import ChessMoveCNN
import json
import time

app = Flask(__name__)

# Initialize global board and Stockfish engine
board = chess.Board()
stockfish = Stockfish(path="./bin/stockfish_20011801_x64")

model = ChessMoveCNN()
model.load_state_dict(torch.load("./model/endplay_weights.pt", map_location=torch.device('cpu')))
model.to('cpu')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/set_fen", methods=["POST"])
def set_fen():
    global board
    data = request.get_json()
    fen = data["fen"]

    try:
        board.set_fen(fen)
        stockfish.set_fen_position(fen)
        return jsonify({"fen": board.fen()})
    except Exception as e:
        return jsonify({"error": "Invalid FEN", "details": str(e)}), 400

@app.route("/move", methods=["POST"])
def move():
    global board
    data = request.get_json()
    user_move = data["move"]  # e.g. "e2e4"

    try:
        move_obj = chess.Move.from_uci(user_move)
        if move_obj in board.legal_moves:
            board.push(move_obj)
        else:
            return jsonify({"error": "illegal move"}), 400
    except:
        return jsonify({"error": "bad move format"}), 400

    output = []
    if not board.is_game_over(): # Endplay responds
        # Get Top 5 moves by stockfish
        stockfish.set_fen_position(board.fen())
        sf_start = time.time()
        top_sf_moves = stockfish.get_top_moves(5)
        sf_end = time.time()
        cnn_start = time.time()
        for dicti in top_sf_moves:
            move = chess.Move.from_uci(dicti['Move'])
            eval = 500 if (dicti["Mate"] or (dicti["Centipawn"] > 500)) else (dicti["Centipawn"]) # cap the eval to +5
            cnn_score = int(model.get_move_score(board, move) * -100)
            total = eval + cnn_score # do the best combined SF + CNN score
            output.append((board.san(move), eval, cnn_score, total))
        cnn_end = time.time()

        print(f"CNN Time: {cnn_end - cnn_start:.2f}s")
        print(f"SF Time: {sf_end - sf_start:.2f}s")

        print(board.fen())
        output.sort(key=lambda x:x[3])
        board.push_san(output[0][0])

    print(output)
    return jsonify({"fen": board.fen(), "output": output})

@app.route("/positions")
def positions():
    positions = []
    with open("./templates/positions.txt", "r") as f:
        for line in f:
            if not line:
                continue
            if "|" in line:
                label, fen = line.strip().split("|", 1)
                positions.append({"type": "position", "label": label, "fen": fen })
            else:
                positions.append({"type":"header", "title": line})
    return jsonify(positions)


if __name__ == "__main__":
    # import os
    # port = int(os.environ.get("PORT", 5000))
    # app.run(debug=True, host="0.0.0.0", port=port)
    pass