from flask import Flask, request, render_template, jsonify
import chess
from stockfish import Stockfish
import torch
from model.ChessMoveCNN import ChessMoveCNN
import json

app = Flask(__name__)

# Initialize global board and Stockfish engine
board = chess.Board()
stockfish = Stockfish(path="./bin/stockfish-ubuntu-x86-64-avx2")
model = ChessMoveCNN()
model.load_state_dict(torch.load('./model/early_stop.pt'))

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

    output = "game over"
    # If game not over, let Endplay respond
    if not board.is_game_over():
        print(board.fen())
        best_moves = model.get_top_moves(board, 5)
        output = [(board.san(move), float(score)) for move, score in best_moves]
        board.push(best_moves[0][0])

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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)