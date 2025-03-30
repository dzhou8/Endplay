from flask import Flask, request, render_template, jsonify
import chess
from stockfish import Stockfish

app = Flask(__name__)

# Initialize global board and Stockfish engine
board = chess.Board()
stockfish = Stockfish(path="./bin/stockfish-ubuntu-x86-64-avx2")

@app.route("/")
def index():
    return render_template("index.html")

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

    # If game not over, let Stockfish respond
    if not board.is_game_over():
        # Set position for Stockfish
        stockfish.set_fen_position(board.fen())
        best_move = stockfish.get_best_move()
        if best_move:
            board.push(chess.Move.from_uci(best_move))

    return jsonify({"fen": board.fen()})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)