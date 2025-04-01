from flask import Flask, request, render_template, jsonify
import chess
from stockfish import Stockfish
import torch
from model.ChessMoveCNN import ChessMoveCNN
import json

app = Flask(__name__)

# Initialize global board and Stockfish engine
board = chess.Board()
stockfish = Stockfish(path="./bin/stockfish-ubuntu-x86-64")

# some print debugging
import os

print("Checking Stockfish path:", os.path.exists("./bin/stockfish-ubuntu-x86-64"))
print("Is Stockfish executable:", os.access("./bin/stockfish-ubuntu-x86-64", os.X_OK))
stockfish.set_fen_position(board.fen())

# try:
#     # Try launching Stockfish manually
#     result = subprocess.run(
#         ["./bin/stockfish-ubuntu-x86-64"],
#         input="uci\nquit\n",
#         capture_output=True,
#         text=True,
#         timeout=5
#     )
#     print("Manual Stockfish run succeeded.")
#     print("STDOUT:\n", result.stdout)
#     print("STDERR:\n", result.stderr)
# except Exception as e:
#     print("Manual Stockfish run failed with exception:", e)

# # Optional: Check if Stockfish process is alive via the Python wrapper
# try:
#     print("Stockfish wrapper check:", stockfish.is_alive())
# except Exception as e:
#     print("Stockfish wrapper failed to start:", e)

model = ChessMoveCNN()
model.load_state_dict(torch.load("./model/endplay_weights.pt", map_location=torch.device('cpu')))
model.to('cpu')
model_size = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Model size: {model_size / 1024 ** 2:.2f} MB")

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
        top_sf_moves = stockfish.get_top_moves(5)
        for dicti in top_sf_moves:
            move = chess.Move.from_uci(dicti['Move'])
            eval = 500 if (dicti["Mate"] or (dicti["Centipawn"] > 500)) else (dicti["Centipawn"]) # cap the eval to +5
            cnn_score = int(model.get_move_score(board, move) * -100)
            total = eval + cnn_score # do the best combined SF + CNN score
            output.append((board.san(move), eval, cnn_score, total))

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
    pass
    # import os
    # port = int(os.environ.get("PORT", 5000))
    # app.run(debug=True, host="0.0.0.0", port=port)