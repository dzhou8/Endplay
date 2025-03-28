import streamlit as st
import chess
import random
import base64
from stockfish import Stockfish

if "board" not in st.session_state:
    st.session_state.board = chess.Board()

stockfish = Stockfish(path='stockfish-ubuntu-x86-64-avx2"')

# Display board
st.write("### Play against Stockfish")
st.write(st.session_state.board.unicode())

# Move input
user_move = st.text_input("Your move (in UCI format, e.g., e2e4):")

if user_move:
    try:
        move = chess.Move.from_uci(user_move)
        if move in st.session_state.board.legal_moves:
            st.session_state.board.push(move)

            # Get Stockfish response
            stockfish.set_fen_position(st.session_state.board.fen())
            best_move = stockfish.get_best_move()
            st.session_state.board.push(chess.Move.from_uci(best_move))
        else:
            st.warning("Illegal move.")
    except:
        st.warning("Invalid move format.")

if st.button("Reset Game"):
    st.session_state.board = chess.Board()
