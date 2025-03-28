import streamlit as st
import streamlit.components.v1 as components

import chess
import chess.svg
import random
import base64
from stockfish import Stockfish

# Create an SVG rendering of the board
def render_svg(board):
    svg = chess.svg.board(board=board, size=400)
    return f"<div>{svg}</div>"

# Show SVG in Streamlit using components
def show_board(board):
    svg_board = render_svg(board)
    components.html(svg_board, height=500)

if "board" not in st.session_state:
    st.session_state.board = chess.Board()

stockfish = Stockfish(path='./stockfish-ubuntu-x86-64-avx2')

# Display board
st.write("### Play against Stockfish")
show_board(st.session_state.board)

# Move input
with st.form("move_form"):
    user_move = st.text_input("Your move (in algebraic format, e.g., e4, Nf3, O-O):", key="user_move")
    submitted = st.form_submit_button("Make Move")

if submitted:
    try:
        move = st.session_state.board.parse_san(user_move)
        print(move)
        if move in st.session_state.board.legal_moves:
            st.session_state.board.push(move)

            # Get Stockfish response
            stockfish.set_fen_position(st.session_state.board.fen())
            best_move = stockfish.get_best_move()
            st.session_state.board.push(chess.Move.from_uci(best_move))

            st.rerun()
        else:
            st.warning("Illegal move.")
    except Exception as e:
        st.warning("Invalid move format. {e}")

if st.button("Reset Game"):
    st.session_state.board = chess.Board()
