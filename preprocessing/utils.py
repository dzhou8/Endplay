import chess
import numpy as np

def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((14, 8, 8), dtype=np.uint8)

    piece_to_index = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    for square, piece in board.piece_map().items():
        row = chess.square_rank(square) # Flip for correct orientation
        col = chess.square_file(square)
        idx = piece_to_index[(piece.piece_type, piece.color)]
        tensor[idx, row, col] = 1

    # explicit positional encoding
    for row in range(8):
        tensor[12, row, :] = row / 7.0
    for col in range(8):
        tensor[13, :, col] = col / 7.0

    return tensor

def move_to_tensor(move: chess.Move) -> np.ndarray:
    start_mask = np.zeros((8, 8), dtype=np.uint8)
    end_mask = np.zeros((8, 8), dtype=np.uint8)

    start_row = chess.square_rank(move.from_square)
    start_col = chess.square_file(move.from_square)

    end_row = chess.square_rank(move.to_square)
    end_col = chess.square_file(move.to_square)

    start_mask[start_row, start_col] = 1
    end_mask[end_row, end_col] = 1
    return np.stack([start_mask, end_mask])

def tensor_to_board(x):
    board = chess.Board.empty()
    index_to_piece = {
        0: (chess.PAWN, chess.WHITE),
        1: (chess.KNIGHT, chess.WHITE),
        2: (chess.BISHOP, chess.WHITE),
        3: (chess.ROOK, chess.WHITE),
        4: (chess.QUEEN, chess.WHITE),
        5: (chess.KING, chess.WHITE),
        6: (chess.PAWN, chess.BLACK),
        7: (chess.KNIGHT, chess.BLACK),
        8: (chess.BISHOP, chess.BLACK),
        9: (chess.ROOK, chess.BLACK),
        10: (chess.QUEEN, chess.BLACK),
        11: (chess.KING, chess.BLACK),
    }

    for i in range(12):
        for row in range(8):
            for col in range(8):
                if x[i, row, col] == 1:
                    rank = row
                    file = col
                    square = chess.square(file, rank)
                    piece_type, color = index_to_piece[i]
                    board.set_piece_at(square, chess.Piece(piece_type, color))

    board.turn = chess.BLACK
    return board

def tensor_to_move(y):
    start_mask, end_mask = y
    from_square = None
    to_square = None

    for row in range(8):
        for col in range(8):
            if start_mask[row, col]:
                from_square = chess.square(col, row)
            if end_mask[row, col]:
                to_square = chess.square(col, row)

    if from_square is not None and to_square is not None:
        return chess.Move(from_square, to_square)
    else:
        return None
    
def get_material_imbalance(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King's value is not considered in material balance
    }
    
    white_material = 0
    black_material = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                white_material += piece_values[piece.piece_type]
            else:
                black_material += piece_values[piece.piece_type]
                
    return white_material - black_material