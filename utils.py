def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((12, 8, 8), dtype=np.uint8)

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
        row = 7 - chess.square_rank(square) # Flip for correct orientation
        col = chess.square_file(square)
        idx = piece_to_index[(piece.piece_type, piece.color)]
        tensor[idx, row, col] = 1

    return tensor

def move_to_tensor(move: chess.Move) -> np.ndarray:
    start_mask = np.zeros((8, 8), dtype=np.uint8)
    end_mask = np.zeros((8, 8), dtype=np.uint8)

    start_row = 7 - chess.square_rank(move.from_square)
    start_col = chess.square_file(move.from_square)

    end_row = 7 - chess.square_rank(move.to_square)
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
                    rank = 7 - row
                    file = col
                    square = chess.square(file, rank)
                    piece_type, color = index_to_piece[i]
                    board.set_piece_at(square, chess.Piece(piece_type, color))

    return board

def tensor_to_move(y):
    start_mask, end_mask = y
    from_square = None
    to_square = None

    for row in range(8):
        for col in range(8):
            if start_mask[row, col]:
                from_square = chess.square(col, 7 - row)
            if end_mask[row, col]:
                to_square = chess.square(col, 7 - row)

    if from_square is not None and to_square is not None:
        return chess.Move(from_square, to_square)
    else:
        return None