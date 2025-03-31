import os
import chess.pgn

def split_pgn(input_file, batch_size=10000):
    base_dir = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    with open(input_file, encoding="utf-8", errors="ignore") as pgn:
        batch_idx = 0
        games_in_batch = 0
        out_file = open(os.path.join(base_dir, f"{base_name}_batch_{batch_idx}.pgn"), "w", encoding="utf-8")

        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            # Export cleanly
            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
            pgn_string = game.accept(exporter)

            # Write to file
            out_file.write(pgn_string.strip() + "\n\n")  # extra newline between games

            games_in_batch += 1

            if games_in_batch >= batch_size:
                out_file.close()
                batch_idx += 1
                games_in_batch = 0
                out_file = open(os.path.join(base_dir, f"{base_name}_batch_{batch_idx}.pgn"), "w", encoding="utf-8")

        out_file.close()
        print(f"Finished splitting into {batch_idx + 1} batches.")

# Example usage:
split_pgn("./LumbrasGigaBase_2023.pgn", batch_size=100000)