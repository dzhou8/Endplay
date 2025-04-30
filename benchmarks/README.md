## Benchmarks

### Evaluation Method
I split the chess games data into a train set and a test set. The test set is shuffled 5% of the total cases, or 7k out of 140k test cases.

The inputs are chess positions with Black to play in a losing position. The output is the move that was chosen by the master-level player in that game.

For evaluation, I primarily used Top1 and Top5 accuracy. A prediction would be counted 'correct' if the top choice was the same as the output.
A prediction wouldbe counted 'correct' in top5, if any of the top 5 predictions was the same as the output.

### Benchmark models
I primarily decided on the following models to compare to:
- random: this would simply predict any legal move, at random
- stockfish: this would take stockfish's prediction, which is objectively the 'best' move in the position
- human: for this metric, I personally created a .pdf of the first 100 test cases, and tried my best to guess the move and checked my answers. I only tried for top1 accuracy

### Results
My goal when starting out was to create a model that is better than Stockfish at predicting human-like moves. 

My ultimate goal was to have my model perform better than any one human, but this may require more resources.

