### DeepNet
Input: 14 planes of 8x8 (12 pieces + 2 positional encoding)

Body:

Residual tower(4x~64)

- 2 convolution
  
- Squeeze Excitation layer
  
- Skip connection
  
Output: 2x8x8, which outputs the starting & ending square

  
### SmallNet

Input: 14 planes of 8x8 (12 pieces + 2 positional encoding)

Body:

Residual tower (1x64)

  - 2 convolution
  
  - Squeeze Excitation Layer
  
  - Skip connection
  
Output: 2x8x8, which outputs the starting & ending square

  

### LC0
For reference, popular NN engine [LeelaZero](https://lczero.org/dev/backend/nn/) uses the following:

Input: 112 planes of 8x8

Body:

Residual tower (20 x 256)

  - 2 convolution
  
  - Squeeze Excitation Layer
  
  - Skip connection
  
Output: 1858 one-hot vector, which encapsulates the possible legal moves out of 4096

