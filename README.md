# Frame Level Speech Classification

This is my code for the Kaggle Competition for the CMU-11785 Introduction to Deep Learning. The competition details can be found in [Kaggle](https://www.kaggle.com/c/11-785-s20-hw1p2/overview). 

## Data

The data can be downloaded at [data](https://www.kaggle.com/c/11-785-s20-hw1p2/data)

The data comes from articles published in the Wall Street Journal (WSJ) that are read aloud and labelled using the original text. The speech data is the mel-spectrograms that have 40 band frequencies for each time step of the speech data, whose dimensions are [frames, time step, 40]. The labels are the index of the phoneme states [0-137].

There are 46 phonemes in the english language. ["+BREATH+", "+COUGH+", "+NOISE+", "+SMACK+", "+UH+", "+UM+", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "SIL", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]. However, a powerful technique in speech recognition is to model speech as a markov process with unobserved states. This model considers observed speech to be dependent on unobserved state transitions. We refer to these unobserved states as phoneme states or subphonemes. For each phoneme, there are 3 respective phoneme states. Therefore for our 46 phonemes, there exist **138** respective phoneme states.

## Model Architecture

The model uses 11-layers MLP, the first three layers have 2048 units followed by 4 linear layers with 1024 layers. Then 3 linear layers with 512 units are used before the final output linear. ReLu activation and BatchNorm are used before each linear layer.

Context k = 12 is applied which extracts 12 frames on both sides at the current time step. Therefore the input data is [Batch_size, 2*k + 1, 40]. After flatten before the linear, the input data is [Batch_size, (2*k + 1) * 40]