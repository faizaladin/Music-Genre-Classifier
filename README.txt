README TXT

The purpose of this program is to classify music using a deep learning algorithm. Using the GTZAN dataset, we use MATLAB's deep learning algorithm toolbox to analyze the Mel-spectrogram images in the dataset to predict what genre a song belongs in. Currently this program has two CNN layers as my computer CPU can only handle two layers. However with each additional layer, the algorithm should be able to provide a more accurate prediction.

Results:
The algorithm had a 75% accuracy in the training model, and 40% accuracy with testing.

What to improve on:
The next step for this project would be to pass a song into the dataset and see how it will analyze it. This can be done by taking the spectrogram of the song and feeding that image into the algorithm. It should then be able to predict which genre it belongs in.

Sources:
https://www.mathworks.com/help/deeplearning/ref/trainnetwork.html#d124e176568

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
