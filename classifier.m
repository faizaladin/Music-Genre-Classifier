%Faiz Aladin
%Final Project

%Path for data set
path = fullfile('Data/images_original/');

%Retreives data set and sets labels to the genre names
dataset = imageDatastore(path, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%Takes 50 of the files in each genre folder to be used as training for the
%deep learning algorithm. The 50 are taken at random. 
numTrainingFiles = 50;
[imdsTrain,imdsTest] = splitEachLabel(dataset,numTrainingFiles,'randomize');

%Deep Learning Setup
layers = [ ...
    %Image input layer (image size is 288x432)
    imageInputLayer([288 432 3])
    %First CNN Layer
    convolution2dLayer(3,16,'Padding',1)
    %Normalizes output from first layer
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2) 
    %Second CNN layer that receives input after data is normalized from the
    %first layer
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer 
    %Retreives information from the CNN layers and classifies the song
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-5, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;

%Returns accuracy of predictions (different from training accuracy)
accuracy = sum(YPred == YTest)/numel(YTest)


