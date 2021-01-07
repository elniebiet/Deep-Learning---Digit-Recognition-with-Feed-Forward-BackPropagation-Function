clc;
clear all;

%%this file will execute the entire application and utilise the 
%%backpropagation function

mainDirectory = "handwritten_number_rgb";

%import imgs and labls for training and validation, 
%the function divides the data into 8/10 for training and 2/10 for validation.
[imgs_Training, lbls_Training, imgs_Validation, lbls_Validation, divisionPoint] = importTVData(mainDirectory);

hidden_Neurons_ = 70; %% number of hidden neurons, frm 100
eta_ = 0.005; %learnng rte
MSETolerance_ = 0.31; %%set lowest MSE tolerance
alpha_ = 0; %momentum set inside the function, 1/num_Samples works just fine
outpt_Neurns_ = 8; %set numb of o/p neurons 1-10 for digits 0-9 , frm 10
maxNumEpochs_ = 5000; %set maximum number of epochs

[valInWeights, valInBias, valOutWeights, valOutBias, achievedMSE_, numEpochsDone_] = backPropagationFunction(imgs_Training, lbls_Training, ...
                                                                        imgs_Validation, lbls_Validation, mainDirectory, ...
                                                                        hidden_Neurons_, outpt_Neurns_, ...
                                                                        eta_, MSETolerance_, alpha_, ...
                                                                        maxNumEpochs_);

%%DIGIT RECOG VALIDATION STARTS
%uncomment these lines to test a single image, insert image to
%workspace and type name in imread function to import the image to matrix
% rawImage = imread('img_17.png');
% levelRawImage = graythresh(rawImage);
% imgs_Validation = im2bw(rawImage, levelRawImage);

%uncomment these lines to validate with same training dataset
%     imgs_Validation = imgs_Training;
%     lbls_Validation = lbls_Training;

%create array to hold predicted digits
predictedDigits = zeros(size(lbls_Validation, 1), 1);

%specify num of imgs to check
numValImages=size(imgs_Validation,3);

%specify num of neurons in out layr
validn_output_Neurons=10;

validationAccuracy=0;

    %Testingforward propagation 
    for i=1:numValImages
        %FORWARD PROPAGATN
        %init of inpt and outpt layrs
        %current input to input layer
        %convert current image to single column matrix of 784px
        for rows=1:size(imgs_Validation,1)
            for cols=1:size(imgs_Validation,2)
                valInInputNeurons(cols+(rows-1)*size(imgs_Validation,2),1) = imgs_Validation(rows,cols,i);
            end
        end
        %set outpt neurn for the particulr input to 1, others remain zero
        valOutputNeuron = zeros(validn_output_Neurons,1);
        valOutputNeuron(lbls_Validation(i)+1,1)=1;

        %compute i/p to Hidden Layr neurons
        zHiddenValid(:,1)= valInWeights * valInInputNeurons + valInBias(:,1); %compute zVal for hidden
        inHiddenNeuronsValid(:,1) = activn_Function(zHiddenValid(:,1)); %compute inpt to hiddn neurons

        %Compute i/p to Outpt layr neurons
        ztestoutput= valOutWeights * inHiddenNeuronsValid(:,1) + valOutBias; %compute zVal for outpt
        valInOutputNeurons=activn_Function(ztestoutput); %compute inpt to outpt neurons

        indxVal = find(valInOutputNeurons == max(valInOutputNeurons)); %get output neuron with max val
        if indxVal == (lbls_Validation(i) + 1) %check if it matches the labels
            validationAccuracy = validationAccuracy + 1; %increment the accuracy
        end

        %hold the predictd digit
        predictedDigits(i) = indxVal-1;
        %print predictd digit
        fprintf('Predicted Digit: %d, Actual digit: %d\n', predictedDigits(i), lbls_Validation(i));

    end

    fprintf('VALIDATION COMPLETE!\n');
    fprintf('Overall Validation Accuracy: %.2f percent (Final Validation Done with last 60values of dataset i.e last 2/10 values.)\n', (validationAccuracy / numValImages)*100);

    %Plot confusion matrix and receiver operating characteristics

    %plotconfusion and plotroc sees outputs as 1 and 0s, we have to convert the
    %desired outputs from single digit arrays 1 and 0s
    lbls_V = zeros(10, size(lbls_Validation, 1));
    for i = 1:size(lbls_Validation, 1)
        digt = lbls_Validation(i);
        lbls_V(digt+1, i) = 1;
    end
    predicted_Digits = zeros(10, size(predictedDigits, 1));
    for i = 1:size(predictedDigits, 1)
        digt = predictedDigits(i);
        predicted_Digits(digt+1, i) = 1;
    end
    plotconfusion(lbls_V, predicted_Digits); %plot confusion matrix
    plotroc(lbls_V, predicted_Digits); %plot receiver operating xtics
%%DIGIT RECOG VALIDATION ENDS


%%%%%BACKPROP ALGORITHM FUNCTION BEGINS HERE
function [Wih, Bh, Whj, Bo, achievedMSE, numEpochsDone] = backPropagationFunction(inputA, desiredOutD, ...
                                                        validA, validD, dataSetLink, ...
                                                        hidden_Neurons, outpt_Neurns, ...
                                                        eta, MSETolerance, alpha, ...
                                                        maxNumEpochs)
    num_Samples = 0; %initialise num of samples
    crossValApplied = 0; %1 if cross validation has been applied already
    
    %set number of samples
    inputDimension = size(size(inputA), 2); %confirm dimension, as input is converted to single col matrix
    if inputDimension == 2
        num_Samples = size(inputA, 2); %get numbr of samples from input matrix
        in_Nrns = size(inputA,1); %set number of input neurons
        crossValApplied = 1; %%do not do cross validation in the code. 
    elseif inputDimension == 3
        num_Samples = size(inputA, 3); %get numbr of samples from input matrix
        in_Nrns = size(inputA,1) * size(inputA,2); %set number of input neurons
    end
    
    %set momentum
    alpha = 1/num_Samples; %1/num_Samples or input neurons is recommended.
    
    epchs = 0;   %initialise main loop counter for running epochs
    
    numRowsTrainingData = size(inputA,1); %number of rows
    
    %define num of input neurons 
    in_Neurons = in_Nrns;
    
    %defn num of hiddn layer neurons
    hidden_Layer_Neurons = hidden_Neurons;
    
    %define num output neurons
    out_Neurons = outpt_Neurns;
    
    %init weights for the input layer
    %it is proven that initialsn weights with sqrt(2/(no of neurons from prev layer)) produces the
    %best performance
    in_Weight_Val = sqrt(2/in_Neurons);
    in_Weights = normrnd(0,in_Weight_Val, hidden_Layer_Neurons, in_Neurons);
    
    %init weights for the output layer
    out_Weight_Val = sqrt(2/hidden_Layer_Neurons);
    out_Weights = normrnd(0, out_Weight_Val, out_Neurons, hidden_Layer_Neurons);
    
    %init hidden layer bias, start with 0
    hidden_Bias = zeros(hidden_Layer_Neurons, 1);
    %init output layer bias, start with 0
    Out_Bias = zeros(out_Neurons, 1);
    
    MSE = 10; % init Mean Squared Error to large value
    trainingAccuracy = 0; %to track progress of training
    maxAccuracy = 1; % max possible accuracy, max is 1 or 100percent
    current_Err = 0; %initialise error
    prev_Accracy = 0; %initialise previous accuracy 
    curr_Accracy = 0; %initialise current accuracy
    %stop at specified percent accuracy to prevent overfitting and also save time
    while MSE > MSETolerance
        prevError = current_Err;
        current_Err = 0; %reset error from last epoch
        epchs=epchs+1; %increment number of epochs so far
        %reset epoch if necessary
        if epchs > maxNumEpochs
            epchs = 0;
            %init weights for the input layer
            %it is proven that initialsn weights with sqrt(2/(no of neurons from prev layer)) produces the
            %best performance
            in_Weight_Val = sqrt(2/in_Neurons);
            in_Weights = normrnd(0,in_Weight_Val, hidden_Layer_Neurons, in_Neurons);

            %init weights for the output layer
            out_Weight_Val = sqrt(2/hidden_Layer_Neurons);
            out_Weights = normrnd(0, out_Weight_Val, out_Neurons, hidden_Layer_Neurons);

            %init hidden layer bias, start with 0
            hidden_Bias = zeros(hidden_Layer_Neurons, 1);
            %init output layer bias, start with 0
            Out_Bias = zeros(out_Neurons, 1);
        end
        prev_Accracy = curr_Accracy;
        %%INITIALISE DELTA TERMS
        %Init the dlta term of the input weights to 0
        d_Term_In_Weights=zeros(hidden_Layer_Neurons,in_Neurons);
        %init the dlta term of the output weights to 0
        d_Term_Out_Weights=zeros(out_Neurons,hidden_Layer_Neurons);
        %init the dlta term of the output bias to 0
        d_Term_Out_Bias=zeros(out_Neurons,1);
        
        %Feed input matrix to input layer, feedforward, backpropagate, correct weights and biases wrt
        %the cost function
        for i=1:num_Samples
            %FORWARD PROPAGATN
            
            %init of inpt and outpt layrs
            
            %current input to input layer
            if inputDimension == 2
                for rows=1:numRowsTrainingData
                    %set input to input neurons of input layer
                    inInputNeurons(rows,1) = inputA(rows,i);
                end
            elseif inputDimension == 3
                %convert current i/p to inpt layer to single column matrx
                for rows=1:numRowsTrainingData
                    for cols=1:numRowsTrainingData
                        %set input to input neurons of input layer
                        inInputNeurons(cols+(rows-1) * numRowsTrainingData,1) = inputA(rows,cols,i);
                    end
                end
            end
            %set outpt neurn for the particular input to 1, others remain zero
            outNeurons = zeros(out_Neurons, 1);
            outNeurons(desiredOutD(i) + 1, 1) = 1;

            %compute i/p to Hidden Layr neurons
            zValHidden(:,1) = in_Weights * inInputNeurons+hidden_Bias(:, 1); %compute zVal for hidden
            inHiddenNeurons(:, 1) = activn_Function(zValHidden(:, 1)); %compute inpt to hiddn neurons

            %Compute i/p to Outpt layr neurons
            zValOutpt = out_Weights*inHiddenNeurons(:, 1) + Out_Bias; %compute zVal for outpt
            inOutputNeurons = activn_Function(zValOutpt); %compute inpt to outpt neurons

            %BACK PROPAGATN
            %find dlta-terms or gradnt of outpt layer with
            d_Term_Output = activn_Function_Derv(zValOutpt).*(inOutputNeurons - outNeurons);
            %find dlta-terms or gradnt of hiddn layer with
            d_Term_Hidden(:,1) = activn_Function_Derv(zValHidden(:,1)).*(out_Weights.')*d_Term_Output;

            %UPDATE WEIGTS AND BIASES BASED ON CST FUNCTN USING dterm
            %update Whj
            dTermProd = d_Term_Output * (inHiddenNeurons(:, 1).'); %dlta term of outpt layer * input of hidden layr neurons
            d_Term_Out_Weights = d_Term_Out_Weights + dTermProd; %modified weight for Whj
            d_Term_Out_Bias = d_Term_Out_Bias + d_Term_Output; %modified bias 
            %update Wih
            dTermProd = d_Term_Hidden(:, 1) * inInputNeurons.'; %dlta term of hidden layer * input of inpt layr neurons
            d_Term_In_Weights = d_Term_In_Weights + dTermProd; %modified weight for Wih
            
            %calc err
            errVal = (inOutputNeurons - outNeurons)' * (inOutputNeurons - outNeurons);
            %add to err countr
            current_Err = current_Err + errVal;
    
            indx = find(inOutputNeurons == max(inOutputNeurons)); %get output neuron with max val
            if indx == (desiredOutD(i) + 1) %check if it matches the labls
                trainingAccuracy = trainingAccuracy + 1; %increment the accracy
            end
        end
        %modify hidden and outpt  weights and biases with lrning rate for next epoch 
        in_Weights= in_Weights - eta .* d_Term_In_Weights; %%inpt to hiddn layer weights
        out_Weights= out_Weights - eta .* d_Term_Out_Weights; %%hiddn to outpt layr weights
        Out_Bias= Out_Bias - eta .* d_Term_Out_Bias; %%outpt layr bias

        %modify dterm weights and biases for next epoch
        d_Term_Out_Weights = alpha .* d_Term_Out_Weights;
        d_Term_In_Weights = alpha .* d_Term_In_Weights; 
        d_Term_Out_Bias = alpha .* d_Term_Out_Bias;
        
        trainingAccuracy = trainingAccuracy / num_Samples; %compute current trainng accuracy

        %compare eror from now and previous and increment or decrement the learning rate
        if epchs > 0
            if current_Err < prevError
                eta=eta*1.0050;
            else
                eta=eta*0.050;
            end
        end
        
        MSE = current_Err/num_Samples; %calculate current Mean Squared Error
        curr_Accracy = floor(trainingAccuracy * 100); %current Accuracy
        if curr_Accracy ~= prev_Accracy
            %display only if accuracy is changed
            fprintf('Current MSE = %.2f Accuracy is %d percent. Still training... \n', MSE, curr_Accracy);
        end
        %%apply cross validation to dataset
        if (curr_Accracy > 70) && (crossValApplied == 0)
            [inputA, desiredOutD, validA, validD] = crossValChange(dataSetLink, inputA, validA, desiredOutD, validD);
            fprintf('\ncross validation applied.\n\n');
            crossValApplied = 1;
        end
    end
    fprintf('TRAINING COMPLETE!\n');
    Wih = in_Weights; 
    Bh = hidden_Bias;
    Whj = out_Weights;
    Bo = Out_Bias;
    achievedMSE = MSE;
    numEpochsDone = epchs;
end
%%BACKPROP ALG FUNCTION ENDS HERE

%sigmoid activatn functn
function res = activn_Function(funcInput)
    res = 1./(1+exp(-1.*funcInput));
end
%sigmoid activatn functn with derivative
function res = activn_Function_Derv(funcInput)
    res = 1./(1+exp(-1.*funcInput));
    %find derivative
    der = 1. - res;
    res = res .* der;
end

%this function makes desired outputs/labels categorical
function catD = categoricalD(DLabels)
    catD = zeros(size(DLabels, 2), 1); %initalise categories
    for i=1:size(DLabels, 2)
       if DLabels(1, i) == 1 catD(i) = 0; end %100 for cat 0
       if DLabels(2, i) == 1 catD(i) = 1; end %010 for cat 1
       if DLabels(3, i) == 1 catD(i) = 2; end %001 for cat 2
    end
end
%this function converts the desired outputs of Iris dataset from categorical back to normal
function mat = convertFromCategorical(Lbls)
    converted = zeros(3, size(Lbls,1)); 
    for i=1:size(Lbls, 1)
        if Lbls(i) == 0 %set category 0 back to 100
           converted(1,i) = 1; 
           converted(2,i) = 0;
           converted(3,i) = 0;
        elseif Lbls(i) == 1 %set category 1 back to 010
           converted(1,i) = 0;
           converted(2,i) = 1;
           converted(3,i) = 0;
        elseif Lbls(i) == 2 %set category 2 back to 001
           converted(1,i) = 0;
           converted(2,i) = 0;
           converted(3,i) = 1;
        end
        
    end
    mat = converted; %return normal iris classes 
end
%%import data from excel file (for training and validtion)
function [images_Training, labels_Training, images_Validation, labels_Validation, division_Point] = importTVData(mainDir)
    file_Name = strcat(mainDir, '/index.csv'); 
    tabl = readtable(file_Name); %imprt data to table
    all_Labels = tabl.Var1; %read first column as labls
    all_Images = tabl.Var2; %read second column as imgs
    all_Images = strcat(mainDir, '/', all_Images);
    totalNumImages = size(all_Labels,1); %get total num of imgs
    division_Point = round(8/10 * totalNumImages); %get splitting point btw trainng and validatn
    trainingImgs = all_Images(1:division_Point); %get traing imgs
    trainingLbls = all_Labels(1:division_Point); %get traing lbls
    validationImgs = all_Images(division_Point+1:totalNumImages); %get valdn imgs
    validationLbls = all_Labels(division_Point+1:totalNumImages); %get valdn lbls
    
    %initialise training and validatn sets
    training_images = zeros(28, 28, division_Point);
    testing_images = zeros(28, 28, totalNumImages - division_Point);
    training_labels = zeros(division_Point, 1);
    testing_labels = zeros(totalNumImages-division_Point, 1);
    
    %load images for training
    for i=1:division_Point
       rawImage = imread(char(trainingImgs(i))); %read imge to matrx
       levelRawImage = graythresh(rawImage); %specify threshold for convrtion from gray
       bwImage = im2bw(rawImage, levelRawImage); %convrt to blck and white
       training_images(:,:,i) = bwImage;
    end
    %load labls for training
    for i=1:division_Point
       training_labels(i) = trainingLbls(i);
    end
    %load images for validation
    for i=1:(totalNumImages-division_Point)
       rawImage = imread(char(validationImgs(i))); %read imge to matrx
       levelRawImage = graythresh(rawImage); %specify threshold for convrtion from gray
       bwImage = im2bw(rawImage, levelRawImage); %convrt to black and white
       testing_images(:,:,i) = bwImage;
    end
    %load labels for validation
    for i=1:(totalNumImages-division_Point)
       testing_labels(i) = validationLbls(i);
    end
    %return trainng and validatn sets
    images_Training = training_images;
    labels_Training = training_labels;
    images_Validation = testing_images;
    labels_Validation = testing_labels;
end
%cross validation: change training and validation dataset
function [new_TrainingSet, new_TrainingLbls, new_ValidationSet, new_ValidationLbls] = crossValChange(dataSetLnk, old_TrainingSet, old_ValidationSet, old_TrainingLbls, old_ValidationLbls)
    
    indxFile = strcat(dataSetLnk,'/index.csv');
    tabl = readtable(indxFile); %load data from file to tabl
    all_Labels = tabl.Var1; %get labls
    all_Images = tabl.Var2; %get imgs
    all_Images = strcat(dataSetLnk, '/', all_Images);
    totalNumImages = size(all_Labels,1); %get totl num of imgs
    division_Point = round(8/10 * totalNumImages); %set splitting point, trainng and validn
    
    %trainngImgs now take the lst 80% while validn Imgs now take the first
    %20%
    trainingImgs = all_Images((size(old_ValidationLbls,1)+1) : totalNumImages);
    trainingLbls = all_Labels((size(old_ValidationLbls,1)+1) : totalNumImages);
    validationImgs = all_Images(1:(size(old_ValidationLbls,1)));
    validationLbls = all_Labels(1:(size(old_ValidationLbls,1)));
    
    %initialise trainng and validn sets
    training_images = zeros(28, 28, division_Point);
    testing_images = zeros(28, 28, totalNumImages - division_Point);
    training_labels = zeros(division_Point, 1);
    testing_labels = zeros(totalNumImages-division_Point, 1);
    
    %set images for training
    for i=1:division_Point
       rawImage = imread(char(trainingImgs(i))); %read imge to matrx
       levelRawImage = graythresh(rawImage); %get threshold for conversn frm gray
       bwImage = im2bw(rawImage, levelRawImage); %convrt to black and white
       training_images(:,:,i) = bwImage;
    end
    %set labels for training
    for i=1:division_Point
       training_labels(i) = trainingLbls(i);
    end
    %set images for validation
    for i=1:(totalNumImages-division_Point)
       rawImage = imread(char(validationImgs(i))); %read imge to matrx
       levelRawImage = graythresh(rawImage); %get threshold for conversn frm gray
       bwImage = im2bw(rawImage, levelRawImage); %convrt to black and white
       testing_images(:,:,i) = bwImage;
    end
    %set labels for validation
    for i=1:(totalNumImages-division_Point)
       testing_labels(i) = validationLbls(i);
    end
    %return new traininng and validatn sets
    new_TrainingSet = training_images;
    new_TrainingLbls = training_labels;
    new_ValidationSet = testing_images;
    new_ValidationLbls = testing_labels;
end
