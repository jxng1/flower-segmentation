close all;
clear;

% SETTINGS
% -------------------------------------------------------------------------
% training settings
trainNewModel = true;
saveModel = true;
maxEpochs = 1;
miniBatchSize = 32;
validationFrequency = 25;

% preprocess settings
preprocessData = true;

% fix missing labels
fixLabels = true;

% random affine settings
randomAffine2DSettings = struct;
randomAffine2DSettings.apply = true; % set to false, if not applied
randomAffine2DSettings.rotationRange = [-45, 45]; % set to 0, 0 if no rot
randomAffine2DSettings.scaleRange = [0.9, 1.5]; % set to 1, 1 if no scale
randomAffine2DSettings.xTranslationRange = [0, 0]; % set to 0, 0 if no translation
randomAffine2DSettings.yTranslationRange = [0, 0]; % set to 0, 0 if no translation
randomAffine2DSettings.reflectXAxis = false;
randomAffine2DSettings.reflectYAxis = true;
randomAffine2DSettings.xShearRange = [-10, 10]; % set to 1, 1 if no yshear
randomAffine2DSettings.yShearRange = [-10, 10]; % set to 1, 1 if no xshear
randomAffine2DSettings.cropSize = [256, 256]; % set to 256, 256 if no crop

% blur settings
blurSettings = struct;
blurSettings.apply = false;
% methods: ['gaussian', 'box', 'median', 'anisotropic']
blurSettings.method = 'gaussian';
% parameters: gaussian={sigma} e.g. {[2, 2]}
% box={kernelSize} e.g. {5}
% median={windowSize} e.g. {[3, 3]}
% anisotropic={numIterations} e.g. {20}
blurSettings.params = {2.5};

% sharpen settings
sharpenSettings = struct;
sharpenSettings.apply = false;
sharpenSettings.amount = 0.5;
sharpenSettings.radius = 2.5;

% enhancement settings
enhancementSettings = struct;
enhancementSettings.apply = true;
enhancementSettings.hueAmount = 0.1; % set to 0 for none
enhancementSettings.saturationAmount = 0.2; % set to 0 for none
enhancementSettings.brightnessAmount = 0.3; % set to 0 for none
enhancementSettings.contrastAmount = 0.4; % set to 0 for none
% -------------------------------------------------------------------------

dirBaseData = 'data/';
dirTrain = 'images_256/';
dirLabels = 'labels_256/';

% don't use images without corresponding label
imageFiles = dir(fullfile(dirBaseData, dirTrain, '*.jpg'));
labelFiles = dir(fullfile(dirBaseData, dirLabels, '*.png'));

% extract names without extensions
imageFilesNames = {imageFiles.name};
labelFilesNames = {labelFiles.name};
imageFilesNames = cellfun(@(x) strtok(x, '.'), ...
    imageFilesNames, ...
    'UniformOutput', false);
labelFileNames = cellfun(@(x) strtok(x, '.'), ...
    labelFilesNames, ...
    'UniformOutput', false);

% find names in both via intersect
commonNames = intersect(imageFilesNames, labelFileNames);

% reconstruct extensions
commonImageFiles = strcat(fullfile(dirBaseData, dirTrain), ...
    filesep, ...
    commonNames, ...
    '.jpg');
commonLabelFiles = strcat(fullfile(dirBaseData, dirLabels), ...
    filesep, ...
    commonNames, ...
    '.png');

% setup imds
imds = imageDatastore(commonImageFiles, ...
    'FileExtensions', '.jpg');

% setup pxds
classNames = ["null", "flower", "leaves", "background", "sky"];
pixelLabelIDs = [0, 1, 2, 3, 4];
pxds = pixelLabelDatastore(commonLabelFiles, ...
    classNames, ...
    pixelLabelIDs, ...
    'FileExtensions', '.png');

% combine datastores
cds = combine(imds, pxds);

% shuffle datastore
cds = shuffle(cds);
% subset data into training, validation and testing
cdsLength = numel(cds.UnderlyingDatastores{1}.Files);
trainIdx = floor(0.80 * cdsLength);
validationIdx = floor(0.10 * cdsLength);
cdsTrain = subset(cds, 1:trainIdx);
cdsValidation = subset(cds, trainIdx:(trainIdx + validationIdx));
cdsTest = subset(cds, (trainIdx + validationIdx):cdsLength);

if trainNewModel    
    % preprocess image given flag
    tcds = cdsTrain.copy;
    if preprocessData
        % replace labels with 'background' and 'flower' class using imclose
        if fixLabels
            tcds = transform(tcds, ...
                @(data)fixClassLabels(data));
            
            figure
            cds = combine(cdsTrain, tcds);
            dataPre = preview(tcds);
            lo1 = labeloverlay(dataPre{1, 1}, dataPre{1, 2});
            %imshowpair(lo1, imread(cdsTrain.UnderlyingDatastores{1}.Files{1}), ...
            %    'montage')
            imshow(lo1);
            title('Augmented(Label Replaced)')
            %title('Augmented(Label Replaced) vs Base')
            axis off;
        end
        
        % Random 2D Affines
        if randomAffine2DSettings.apply
            tcds = transform(tcds, ...
                @(data)doRandomAffine2D(data, ...
                randomAffine2DSettings));

            figure
            cds = combine(cdsTrain, tcds);
            dataPre = preview(tcds);
            lo1 = labeloverlay(dataPre{1, 1}, dataPre{1, 2});
            %imshowpair(lo1, imread(cdsTrain.UnderlyingDatastores{1}.Files{1}), ...
            %    'montage')
            imshow(lo1);
            title('Augmented(2D Affine)')
            %title('Augmented(Label Replaced) vs Base')
            axis off;
        end
    
        % Enhancement
        if enhancementSettings.apply
            tcds = transform(tcds, ...
                @(data)doEnhancement(data, enhancementSettings));

            figure
            cds = combine(cdsTrain, tcds);
            dataPre = preview(tcds);
            lo1 = labeloverlay(dataPre{1, 1}, dataPre{1, 2});
            %imshowpair(lo1, imread(cdsTrain.UnderlyingDatastores{1}.Files{1}), ...
            %    'montage')
            imshow(lo1);
            title('Augmented(Enchanced)')
            %title('Augmented(Label Replaced) vs Base')
            axis off;
        end

        % Random Blur
        if blurSettings.apply
            blurMethod = blurSettings.method;
            if strcmp(blurMethod, 'gaussian')
                tcds = transform(tcds, ...
                @(data)doBlur(data, blurSettings));
            elseif strcmp(blurMethod, 'box')
                tcds = transform(tcds, ...
                @(data)doBlur(data, blurSettings));
            elseif strcmp(blurMethod, 'median')
                tcds = transform(tcds, ...
                @(data)doBlur(data, blurSettings));
            elseif strcmp(blurMethod, 'anisotropic')
                tcds = transform(tcds, ...
                @(data)doBlur(data, blurSettings));
            else
                error("Blur method is invalid.");
            end

            figure
            cds = combine(cdsTrain, tcds);
            dataPre = preview(tcds);
            lo1 = labeloverlay(dataPre{1, 1}, dataPre{1, 2});
            %imshowpair(lo1, imread(cdsTrain.UnderlyingDatastores{1}.Files{1}), ...
            %    'montage')
            imshow(lo1);
            title('Augmented(Blurred)')
            %title('Augmented(Label Replaced) vs Base')
            axis off;
        end

        % Sharpen
        if sharpenSettings.apply
            tcds = transform(tcds, ...
                @(data)doSharpen(data, sharpenSettings));

           figure
            cds = combine(cdsTrain, tcds);
            dataPre = preview(tcds);
            lo1 = labeloverlay(dataPre{1, 1}, dataPre{1, 2});
            %imshowpair(lo1, imread(cdsTrain.UnderlyingDatastores{1}.Files{1}), ...
            %    'montage')
            imshow(lo1);
            title('Augmented(Sharpened)')
            %title('Augmented(Label Replaced) vs Base')
            axis off;
        end

        % form final training combined datastore
        cdsTrainFinal = tcds;
    else
        cdsTrainFinal = cdsTrain;
    end
    
    % setup net
    imageSize = [256, 256, 3];
    numClasses = numel(classNames);

    % encoder layers
    encoderLayers = [
        imageInputLayer(imageSize, 'Name', 'input_image')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_1')
        batchNormalizationLayer('Name', 'bn1_1')
        reluLayer('Name', 'relu1_1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_2')
        batchNormalizationLayer('Name', 'bn1_2')
        reluLayer('Name', 'relu1_2')
        
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2_1')
        batchNormalizationLayer('Name', 'bn2_1')
        reluLayer('Name', 'relu2_1')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2_2')
        batchNormalizationLayer('Name', 'bn2_2')
        reluLayer('Name', 'relu2_2')
        
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3_1')
        batchNormalizationLayer('Name', 'bn3_1')
        reluLayer('Name', 'relu3_1')
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3_2')
        batchNormalizationLayer('Name', 'bn3_2')
        reluLayer('Name', 'relu3_2')
        
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
        
        convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv4_1')
        batchNormalizationLayer('Name', 'bn4_1')
        reluLayer('Name', 'relu4_1')
        convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv4_2')
        batchNormalizationLayer('Name', 'bn4_2')
        reluLayer('Name', 'relu4_2')

        dropoutLayer(0.5, 'Name', 'drop4')

        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4')

        convolution2dLayer(3, 1024, 'Padding', 'same', 'Name', 'conv5_1')
        batchNormalizationLayer('Name', 'bn5_1')
        reluLayer('Name', 'relu5_1')
        convolution2dLayer(3, 1024, 'Padding', 'same', 'Name', 'conv5_2')
        batchNormalizationLayer('Name', 'bn5_2')
        reluLayer('Name', 'relu5_2')
        dropoutLayer(0.5, 'Name', 'drop5')
    ];

    % decoder layers
    decoderLayers = [
        transposedConv2dLayer(2, 512, 'Stride', 2, 'Name', 'transconv4')
        batchNormalizationLayer('Name', 'bn_transconv4')
        concatenationLayer(3, 2, 'Name', 'concat4')
        convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'upconv4_1')
        batchNormalizationLayer('Name', 'bn_upconv4_1')
        reluLayer('Name', 'uprelu4_1')
        convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'upconv4_2')
        batchNormalizationLayer('Name', 'bn_upconv4_2')
        reluLayer('Name', 'uprelu4_2')
        
        transposedConv2dLayer(2, 256, 'Stride', 2, 'Name', 'transconv3')
        batchNormalizationLayer('Name', 'bn_transconv3')
        concatenationLayer(3, 2, 'Name', 'concat3')
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'upconv3_1')
        batchNormalizationLayer('Name', 'bn_upconv3_1')
        reluLayer('Name', 'uprelu3_1')
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'upconv3_2')
        batchNormalizationLayer('Name', 'bn_upconv3_2')
        reluLayer('Name', 'uprelu3_2')
        
        transposedConv2dLayer(2, 128, 'Stride', 2, 'Name', 'transconv2')
        batchNormalizationLayer('Name', 'bn_transconv2')
        concatenationLayer(3, 2, 'Name', 'concat2')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'upconv2_1')
        batchNormalizationLayer('Name', 'bn_upconv2_1')
        reluLayer('Name', 'uprelu2_1')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'upconv2_2')
        batchNormalizationLayer('Name', 'bn_upconv2_2')
        reluLayer('Name', 'uprelu2_2')
        
        transposedConv2dLayer(2, 64, 'Stride', 2, 'Name', 'transconv1')
        batchNormalizationLayer('Name', 'bn_transconv1')
        concatenationLayer(3, 2, 'Name', 'concat1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'upconv1_1')
        batchNormalizationLayer('Name', 'bn_upconv1_1')
        reluLayer('Name', 'uprelu1_1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'upconv1_2')
        batchNormalizationLayer('Name', 'bn_upconv1_2')
        reluLayer('Name', 'uprelu1_2')
        
        convolution2dLayer(1, numClasses, 'Name', 'output_conv')
        softmaxLayer('Name', 'softmax')
        pixelClassificationLayer('Name', 'output')
    ];

    layers = [
        encoderLayers
        decoderLayers
    ];

    net = layerGraph(layers);

    net = disconnectLayers(net, 'bn_transconv4', 'concat4/in1');
    net = connectLayers(net, 'drop4', 'concat4/in1');
    net = connectLayers(net, 'bn_transconv4', 'concat4/in2');

    net = disconnectLayers(net, 'bn_transconv3', 'concat3/in1');
    net = connectLayers(net, 'conv3_2', 'concat3/in1');
    net = connectLayers(net, 'bn_transconv3', 'concat3/in2');

    net = disconnectLayers(net, 'bn_transconv2', 'concat2/in1');
    net = connectLayers(net, 'conv2_2', 'concat2/in1');
    net = connectLayers(net, 'bn_transconv2', 'concat2/in2');

    net = disconnectLayers(net, 'bn_transconv1', 'concat1/in1');
    net = connectLayers(net, 'conv1_2', 'concat1/in1');
    net = connectLayers(net, 'bn_transconv1', 'concat1/in2');

    %analyzeNetwork(net);
    
    % specify training options
    options = trainingOptions('sgdm', ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'InitialLearnRate', 1e-4, ...
        'L2Regularization', 1e-2, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'gpu', ...
        'ValidationData', cdsValidation, ...
        'ValidationFrequency', validationFrequency, ...
        'Verbose', true);

    %analyzeNetwork(net);

    % net = dlnetwork(net, 'Initialize', true);
    % Y = rand(numClasses, 12);
    % labels = randi(numClasses, [1, 12]);
    % T = onehotencode(labels, 1, 'ClassNames', 1:numClasses);
    % weights = dlarray(rand(1, numClasses));

    %lossFcn = @(Y, T) weightedCrossEntropyLoss(Y, T, weights);
    %[net, info] = trainnet(cdsTrainFinal, net, lossFcn, options);
    [net, info] = trainNetwork(cdsTrainFinal, net, options);

    % save network after training
    dtStr = string(datetime('now', 'Format', 'yyyyMMdHHmmss'));
    if saveModel
        % for repeated testing
        % save(fullfile('saved_models', 'segmentationOwn', dtStr + '_segmentownnet.mat'), ...
        %     'net');
        % final save
        save('segmentownnet.mat', 'net');
    end

    pretrainedModel.net = net;
end

% TODO: needs work upon for purely loading a trained model(final step)
if ~trainNewModel
    % pretrainedModel = load(fullfile('saved_models', ...
    %     'segmentationOwn', ...
    %     '20240425214302_segmentownnet.mat'))
    pretrainedModel = load('segmentownnet.mat')
end

% segment images
results = semanticseg(cdsTest.UnderlyingDatastores{1}, ...
    pretrainedModel.net, ...
    'MiniBatchSize', 1, ...
    'Classes', ["null", "flower", "leaves", "background", "sky"], ...
    'WriteLocation', "temp");

% segmented labels output using datetime
dt = datetime('now', ...
    'TimeZone', 'local', ...
    'Format', 'yyyyMMdHHmmss');
dtStr = string(dt);
saveLocationStr = dtStr; %+ '_' + detailStr;
mkdir(fullfile('images', 'segmentationOwn'), ...
    fullfile(saveLocationStr));

% show a couple result images
% numToShow = randi(numel(results.Files), [1, 16]); % randomised
showCollage(cdsTest.UnderlyingDatastores{1}, results, double(1):double(16)); % show 1:16
% save collage
saveas(gcf, ...
    fullfile('images', 'segmentationOwn', ...
    saveLocationStr, 'collage.png'));

% evaluate model
metrics = evaluateSemanticSegmentation(results, ...
    cdsTest.UnderlyingDatastores{2}, ...
    "Metrics", "all");

% plot metrics
% calculate conf matrix
figure
cm = confusionchart(metrics.ConfusionMatrix.Variables, ...
  classNames, Normalization='row-normalized');
cm.Title = 'Normalized Confusion Matrix (%)';
% save conf matrix
saveas(gcf, ...
    fullfile('images', 'segmentationOwn', ...
    saveLocationStr, 'confMat.png'));

% calculate mean IoU
imageIoU = metrics.ImageMetrics.MeanIoU;
figure
histogram(imageIoU)
title('Image Mean IoU');
% save histogram
saveas(gcf, ...
    fullfile('images', 'segmentationOwn', ...
    saveLocationStr, 'hist.png'));

% save settings to a .txt file(for readability)
fileID = fopen(fullfile('images', ...
    'segmentationOwn', ...
    saveLocationStr, ...
    'settings.txt'), "w");
settingsText = sprintf(['Saved at: %s\n\n' ...
    'trainNewModel=%s\n' ...
    'saveModel=%s\n' ...
    'maxEpochs=%d\n', ...
    'miniBatchSize=%d\n' ...
    'validationFrequency=%d\n' ...
    '\nPreprocessing Settings\n', ...
    'preprocessData=%s\n', ...
    'fixLabels=%s\n', ...
    '\nRandom Affine Settings\n', ...
    'apply=%s\n', ...
    'rotationRange=%s\n', ...
    'scaleRange=%s\n', ...
    'xTranslationRange=%s\n', ...
    'yTranslationRange=%s\n', ...
    'reflectXAxis=%s\n', ...
    'reflectYAxis=%s\n', ...
    'xShearRange=%s\n', ...
    'yShearRange=%s\n', ...
    'cropSize=%s\n', ...
    '\nBlur Settings\n', ...
    'apply=%s\n', ...
    'method=%s\n', ...
    'params=%s\n', ...
    '\nSharpen Settings\n', ...
    'apply=%s\n', ...
    'amount=%.2f\n', ...
    'radius=%.2f\n'], ...
    dtStr, ...
    string(trainNewModel), ...
    string(saveModel), ...
    maxEpochs, ...
    miniBatchSize, ...
    validationFrequency, ...
    string(preprocessData), ...
    string(fixLabels), ...
    string(randomAffine2DSettings.apply), ...
    num2str(randomAffine2DSettings.rotationRange), ...
    num2str(randomAffine2DSettings.scaleRange), ...
    num2str(randomAffine2DSettings.xTranslationRange), ...
    num2str(randomAffine2DSettings.yTranslationRange), ...
    string(randomAffine2DSettings.reflectXAxis), ...
    string(randomAffine2DSettings.reflectYAxis), ...
    num2str(randomAffine2DSettings.xShearRange), ...
    num2str(randomAffine2DSettings.yShearRange), ...
    num2str(randomAffine2DSettings.cropSize), ...
    string(blurSettings.apply), ...
    blurSettings.method, ...
    string(blurSettings.params), ...
    string(sharpenSettings.apply), ...
    sharpenSettings.amount, ...
    sharpenSettings.radius);
fwrite(fileID, settingsText);
fclose(fileID);

% FUNCTIONS
% -------------------------------------------------------------------------
function showCollage(imds, pxds, numToShow)
    images = {};
    for idx = 1:numel(numToShow)
        overlayOut = labeloverlay(readimage(imds, numToShow(idx)), ...
            readimage(pxds, numToShow(idx)));
        images = cat(4, images, overlayOut);
    end
    % show collage of images
    figure
    montage(images, ...
        'Size', [4, numel(numToShow) / 4]);
    title("Example Collage");
    axis off;
end

function out = doRandomAffine2D(data, options)
    I = data{1};
    C = data{2};

    transform = randomAffine2d('Rotation', options.rotationRange, ...
        'Scale', options.scaleRange, ...
        'XTranslation', options.xTranslationRange, ...
        'YTranslation', options.yTranslationRange, ...
        'XReflection', options.reflectXAxis, ...
        'YReflection', options.reflectYAxis, ...
        'XShear', options.xShearRange, ...
        'YShear', options.yShearRange);

    tI = imwarp(I, transform);
    tC = imwarp(C, transform);

    cropSize = options.cropSize;
    if cropSize < 256
        window = centerCropWindow2d(size(I), options.cropSize);
        tI = imcrop(tI, window);
        tC = imcrop(tC, window);
    end 

    [width, length] = size(tC);
    if width ~= 256 | length ~= 256
        tI = imresize(tI, [256, 256], 'box');
        tC = imresize(tC, [256, 256], 'box');
    end

    out{1} = tI;
    out{2} = tC;
end

function out = doBlur(data, options)
    I = data{1};

    % blur
    blurMethod = options.method;
    params = options.params;
    if strcmp(blurMethod, 'gaussian')
        param = params{1};
        blurredI = imgaussfilt(I, param);
    elseif strcmp(blurMethod, 'box')
        param = params{1};
        boxBlurKernel = ones(param) / (param * param);
        blurredI = imfilter(I, boxBlurKernel);
    elseif strcmp(blurMethod, 'median')
        param = params{1};
        blurredI = medfilt2(I, param);
    elseif strcmp(blurMethod, 'anisotropic')
        param = params{1};
        % imdiffusefilt must be done to each colour channel individually
        doubleI = im2double(I);
        
        diffusedIR = imdiffusefilt(doubleI(:, :, 1), ...
            'NumberOfIterations', param);
        diffusedIG = imdiffusefilt(doubleI(:, :, 2), ...
            'NumberOfIterations', param);
        diffusedIB = imdiffusefilt(doubleI(:, :, 2), ...
            'NumberOfIterations', param);
        blurredI = cat(3, diffusedIR, diffusedIG, diffusedIB);
    end

    out{1} = blurredI;
    out{2} = data{2}; % unchanged
end

function out = doSharpen(data, options)
    I = data{1};

    sharpenedI = imsharpen(I, ...
        'Amount', options.amount, ...
        'Radius', options.radius);

    out{1} = sharpenedI;
    out{2} = data{2}; % unchanged
end

function out = doEnhancement(data, options)
    I = data{1};

    enhancedI = jitterColorHSV(I, ...
        'Contrast', options.contrastAmount, ...
        'Hue', options.hueAmount, ...
        'Saturation', options.saturationAmount, ...
        'Brightness', options.brightnessAmount);

    out{1} = enhancedI;
    out{2} = data{2};
end

function out = fixClassLabels(data)
    C = data{2};

    numeric = int16(double(C));

    se = strel('disk', 12, 8);
    closedF = imdilate(numeric == 2, se);
    closedB = imdilate(numeric == 4, se);

    % closedF = imdilate(closedF == 2, strel('square', 3));
    % closedB = imdilate(closedB == 4, strel('diamond', 3));

    numeric(closedF & numeric == 1) = 2;
    numeric(closedB & (numeric == 3 | numeric == 5)) = 4;

    C = categorical(numeric, ...
        [1, 2, 3, 4, 5], ...
        {'null', 'flower', 'leaves', 'background', 'sky'});

    out{1} = data{1};
    out{2} = C;
end

% unused
% function loss = weightedCrossEntropyLoss(Y, T, weights)
%     % Replace NaN values with 0
%     T(isnan(T)) = 0;
% 
%     loss =  crossentropy(Y, T, weights, 'WeightsFormat', 'C');
% end