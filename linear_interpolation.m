% Set parameters
clear
tic
display = true;
save_net = false;
perform_SVD = false;

Setting = 1;
sensorNum = 30; % set number of sensors
epsilon = 0.1; % set noise factor

%% Get sensor locations
switch Setting
    case 1
        sensors = load('C:\Users\alonf\OneDrive - Technion\Thesis\DL Interpolation\loc_final_Entropy_median1.mat', 'loc_final');
    case 2
        sensors = load('C:\Users\alonf\OneDrive - Technion\Thesis\DL Interpolation\loc_final_Entropy_median2.mat', 'loc_final');
end
sensors = sensors.loc_final;

%% create data store
% get file list and make datastore
switch Setting
    case 1 
        folderPath = "C:\Users\alonf\OneDrive - Technion\Thesis\DL Interpolation\sinthetic";
    case 2
        folderPath = "C:\Users\alonf\OneDrive - Technion\Thesis\DL Interpolation\sinthetic2";
end
ds = fileDatastore(folderPath,"IncludeSubfolders",true,"FileExtensions",".mat","ReadFcn",make_read_func(sensorNum,epsilon));

% split to train, validation and test:
rng(0)
[trainId,valId,testId] = dividerand(numel(ds.Files),0.7,0.1,0.2);
trainData = subset(ds,trainId);
valData = subset(ds,valId);
testData = subset(ds,testId);

%% display data as montage
if display
reset(testData)
fig1 = figure(Position=[0 0 1500 2000]);
img_num = 9;
images = cell(1,img_num);
for i=1:img_num
    X = read(testData);
    subplot('Position',[0 0.05 0.88 0.85])
    imagesc(X{1,2})
    colorbar
    colormap hot
    axis off
    ax = gca;
    ax.FontSize = 50;
    F = getframe(fig1); 
    images{1,i} = F.cdata; 
end
close all
montage(images)
end

%% make net
layers = [
    imageInputLayer([sensorNum 1 1],"Name","imageinput")
    fullyConnectedLayer(10000,"Name","fc")
    depthToSpace2dLayer([100 100],"Name","depthToSpace","Mode","crd")
    regressionLayer("Name","regressionoutput")];
% analyzeNetwork(layers)

%% training
% set training hyper=parameters
options = trainingOptions("adam", ...
    MaxEpochs=10, ...
    MiniBatchSize=128, ...
    ValidationData=valData, ...
    ValidationPatience=500, ...
    Shuffle="every-epoch", ...
    LearnRateSchedule="piecewise", ...
    InitialLearnRate=0.1, ...
    LearnRateDropPeriod = 1, ...
    LearnRateDropFactor = 0.1, ...
    L2Regularization = 5, ...
    Verbose = 0, ...
    OutputNetwork="best-validation-loss");

% train and save trained network
net = trainNetwork(trainData,layers,options);
if save_net
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save("trainedVecToImageRegressionNet-"+modelDateTime+".mat","net");    
end

%% Evaluate model
reset(testData)
N = numel(testData.Files);
corr_list = zeros(N,1);
rmse_list = zeros(N,1);
i = 1;
while hasdata(testData)
    X = read(testData);
    ypred = predict(net,X{1,1});
    corr_list(i) = mat_corr(X{1,2},ypred);
    rmse_list(i) = rmse(X{1,2}, ypred,"all");
    i = i + 1;
end
disp("Summery")
Setting
sensorNum
epsilon
meanCorr = mean(corr_list)
meanRMSE = mean(rmse_list)

%%
if display
% display 50 test samples as video
reset(testData)
figure
sgtitle('Model predictions on test samples')
for i=1:50
    X = read(testData);
    X_min = min(X{1,2}(:));
    X_max = max(X{1,2}(:))+1e-16;
    subplot(1,4,1)
    imagesc(X{1,1},[X_min X_max])
    colorbar
    title("input - sensors")
    subplot(1,4,2)
    ypred = predict(net,X{1,1});
    imagesc(ypred,[X_min X_max])
    colorbar
    corr = mat_corr(X{1,2},ypred);
    title(sprintf("prediction - correlation = %.3f",corr))
    subplot(1,4,3)
    imagesc(X{1,2},[X_min X_max])
    colorbar
    title("true map")
    subplot(1,4,4)
    z = accuracyCriterion(X{1,2}, ypred,0.5 ,2 ,0.10 * X_max);
    imagesc(z)
    acc = sum(z,"all")/numel(z);
    title(sprintf("binary map  - accuracy %.2f%%",100*acc))
    colormap hot
    pause(3)
end
end

%% display as montage
% display line_num examples in montage
if display
fig1 = figure("Position",[150 150 4000 800]);
reset(testData)
line_num = 8;
images = cell(1,line_num);
hold on
for i=1:line_num
    hold on
    X = read(testData);
    X_min = min(X{1,2}(:));
    X_max = max(X{1,2}(:));
    subplot(1,4,1,'Position',[0 0.05 0.01 0.90])
    imagesc(X{1,1}(:,:,1),[X_min X_max])
    axis off
    subplot(1,4,2,'Position',[0.02 0.05 0.25 0.90])
    ypred = predict(net,X{1,1});
    imagesc(ypred,[X_min X_max])
    axis off
    subplot(1,4,3,'Position',[0.28 0.05 0.28 0.90])
    imagesc(X{1,2},[X_min X_max])
    axis off
    colorbar
    ax = gca; 
    ax.FontSize = 60; 
    subplot(1,4,4,'Position',[0.65 0.05 0.23 0.90])
    z = accuracyCriterion(X{1,2}, ypred,0.5 ,2 ,0.10 * X_max);
    imagesc(z)
    colormap hot
    axis off
    F = getframe(fig1); 
    images{1,i} = F.cdata; 
end
hold off
close all
montage(images)
% title('Model predictions on test samples')
end

%% Plot standard basis 
% display line_num examples in montage
bias = predict(net,zeros(sensorNum,1));
if display
fig1 = figure();
sgtitle("Model predictions for zero - bias")
imagesc(bias)
colorbar
pause(3)
SB = eye(sensorNum);
hold on
for i=1:sensorNum
    sgtitle(sprintf("Model predictions on standartd basis %d",i))
    subplot(1,3,1)
    imagesc(sensorLocations(i,sensors),[0 2])
    colorbar
    colormap hot
    v = SB(:,i);
    subplot(1,3,2)
    imagesc(v)
    ypred = predict(net,v)-bias;
    subplot(1,3,3)
    imagesc(ypred)
    colorbar
    pause(5)
end
end

%%
if display
figure
imagesc(net.Layers(2,1).Weights')
colorbar

figure
imagesc(net.Layers(2,1).Bias')
colorbar
end

%% Plot SVD decomposition
% display line_num examples in montage
if perform_SVD
W = net.Layers(2,1).Weights;
[U,S,V] = svd(W);

if display
fig1 = figure();
imagesc(bias)
colorbar
pause(3)
sgtitle("Model predictions for zero - bias")
hold on
for i=1:sensorNum
    sgtitle(sprintf("Model predictions on orthogonal basis %d",i));
    v = V(:,i);
    subplot(1,2,1)
    imagesc(v)
    colorbar
    colormap hot
    ypred = predict(net,v)-bias;
    subplot(1,2,2)
    imagesc(ypred)
    colorbar
    pause(3)
end
end

%% Analyze matrix U
if display
fig2 = figure();
sgtitle("Model predictions for zero - bias")

imagesc(bias)
colorbar
pause(3)

W = net.Layers(2,1).Weights;
[U,S,V] = svd(W);
hold on
for i=1:sensorNum
    sgtitle("Orthogonal basis of matrix U");
    u = reshape(U(:,i),[100,100])';
    subplot(1,2,1)
    imagesc(u)
    title(sprintf("the - %d  vector in U",i))
    colorbar
    colormap hot
    subplot(1,2,2)
    plot((0:i),[0; diag(S(1:i,1:i))])
    title(sprintf("Singular value is %d",S(i,i)))
    pause(4)
end
figure
plot(diag(S))
end
end

%%
toc
beep
pause(1)
beep
pause(1)
beep

%% suporting functions
function p = make_read_func(sensorNum,epsilon)
p = @read_func;

   function [S] = read_func(f)
    file = load(f);
    V = file.X(sub2ind(size(file.X), file.sensors(:,1), file.sensors(:,2)));
    V = round(V + epsilon*randn(1,sensorNum).*V);
    S = [{V(1:sensorNum)'} {file.X}];
   end
end

function [z] = accuracyCriterion(x,y, lowerBound, upperBound, Threshold)
    z = (x./y <= upperBound & x./y >= lowerBound) | (x <= Threshold & y <= Threshold);
end

function [corr] = mat_corr(A,B)
    A = A-mean(A,"all");
    B = B-mean(B,"all");
    corr = trace(A'*B)/(norm(A,"fro")*norm(B,"fro")+1e-12);
end
