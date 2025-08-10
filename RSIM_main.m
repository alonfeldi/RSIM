
function RSIM_main()
% RSIM_main — End-to-end training & eval on synthetic .mat files (X, sensors)
% Requires: Deep Learning Toolbox
% Data folder must contain .mat files with fields: X (HxW), sensors (s x 2, 1-based)

%% ------- User params -------
p.display       = true;      % figures & montages
p.save_net      = false;     % save trained net
p.perform_SVD   = false;     % SVD analysis of FC weights
p.sensorNum     = 30;        % number of sensors to sample per map
p.epsilon       = 0.10;      % multiplicative noise factor (std of N(0,epsilon))
p.valFrac       = 0.10;      % validation split
p.testFrac      = 0.20;      % test split
p.rngSeed       = 0;         % for reproducibility
p.execEnv       = "auto";    % "auto" | "gpu" | "cpu"

%% ------- Pick data folder (or set path here) -------
dataFolder = uigetdir(pwd,'Select folder with .mat files (fields: X, sensors)');
if dataFolder==0, error('No folder selected.'); end

%% ------- Peek first file to infer grid size -------
list = dir(fullfile(dataFolder,'*.mat'));
if isempty(list), error('No .mat files found in %s', dataFolder); end
tmp = load(fullfile(list(1).folder, list(1).name));
assert(isfield(tmp,'X') && isfield(tmp,'sensors'), 'Each .mat must have X and sensors.');
[H,W] = size(tmp.X);
P = H*W;

%% ------- Build datastore -------
readFcn = make_read_func(H,W,p.sensorNum,p.epsilon);
ds = fileDatastore(dataFolder, "IncludeSubfolders", true, ...
    "FileExtensions", ".mat", "ReadFcn", readFcn);

N = numel(ds.Files);
if N < 10
    warning('Only %d files found — training may be unstable.', N);
end

%% ------- Split data -------
rng(p.rngSeed);
idx = randperm(N);
nVal  = max(1, round(p.valFrac  * N));
nTest = max(1, round(p.testFrac * N));
valId  = idx(1:nVal);
testId = idx(nVal+1 : nVal+nTest);
trainId = idx(nVal+nTest+1 : end);

trainData = subset(ds, trainId);
valData   = subset(ds, valId);
testData  = subset(ds, testId);

%% ------- Optional: quick visual sanity check -------
if p.display
    reset(testData);
    f = figure('Position',[100 100 1400 900]);
    t = tiledlayout(f,3,3,'Padding','compact','TileSpacing','compact');
    title(t,'Random ground-truth maps','FontSize',14);
    for k=1:min(9,numel(testData.Files))
        Xpair = read(testData);
        nexttile; imagesc(Xpair{1,2}); axis image off; colormap hot; colorbar;
    end
    drawnow; reset(testData);
end

%% ------- Define model (linear layer + depthToSpace) -------
layers = [
    imageInputLayer([p.sensorNum 1 1], "Name","input", "Normalization","none")
    fullyConnectedLayer(P, "Name","fc", "BiasLearnRateFactor",1, "Bias",zeros(P,1))
    depthToSpace2dLayer([H W], "Name","depthToSpace", "Mode","crd")
    regressionLayer("Name","regression")
];

%% ------- Training options -------
options = trainingOptions("adam", ...
    MaxEpochs=10, ...
    MiniBatchSize=128, ...
    ValidationData=valData, ...
    ValidationPatience=5, ...
    Shuffle="every-epoch", ...
    LearnRateSchedule="piecewise", ...
    InitialLearnRate=0.1, ...
    LearnRateDropPeriod=1, ...
    LearnRateDropFactor=0.1, ...
    L2Regularization=5, ...
    Plots="none", ...
    Verbose=false, ...
    ExecutionEnvironment=p.execEnv, ...
    OutputNetwork="best-validation-loss");

%% ------- Train -------
net = trainNetwork(trainData, layers, options);
if p.save_net
    [fName, fPath] = uiputfile('trained_RSIM_net.mat','Save trained network as');
    if ischar(fName)
        save(fullfile(fPath,fName), "net", "p", "-v7.3");
    end
end

%% ------- Evaluate -------
reset(testData);
numT = numel(testData.Files);
corr_list = zeros(numT,1);
rmse_list = zeros(numT,1);

i = 1;
while hasdata(testData)
    S = read(testData);              % S{1} = z (sensor vector) ; S{2} = X (HxW)
    ypred = predict(net, S{1});
    corr_list(i) = mat_corr(S{2}, ypred);
    rmse_list(i) = rmse_local(S{2}, ypred);
    i = i+1;
end

fprintf('\n=== Summary ===\n');
fprintf('Mean Corr: %.3f | Mean RMSE: %.3f\n', mean(corr_list), mean(rmse_list));

%% ------- Qualitative panel -------
if p.display
    reset(testData);
    K = min(8, numel(testData.Files));
    f = figure('Position',[100 100 1400 720]); t = tiledlayout(f,K,4,'Padding','compact','TileSpacing','compact');
    title(t, 'Test samples: sensors → prediction → ground truth → binary accuracy','FontSize',14);
    for k=1:K
        S = read(testData);
        Xtrue = S{2};
        z     = S{1};

        Xmin = min(Xtrue(:)); Xmax = max(Xtrue(:))+1e-16;
        ypred = predict(net, z);

        nexttile; imagesc(reshape(z, [numel(z) 1])); axis tight off; colorbar; title('sensors');
        nexttile; imagesc(ypred, [Xmin Xmax]); axis image off; colorbar; title('prediction');
        nexttile; imagesc(Xtrue, [Xmin Xmax]); axis image off; colorbar; title('ground truth');

        Zbin = accuracyCriterion(Xtrue, ypred, 0.5, 2, 0.10*Xmax);
        acc  = 100*sum(Zbin,'all')/numel(Zbin);
        nexttile; imagesc(Zbin); axis image off; colorbar; title(sprintf('binary acc %.1f%%',acc));
        colormap hot
    end
end

%% ------- Optional: SVD analysis -------
if p.perform_SVD
    W = net.Layers(2,1).Weights;      % (P x s)
    [U,S,V] = svd(W, 'econ');
    figure; plot(diag(S),'.-'); grid on; title('Singular values of FC weights'); xlabel('i'); ylabel('\sigma_i');

    r = min(8, size(U,2));
    f = figure('Position',[100 100 1200 600]);
    t = tiledlayout(f,2, r/2, 'Padding','compact','TileSpacing','compact');
    title(t,'Top left singular vectors (reshaped)','FontSize',14);
    for i=1:r
        Ui = reshape(U(:,i),[H W]);
        nexttile; imagesc(Ui); axis image off; colorbar; colormap hot; title(sprintf('U_%d',i));
    end
end
end

%% ---------- helpers ----------
function p = make_read_func(H,W,sensorNum,epsilon)
% returns a ReadFcn for fileDatastore: file → {z, X}
p = @read_func;
    function S = read_func(fpath)
        file = load(fpath);
        X = file.X;
        assert(isequal(size(X),[H W]), 'Unexpected X size in %s', fpath);

        Sij = file.sensors;
        sAvail = size(Sij,1);
        if sensorNum > sAvail
            error('Requested sensorNum=%d > available=%d in %s', sensorNum, sAvail, fpath);
        end
        Sij = Sij(1:sensorNum, :);

        linIdx = sub2ind([H W], Sij(:,1), Sij(:,2));
        z = X(linIdx);

        z = round( z .* (1 + epsilon*randn(size(z))) );

        S = { reshape(z, [sensorNum 1 1]), X };
    end
end

function z = accuracyCriterion(x,y, lowerBound, upperBound, Threshold)
    ratio = x./(y + 1e-12);
    z = (ratio <= upperBound & ratio >= lowerBound) | (x <= Threshold & y <= Threshold);
end

function c = mat_corr(A,B)
    A = A - mean(A,"all"); B = B - mean(B,"all");
    c = sum(A(:).*B(:)) / (norm(A(:))*norm(B(:)) + 1e-12);
end

function r = rmse_local(A,B)
    D = A - B;
    r = sqrt(mean(D(:).^2));
end
