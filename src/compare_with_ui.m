function compare_with_ui()
%COMPARE_WITH_UI One-stop UI to compare interpolation methods (visual + numeric).
%   Loads .mat files with fields: X (HxW), sensors (s x 2, 1-based).
%   Optional RSIM (provide MAT with variable `net`).
%   Choose methods, metrics, whether to show figures, and how many panels to save.
%   Uses a single figure window (overwrites between samples).
%
%   Author: Alon Feldman (UI/refactor by ChatGPT)

%% -------------------- UI: Data folder --------------------
dataFolder = uigetdir(pwd, 'Select folder with .mat files (must contain X, sensors)');
if dataFolder==0, disp('Canceled.'); return; end

% Peek first file to infer grid and sensor list
files = dir(fullfile(dataFolder,'*.mat'));
if isempty(files), error('No .mat files in %s', dataFolder); end
probe = load(fullfile(files(1).folder, files(1).name));
assert(isfield(probe,'X') && isfield(probe,'sensors'), 'Each MAT must have X and sensors.');

H = size(probe.X,1); W = size(probe.X,2);
maxSensors = size(probe.sensors,1);

%% -------------------- UI: Optional RSIM net --------------------
rs = questdlg('Include RSIM (load a trained net)?','RSIM','Yes','No','No');
useRSIM = strcmp(rs,'Yes');
net = [];
if useRSIM
    [nf, np] = uigetfile({'*.mat','MAT-files (*.mat)'}, 'Select MAT file containing variable "net"');
    if isequal(nf,0), useRSIM = false; else
        S = load(fullfile(np,nf));
        if isfield(S,'net'), net = S.net; else
            warndlg('Selected MAT does not contain variable "net". RSIM will be disabled.','RSIM');
            useRSIM = false;
        end
    end
end

%% -------------------- UI: Methods to compare --------------------
displayNames = {'Linear','Natural (NN)','Nearest','IDW','Universal Kriging','Ordinary Kriging'};
if useRSIM, displayNames{end+1} = 'RSIM'; end

defaults = true(1, numel(displayNames));

[idx, ok] = listdlg('PromptString','Select methods to compare:', ...
    'ListString', displayNames, 'InitialValue', find(defaults), ...
    'ListSize',[320 190], 'SelectionMode','multiple');
if ~ok || isempty(idx), disp('Canceled.'); return; end
methods = displayNames(idx);

%% -------------------- UI: Numeric + Visual + Save --------------------
answ = inputdlg( ...
    {'SensorNum (<= available)', 'Noise epsilon (e.g., 0.1)', 'Compute metrics? (true/false)'}, ...
    'Params', 1, {num2str(min(30,maxSensors)),'0.1','true'});
if isempty(answ), disp('Canceled.'); return; end

sensorNum = min(str2double(answ{1}), maxSensors);
epsilon   = str2double(answ{2});
doMetrics = any(strcmpi(answ{3},{'true','1','yes','y'}));

sf = questdlg('Show visual panels during run?','Visualization','Yes','No','Yes');
showFigures = strcmp(sf,'Yes');

saveQ = inputdlg({'How many panels to save? (0 = none)'}, 'Saving', 1, {'0'});
if isempty(saveQ), disp('Canceled.'); return; end
saveCount = max(0, round(str2double(saveQ{1})));
saveDir = '';
if saveCount > 0
    saveDir = uigetdir(dataFolder, 'Select output folder for saved panels/summary');
    if saveDir==0, saveCount = 0; end
end

%% -------------------- Build datastore --------------------
readFcn = make_read_func(sensorNum, epsilon);
ds = fileDatastore(dataFolder, "IncludeSubfolders", true, ...
    "FileExtensions", ".mat", "ReadFcn", readFcn);
N = numel(ds.Files);
if N==0, error('No .mat files found.'); end

saveIdx = [];
if saveCount > 0
    saveCount = min(saveCount, N);
    saveIdx = unique(round(linspace(1, N, saveCount)));
end

Sij = probe.sensors(1:sensorNum,:);
Ys  = Sij(:,1);  
Xs  = Sij(:,2);  
[Xq, Yq] = meshgrid(1:W, 1:H);
[X1i, X2i] = meshgrid(1:W, 1:H);

%% -------------------- Metrics containers --------------------
M = numel(methods);
RMSE = nan(N, M);
CORR = nan(N, M);

if showFigures
    fig = figure('Color','w','Position',[100 100 1100 1400]);
end

%% -------------------- Main loop --------------------
reset(ds);
for i = 1:N
    S = read(ds);     
    zvec  = S{1}(:);
    Xtrue = S{2};

    Xmin = min(Xtrue(:));
    Xmax = max(Xtrue(:))+1e-16;

    if showFigures
        clf; 
        rows = ceil((M+1)/2); cols = 2;
        t = tiledlayout(rows, cols, 'TileSpacing','tight','Padding','tight'); 
        colormap hot
    end

    for m = 1:M
        label = methods{m};
        Y = nan(H,W);   

        switch label
            case 'Linear'
                F = scatteredInterpolant(Xs, Ys, zvec, 'linear','linear');
                Y = F(Xq, Yq);
            case 'Natural (NN)'
                F = scatteredInterpolant(Xs, Ys, zvec, 'natural','linear');
                Y = F(Xq, Yq);
            case 'Nearest'
                F = scatteredInterpolant(Xs, Ys, zvec, 'nearest','nearest');
                Y = F(Xq, Yq);
            case 'IDW'
                Fint = idw([Xs Ys], zvec, [X1i(:), X2i(:)]);
                Y = reshape(Fint, H, W);
            case 'Universal Kriging'
                try
                    Y = UniversalKriging_PyKrige(Xs, Ys, zvec, 1:W, 1:H);
                catch
                    Y = nan(H,W);
                end
            case 'Ordinary Kriging'
                try
                    Y = OrdinaryKriging_PyKrige(Xs, Ys, zvec, 1:W, 1:H);
                catch
                    Y = nan(H,W);
                end
            case 'RSIM'
                if useRSIM && ~isempty(net)
                    Y = predict(net, reshape(zvec,[numel(zvec) 1 1]));
                else
                    Y = nan(H,W);
                end
        end

        if doMetrics
            CORR(i,m) = mat_corr(Xtrue, Y);
            RMSE(i,m) = rmse_local(Xtrue, Y);
        end

        if showFigures
            nexttile; imagesc(Y, [Xmin Xmax]); hold on
            plot(Xs, Ys, 'g.', 'MarkerSize', 6); axis image off
            if doMetrics
                if all(isnan(Y),'all')
                    title(sprintf('%s (N/A)', label));
                else
                    title(sprintf('%s  r=%.2f  RMSE=%.1f', label, CORR(i,m), RMSE(i,m)));
                end
            else
                title(label);
            end
        end
    end

    if showFigures
        nexttile; imagesc(Xtrue, [Xmin Xmax]); axis image off; title('Ground truth');
        drawnow;
        if ismember(i, saveIdx) && saveCount>0
            exportgraphics(gcf, fullfile(saveDir, sprintf('panel_%03d.png', i)), 'Resolution', 200);
        end
    end
end

%% -------------------- Summary metrics --------------------
if doMetrics
    meanRMSE = mean(RMSE, 1, 'omitnan');
    meanCORR = mean(CORR,  1, 'omitnan');

    T = table(methods(:), meanRMSE(:), meanCORR(:), ...
        'VariableNames', {'Method','Mean_RMSE','Mean_Corr'});
    disp('=== Summary (means across files) ==='); disp(T);

    if saveCount>0
        writetable(T, fullfile(saveDir, 'summary_metrics.csv'));
    end
end

end % ====== compare_with_ui ======


%% ================= Helpers =================
function p = make_read_func(sensorNum,epsilon)
p = @read_func;
    function S = read_func(f)
        file = load(f);
        assert(isfield(file,'X') && isfield(file,'sensors'), 'MAT must have X and sensors');
        [H,W] = size(file.X);
        linIdx = sub2ind([H W], file.sensors(:,1), file.sensors(:,2));
        V = file.X(linIdx);
        V = round(V(1:sensorNum) .* (1 + epsilon*randn(sensorNum,1)));
        S = {reshape(V,[sensorNum 1 1])  file.X};
    end
end

function c = mat_corr(A,B)
    if all(isnan(B),'all'), c = NaN; return; end
    A = A - mean(A,"all");
    B = B - mean(B,"all");
    c = sum(A(:).*B(:)) / (norm(A(:))*norm(B(:)) + 1e-12);
end

function r = rmse_local(A,B)
    if all(isnan(B),'all'), r = NaN; return; end
    D = A - B;
    r = sqrt(mean(D(:).^2));
end

function Fint = idw(X0,F0,Xint,p,rad,L)
    if nargin < 6, L = 2; end
    if nargin < 5, rad = inf; end
    if nargin < 4, p = 2; end
    Q = size(Xint,1);
    Fint = zeros(Q,1);
    for ip = 1:Q
        dx = abs(X0(:,1) - Xint(ip,1)).^L;
        dy = abs(X0(:,2) - Xint(ip,2)).^L;
        D  = (dx + dy).^(1/L);
        D(D==0) = eps; D(D>rad) = inf;
        W = 1 ./ (D.^p);
        Fint(ip) = sum(W.*F0) / sum(W);
    end
end

function Z = UniversalKriging_PyKrige(x,y,z,Xq,Yq)
    if max(z)-min(z)==0, z(1) = z(1) + 1e-12; end
    Z = pyrun(["from pykrige.uk import UniversalKriging", ...
               "UK = UniversalKriging(x,y,z,variogram_model='linear',drift_terms=['regional_linear'])", ...
               "zgrid, ss = UK.execute('grid', Xq, Yq)"], ...
               "zgrid", x=x, y=y, z=z, Xq=Xq, Yq=Yq);
    Z = double(Z);
end

function Z = OrdinaryKriging_PyKrige(x,y,z,Xq,Yq)
    if max(z)-min(z)==0, z(1) = z(1) + 1e-12; end
    Z = pyrun(["from pykrige.ok import OrdinaryKriging", ...
               "OK = OrdinaryKriging(x,y,z,variogram_model='linear',verbose=False,enable_plotting=False)", ...
               "zgrid, ss = OK.execute('grid', Xq, Yq)"], ...
               "zgrid", x=x, y=y, z=z, Xq=Xq, Yq=Yq);
    Z = double(Z);
end
