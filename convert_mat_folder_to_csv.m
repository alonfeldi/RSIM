
function convert_mat_folder_to_csv(srcFolder, outFolder)
% Exports X to map_*.csv and sensors to sensors.csv per file
if nargin<1 || ~isfolder(srcFolder)
    srcFolder = uigetdir(pwd,'Select folder with .mat files (X, sensors)');
    if srcFolder==0, return; end
end
if nargin<2
    outFolder = uigetdir(pwd,'Select output folder for CSV');
    if outFolder==0, return; end
end
files = dir(fullfile(srcFolder,'*.mat'));
if isempty(files), error('No .mat files in %s', srcFolder); end

for k=1:numel(files)
    S = load(fullfile(files(k).folder, files(k).name));
    assert(isfield(S,'X') && isfield(S,'sensors'));
    writematrix(S.X, fullfile(outFolder, sprintf('map_%d.csv',k)));
    if k==1
        writematrix(S.sensors, fullfile(outFolder, 'sensors.csv')); % once
    end
end
fprintf('Exported %d maps to %s (plus sensors.csv)\n', numel(files), outFolder);
end
