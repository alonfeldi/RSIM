%% 
sensor_num = 30;
sensor_map = zeros(100,100);

reset(testData)
N = numel(testData.Files);
corr_list = zeros(N,1);
rmse_list = zeros(N,1);
i = 1;
[Xq,Yq] = meshgrid(1:100);
Xs = sensors(:,1);
Ys = sensors(:,2);

while hasdata(testData)
    X = read(testData);
    X_min = min(X{1,2}(:));
    X_max = max(X{1,2}(:))+1e-16;
    F = scatteredInterpolant(Xs,Ys,X{1,1});
    F.Method = 'natural';
    F.ExtrapolationMethod = 'linear';
    ypred = F(Xq,Yq)';
    % present
    for i=1:sensor_num
        sensor_map(Xs(i),Ys(i)) = X{1,1}(i);
    end
    subplot(1,3,1)
    imagesc(sensor_map,[0 X_max])
    colorbar
    title("input - sensors")
    subplot(1,3,2)
    imagesc(ypred,[0 X_max])
    colorbar
    corr = mat_corr(X{1,2},ypred);
    title(sprintf("prediction - correlation = %.3f",corr))
    subplot(1,3,3)
    imagesc(X{1,2},[0 X_max])
    colorbar
    title("true map")
    pause(3)
    corr_list(i) = mat_corr(X{1,2},ypred);
    rmse_list(i) = rmse(X{1,2}, ypred,"all");
    i = i + 1;
end

meanCorr = mean(corr_list)
meanRMSE = mean(rmse_list)

%%




%%
function [corr] = mat_corr(A,B)
    % this function returns the cosine angle calculated by the 
    % Frobenius inner product of matricies A and B devided by their
    % Forbenius norm.
    A = A-mean(A,"all");
    B = B-mean(B,"all");
    corr = trace(A'*B)/(norm(A,"fro")*norm(B,"fro")+1e-12);
end
