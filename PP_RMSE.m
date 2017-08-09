function [rmse] = PP_RMSE(gnd, test)
if size(gnd, 3) == 3,
    gnd = rgb2ycbcr(gnd);
    gnd = gnd(:, :, 1);
end

if size(test, 3) == 3,
    test = rgb2ycbcr(test);
    test = test(:, :, 1);
end

% indx = (gnd == 0);
% gnd(indx) = [];
% test(indx) = [];

imdff = double(gnd) - double(test);
imdff = imdff(:);

rmse = sqrt(mean(imdff.^2));
end