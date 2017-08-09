function [psnr] = PP_PSNR(gnd, test)
if size(gnd, 3) == 3,
    gnd = rgb2ycbcr(gnd);
    gnd = gnd(:, :, 1);
end

if size(test, 3) == 3,
    test = rgb2ycbcr(test);
    test = test(:, :, 1);
end

imdff = double(gnd) - double(test);
imdff = imdff(:);

rmse = sqrt(mean(imdff.^2));

psnr = 20*log10(255/rmse);

end