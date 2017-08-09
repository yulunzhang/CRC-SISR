function res = calc_PSNR(f, g)
gnd = imread(f); % original
test = imread(g); % distorted
rmse = PP_RMSE(gnd, test);
res = 20*log10(255/rmse);

end