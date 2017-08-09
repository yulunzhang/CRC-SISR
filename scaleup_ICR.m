function [imgs] = scaleup_ICR(conf, imgs, low)
border_temp = conf.border;
conf.border = [0 0]; % border of the image (to ignore)
imgs = scaleup_ICR_one(conf, imgs, low);
conf.border = border_temp;
fprintf('\n');
end
