function result = PP_glob(directory, method, upscale, pattern)
folder_data = fileparts(directory);
fid = fopen(fullfile(folder_data, 'TrainFileNameList.txt'), 'r');
filelist_temp = textscan(fid, '%s\n');
filelist = filelist_temp{1};
fclose(fid);
num_im = numel(filelist);
result = cell(num_im, 1);
for idx_im = 1:num_im
    result{idx_im} = fullfile(directory, [filelist{idx_im} '_' method '_x' num2str(upscale) pattern]);
end

end
