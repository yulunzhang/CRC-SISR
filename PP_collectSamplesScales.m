function [lores hires] = PP_collectSamplesScales(conf, midres, ohires, numscales, scalefactor)

lores = [];
hires = [];

for scale = 1:numscales
    sfactor = scalefactor^(scale-1);
    % 
    cmidres = resize(midres, sfactor, 'bicubic');
    chires = resize(ohires, sfactor, 'bicubic');
    % 
    cmidres = modcrop(cmidres, conf.scale); % crop a bit (to simplify scaling issues)
    chires = modcrop(chires, conf.scale); % crop a bit (to simplify scaling issues)
    
    features = collect(conf, cmidres, conf.upsample_factor, conf.filters);
    clear cmidres
    % Scale down images
    clores = resize(chires, 1/conf.scale, conf.interpolate_kernel);
    interpolated = resize(clores, conf.scale, conf.interpolate_kernel);
    clear clores
    patches = cell(size(chires));
    for i = 1:numel(patches) % Remove low frequencies
        patches{i} = chires{i} - interpolated{i};
    end
    clear chires interpolated

    hires = [hires collect(conf, patches, conf.scale, {})];
    
    lores = [lores conf.V_pca' * features];
end