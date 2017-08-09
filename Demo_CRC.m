% Collaborative Representation Cascade for Single-Image Super-Resolution
% Example code
% This code is built on the example code of "Anchored Neighborhood
% Regression for Fast Example-Based Super-Resolution".
% Please reference to papers:
% [1] Radu Timofte, Vincent De Smet, Luc Van Gool.
% Anchored Neighborhood Regression for Fast Example-Based Super-Resolution.
% International Conference on Computer Vision (ICCV), 2013. 
%
% [2] Radu Timofte, Vincent De Smet, Luc Van Gool.
% A+: Adjusted Anchored Neighborhood Regression for Fast Super-Resolution.
% Asian Conference on Computer Vision (ACCV), 2014. 
%
% [3] Yongbing Zhang, Yulun Zhang, Jian Zhang, Dong Xu, Yun Fu, Yisen Wang, Xiangyang Ji, Qionghai Dai 
% Collaborative Representation Cascade for Single-Image Super-Resolution
% IEEE Transactions on Systems, Man, and Cybernetics: Systems (TSMC), vol. PP, no. 99, pp. 1-16, 2017.
%
% [4] Yulun Zhang, Kaiyu Gu, Yongbing Zhang, Jian Zhang, Qionghai Dai 
% Image Super-Resolution based on Dictionary Learning and Anchored Neighborhood Regression with Mutual Inconherence
% IEEE International Conference on Image Processing (ICIP2015), Quebec, Canada, Sep. 2015.
%
% [5] Yulun Zhang, Yongbing Zhang, Jian Zhang, Haoqian Wang, Qionghai Dai 
% Single Image Super-Resolution via Iterative Collaborative Representation
% Paci?c-Rim Conference on Multimedia (PCM2015), Gwangju, Korea, Sep. 2015.
%
% For any questions, email me by yulun100@gmail.com
%

clear all; close all; clc  
warning off all
    
p = pwd;
addpath(fullfile(p, '/methods'));  % the upscaling methods
addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm
addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm

imgscale = 1; % the scale reference we work with
flag = 0;       % flag = 0 - only GR, ANR, A+, ICR, CRC-1, CRC-2, and bicubic methods, the other get the bicubic result by default
                % flag = 1 - all the methods are applied

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Here, we can change upscaling factors (x2/3/4), test sets, and image
%%% patterns
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
upscaling = 2; % the magnification factor x2, x3, x4...
input_dir = 'Set10Dong'; % Set5, Set14, Set10Dong, SetSRF
pattern = '*.bmp'; % Pattern to process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dict_sizes = [2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536];
neighbors = [1:1:12, 16:4:32, 40:8:64, 80:16:128, 256, 512, 1024];

clusterszA = 2048; % neighborhood size for A+
clusterszApp = 2048; % neighborhood size for A++

lam1_miksvd = 0.09;
llambda_App = 0.1; 

disp('The experiment corresponds to the results from Table 2 in the referenced [1] and [2] papers.');

disp(['The experiment uses ' input_dir ' dataset and aims at a magnification of factor x' num2str(upscaling) '.']);
if flag==1
    disp('All methods are employed : Bicubic, Zeyde et al., GR, ANR, NE+LS, NE+NNLS, NE+LLE, A+, ICR, CRC-1, CRC-2.');    
else
    disp('We run only for Bicubic, GR, ANR and A+ methods, the other get the Bicubic result by default.');
end

fprintf('\n\n');

for d=10    %1024
    %% Train dictionaries
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%       Train dictionaries        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tag = [input_dir '_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms'];
    
    disp(['Upscaling x' num2str(upscaling) ' ' input_dir ' with Zeyde dictionary of size = ' num2str(dict_sizes(d))]);
    
    mat_file = ['conf_Zeyde_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];    
    
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf');
    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using Zeyde approach...']);
        % Simulation settings
        conf.scale = upscaling; % scale-up factor
        conf.level = 1; % # of scale-ups to perform
        conf.window = [3 3]; % low-res. window size
        conf.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf.filters = {G, G.', L, L.'}; % 2D versions
        conf.interpolate_kernel = 'bicubic';

        conf.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf.overlap = [1 1]; % partial overlap (for faster training)
        end
        
        startt = tic;
        conf = learn_dict(conf, load_images(...            
            glob('CVPR08-SR/Data/Training', '*.bmp') ...
            ), dict_sizes(d));       
        conf.overlap = conf.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf.trainingtime = toc(startt);
        toc(startt)
        
        save(mat_file, 'conf');                       
        
        % train call        
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%  Dictionary learning via MIKSVD  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    mat_file = ['conf_MIKSVD_' num2str(dict_sizes(d)) '_x' num2str(upscaling) '_lam1_' num2str(lam1_miksvd*1000)];
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf_MIKSVD');
    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using MI-KSVD approach...']);
        
        % Simulation settings
        conf_MIKSVD.scale = upscaling; % scale-up factor
        conf_MIKSVD.level = 1; % # of scale-ups to perform
        conf_MIKSVD.window = [3 3]; % low-res. window size
        conf_MIKSVD.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf_MIKSVD.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf_MIKSVD.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf_MIKSVD.filters = {G, G.', L, L.'}; % 2D versions
        conf_MIKSVD.interpolate_kernel = 'bicubic';

        conf_MIKSVD.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf_MIKSVD.overlap = [1 1]; % partial overlap (for faster training)
        end
        
        startt = tic;
        conf_MIKSVD = learn_dict_MIKSVD(conf_MIKSVD, load_images(glob('CVPR08-SR/Data/Training', '*.bmp')), dict_sizes(d), lam1_miksvd);       
        conf_MIKSVD.overlap = conf_MIKSVD.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf_MIKSVD.trainingtime = toc(startt);
        toc(startt)
        
        save(mat_file, 'conf_MIKSVD');                       
        
        % train call        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% train dictionary in the Aplus feature space, first and second order deviation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mat_file = ['conf_ICR_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];    
    
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf_ICR');
    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using Zeyde approach...']);
        % Simulation settings
        conf_ICR.scale = upscaling; % scale-up factor
        conf_ICR.level = 1; % # of scale-ups to perform
        conf_ICR.window = [3 3]; % low-res. window size
        conf_ICR.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf_ICR.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf_ICR.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf_ICR.filters = {G, G.', L, L.'}; % 2D versions
        conf_ICR.interpolate_kernel = 'bicubic';

        conf_ICR.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf_ICR.overlap = [1 1]; % partial overlap (for faster training)
        end
        
        startt = tic;
        % 
        conf_ICR = PP_learn_dict(conf_ICR, load_images(PP_glob('CVPR08-SR/Data/Training_91_Mid', 'Aplus', upscaling, '.bmp')), load_images(PP_glob('CVPR08-SR/Data/Training_91_HR', 'Gnd', upscaling, '.bmp')), dict_sizes(d));       
        conf_ICR.overlap = conf_ICR.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf_ICR.trainingtime = toc(startt);
        toc(startt)
        
        save(mat_file, 'conf_ICR');                       
        
        % train call        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% train dictionary in the App feature space, first and second order deviation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mat_file = ['conf_ICR_App_1_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];    
    
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf_ICR_App_1');
    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using Zeyde approach...']);
        % Simulation settings
        conf_ICR_App_1.scale = upscaling; % scale-up factor
        conf_ICR_App_1.level = 1; % # of scale-ups to perform
        conf_ICR_App_1.window = [3 3]; % low-res. window size
        conf_ICR_App_1.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf_ICR_App_1.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf_ICR_App_1.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf_ICR_App_1.filters = {G, G.', L, L.'}; % 2D versions
        conf_ICR_App_1.interpolate_kernel = 'bicubic';

        conf_ICR_App_1.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf_ICR_App_1.overlap = [1 1]; % partial overlap (for faster training)
        end
        
        startt = tic;
        % 
        conf_ICR_App_1 = PP_learn_dict_MIKSVD(conf_ICR_App_1, load_images(PP_glob('CVPR08-SR/Data/Training_91_Mid', 'App', upscaling, '.bmp')), load_images(PP_glob('CVPR08-SR/Data/Training_91_HR', 'Gnd', upscaling, '.bmp')), dict_sizes(d), lam1_miksvd);       
        conf_ICR_App_1.overlap = conf_ICR_App_1.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf_ICR_App_1.trainingtime = toc(startt);
        toc(startt)
        
        save(mat_file, 'conf_ICR_App_1');                       
        
        % train call        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% train dictionary in the ICR_App_1 feature space, first and second order deviation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mat_file = ['conf_ICR_App_2_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];    
    
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf_ICR_App_2');
    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using MIKSVD...']);
        % Simulation settings
        conf_ICR_App_2.scale = upscaling; % scale-up factor
        conf_ICR_App_2.level = 1; % # of scale-ups to perform
        conf_ICR_App_2.window = [3 3]; % low-res. window size
        conf_ICR_App_2.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf_ICR_App_2.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf_ICR_App_2.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf_ICR_App_2.filters = {G, G.', L, L.'}; % 2D versions
        conf_ICR_App_2.interpolate_kernel = 'bicubic';

        conf_ICR_App_2.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf_ICR_App_2.overlap = [1 1]; % partial overlap (for faster training)
        end
        
        startt = tic;
        % 
        conf_ICR_App_2 = PP_learn_dict_MIKSVD(conf_ICR_App_2, load_images(PP_glob('CVPR08-SR/Data/Training_91_Mid', 'ICR_App_1', upscaling, '.bmp')), load_images(PP_glob('CVPR08-SR/Data/Training_91_HR', 'Gnd', upscaling, '.bmp')), dict_sizes(d), lam1_miksvd);       
        conf_ICR_App_2.overlap = conf_ICR_App_2.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf_ICR_App_2.trainingtime = toc(startt);
        toc(startt)
        
        save(mat_file, 'conf_ICR_App_2');                       
        
        % train call        
    end

    
    %%  Compute Regressors               
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%  Compute Regressors                             %%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if dict_sizes(d) < 1024
        lambda = 0.01;
    elseif dict_sizes(d) < 2048
        lambda = 0.1;
    elseif dict_sizes(d) < 8192
        lambda = 1;
    else
        lambda = 5;
    end
    
    %% GR
    if dict_sizes(d) < 10000
        conf.ProjM = inv(conf.dict_lores'*conf.dict_lores+lambda*eye(size(conf.dict_lores,2)))*conf.dict_lores';    
        conf.PP = (1+lambda)*conf.dict_hires*conf.ProjM;
    else
        % here should be an approximation
        conf.PP = zeros(size(conf.dict_hires,1), size(conf.V_pca,2));
        conf.ProjM = [];
    end
    
    conf.filenames = glob(input_dir, pattern); % Cell array 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% A list of SR methods    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    conf.desc = {'Gnd', 'Bicubic', 'Zeyde', 'GR', 'ANR', 'NE_LS','NE_NNLS','NE_LLE', 'Aplus',  'ICR', 'CRC-1', 'CRC-2'};
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    conf.results = {};
    
    %conf.points = [1:10:size(conf.dict_lores,2)];
    conf.points = [1:1:size(conf.dict_lores,2)];
    
    conf.pointslo = conf.dict_lores(:,conf.points);
    conf.pointsloPCA = conf.pointslo'*conf.V_pca';
    
    % precompute for ANR the anchored neighborhoods and the projection matrices for
    % the dictionary 
    
    conf.PPs = [];    
    if  size(conf.dict_lores,2) < 40
        clustersz = size(conf.dict_lores,2);
    else
        clustersz = 40;
    end
    D = abs(conf.pointslo'*conf.dict_lores);    
    
    for i = 1:length(conf.points)
        [vals idx] = sort(D(i,:), 'descend');
        if (clustersz >= size(conf.dict_lores,2)/2)
            conf.PPs{i} = conf.PP;
        else
            Lo = conf.dict_lores(:, idx(1:clustersz));        
            conf.PPs{i} = 1.01*conf.dict_hires(:,idx(1:clustersz))*inv(Lo'*Lo+0.01*eye(size(Lo,2)))*Lo';    
        end
    end    
    
    ANR_PPs = conf.PPs; % store the ANR regressors
    
    %% A+ computing the regressors
    Aplus_PPs = [];
        
    fname = ['Aplus_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszA) 'nn_5mil.mat'];
    
    if exist(fname,'file')
       load(fname);
    else
        %%
       disp('Compute A+ regressors');
       ttime = tic;
       tic
       [plores phires] = collectSamplesScales(conf, load_images(...            
        glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98);  

        if size(plores,2) > 5000000                
            plores = plores(:,1:5000000);
            phires = phires(:,1:5000000);
        end
        number_samples = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);    
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n

        llambda = 0.1;

        for i = 1:size(conf.dict_lores,2)
            D = pdist2(single(plores'),single(conf.dict_lores(:,i)'));
            [~, idx] = sort(D);                
            Lo = plores(:, idx(1:clusterszA));                                    
            Hi = phires(:, idx(1:clusterszA));
            Aplus_PPs{i} = Hi*inv(Lo'*Lo+llambda*eye(size(Lo,2)))*Lo'; 
            %Aplus_PPs{i} = Hi*(inv(Lo*Lo'+llambda*eye(size(Lo,1)))*Lo)'; 
        end        
        clear plores
        clear phires
        
        ttime = toc(ttime);        
        save(fname,'Aplus_PPs','ttime', 'number_samples');   
        toc
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% App (also CRC-0/ICR) computing the regressors
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    App_PPs = [];   
    fname = ['App_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszApp) 'nn_5mil_lam1_' num2str(lam1_miksvd*1000) '_lam2_' num2str(llambda_App*1000) '.mat'];
    
    if exist(fname,'file')
       load(fname);
    else
        %%
        disp('Compute A++ (also CRC-0/IRC) regressors'); 
        ttime = tic;
        tic
        [plores phires] = collectSamplesScales(conf_MIKSVD, load_images(glob('CVPR08-SR/Data/Training', '*.bmp')), 12, 0.98);  
        
        if size(plores,2) > 5000000                
            plores = plores(:,1:5000000);
            phires = phires(:,1:5000000);
        end
        number_samples = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);    
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n

        %llambda_App = 0.1;

        for i = 1:size(conf.dict_lores,2)
            D = abs(single(plores')*single(conf_MIKSVD.dict_lores_MIKSVD(:,i)));
            [~, idx] = sort(D, 'descend');                
            Lo = plores(:, idx(1:clusterszApp));                                    
            Hi = phires(:, idx(1:clusterszApp));
            App_PPs{i} = (Hi/(Lo'*Lo+llambda_App*eye(size(Lo,2))))*Lo';
             
        end        
        clear plores
        clear phires
        
        ttime = toc(ttime);        
        save(fname,'App_PPs','ttime', 'number_samples');   
        toc
    end    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ICR_App_1 (also CRC-1) computing the regressors
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ICR_App_1_PPs = [];

    fname = ['ICR_App_1_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszA) 'nn_5mil.mat'];

    if exist(fname,'file')
       load(fname);
    else
        %%
       disp('Compute ICR_App_1 (also CRC-1) regressors');
       ttime = tic;
       tic
       [plores phires] = PP_collectSamplesScales(conf_ICR_App_1, load_images(PP_glob('CVPR08-SR/Data/Training_91_Mid', 'App', upscaling, '.bmp')), load_images(PP_glob('CVPR08-SR/Data/Training_91_HR', 'Gnd', upscaling, '.bmp')), 12, 0.98);  

        if size(plores,2) > 5000000                
            plores = plores(:,1:5000000);
            phires = phires(:,1:5000000);
        end
        number_samples = size(plores,2);

        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);    
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n

        llambda_ICR_App_1 = 0.1;

        for i = 1:size(conf_ICR_App_1.dict_lores_MIKSVD,2)
            D = abs(single(plores')*single(conf_ICR_App_1.dict_lores_MIKSVD(:,i)));
            [~, idx] = sort(D, 'descend'); 
            Lo = plores(:, idx(1:clusterszA));                                    
            Hi = phires(:, idx(1:clusterszA));
            %ICR_PPs{i} = Hi*inv(Lo'*Lo+llambda*eye(size(Lo,2)))*Lo'; 
            ICR_App_1_PPs{i} = Hi*((Lo'*Lo+llambda_ICR_App_1*eye(size(Lo,2)))\Lo'); 
        end        
        clear plores
        clear phires

        ttime = toc(ttime);        
        save(fname,'ICR_App_1_PPs','ttime', 'number_samples');   
        toc
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ICR_App_2 (also CRC-2) computing the regressors
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ICR_App_2_PPs = [];

    fname = ['ICR_App_2_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszA) 'nn_5mil.mat'];

    if exist(fname,'file')
       load(fname);
    else
        %%
       disp('Compute ICR_App_2 (also CRC-2)  regressors');
       ttime = tic;
       tic
       [plores phires] = PP_collectSamplesScales(conf_ICR_App_2, load_images(PP_glob('CVPR08-SR/Data/Training_91_Mid', 'ICR_App_1', upscaling, '.bmp')), load_images(PP_glob('CVPR08-SR/Data/Training_91_HR', 'Gnd', upscaling, '.bmp')), 12, 0.98);  

        if size(plores,2) > 5000000                
            plores = plores(:,1:5000000);
            phires = phires(:,1:5000000);
        end
        number_samples = size(plores,2);

        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);    
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n

        llambda_ICR_App_2 = 0.1;

        for i = 1:size(conf_ICR_App_2.dict_lores_MIKSVD,2)
            D = abs(single(plores')*single(conf_ICR_App_2.dict_lores_MIKSVD(:,i)));
            [~, idx] = sort(D, 'descend'); 
            Lo = plores(:, idx(1:clusterszA));                                    
            Hi = phires(:, idx(1:clusterszA));
            %ICR_PPs{i} = Hi*inv(Lo'*Lo+llambda*eye(size(Lo,2)))*Lo'; 
            ICR_App_2_PPs{i} = Hi*((Lo'*Lo+llambda_ICR_App_2*eye(size(Lo,2)))\Lo'); 
        end        
        clear plores
        clear phires

        ttime = toc(ttime);        
        save(fname,'ICR_App_2_PPs','ttime', 'number_samples');   
        toc
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Super-Resolution by different methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%    Super-Resolution by different methods    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    conf.result_dirImages = qmkdir([input_dir '/results_' tag]);
    conf.result_dirImagesRGB = qmkdir([input_dir '/results_' tag 'RGB']);
    conf.result_dirRGB = qmkdir(['ResultsRGB-' sprintf('%s_x%d-', input_dir, upscaling) datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);
    
    %%
    t = cputime;    
        
    conf.countedtime = zeros(numel(conf.desc),numel(conf.filenames));
    
    res =[];
    for i = 1:numel(conf.filenames)
        f = conf.filenames{i};
        [p, n, x] = fileparts(f);
        [img, imgCB, imgCR] = load_images({f}); 
        if imgscale<1
            img = resize(img, imgscale, conf.interpolate_kernel);
            imgCB = resize(imgCB, imgscale, conf.interpolate_kernel);
            imgCR = resize(imgCR, imgscale, conf.interpolate_kernel);
        end
        sz = size(img{1});
        
        fprintf('%d/%d\t"%s" [%d x %d]\n', i, numel(conf.filenames), f, sz(1), sz(2));
    
        img = modcrop(img, conf.scale^conf.level);
        imgCB = modcrop(imgCB, conf.scale^conf.level);
        imgCR = modcrop(imgCR, conf.scale^conf.level);

            low = resize(img, 1/conf.scale^conf.level, conf.interpolate_kernel);
            if ~isempty(imgCB{1})
                lowCB = resize(imgCB, 1/conf.scale^conf.level, conf.interpolate_kernel);
                lowCR = resize(imgCR, 1/conf.scale^conf.level, conf.interpolate_kernel);
            end
            
        interpolated = resize(low, conf.scale^conf.level, conf.interpolate_kernel);
        if ~isempty(imgCB{1})
            interpolatedCB = resize(lowCB, conf.scale^conf.level, conf.interpolate_kernel);    
            interpolatedCR = resize(lowCR, conf.scale^conf.level, conf.interpolate_kernel);    
        end
        
        res{1} = interpolated;
                        
        
        if (flag == 1)
            startt = tic;
            res{2} = scaleup_Zeyde(conf, low);
            toc(startt)
            conf.countedtime(2,i) = toc(startt);    
        else
            res{2} = interpolated;
        end
        
        %if flag == 1
            startt = tic;
            res{3} = scaleup_GR(conf, low);
            toc(startt)
            conf.countedtime(3,i) = toc(startt);    
        %else
            %res{3} = interpolated;
        %end
        
        startt = tic;
        conf.PPs = ANR_PPs;
        res{4} = scaleup_ANR(conf, low);
        toc(startt)
        conf.countedtime(4,i) = toc(startt);    
        
        if flag == 1
            startt = tic;
            if 12 < dict_sizes(d)
                res{5} = scaleup_NE_LS(conf, low, 12);
            else
                res{5} = scaleup_NE_LS(conf, low, dict_sizes(d));
            end
            toc(startt)
            conf.countedtime(5,i) = toc(startt);    
        else
            res{5} = interpolated;
        end
        
        if flag == 1
            startt = tic;
            if 24 < dict_sizes(d)
                res{6} = scaleup_NE_NNLS(conf, low, 24);
            else
                res{6} = scaleup_NE_NNLS(conf, low, dict_sizes(d));
            end
            toc(startt)
            conf.countedtime(6,i) = toc(startt);    
        else
            res{6} = interpolated;
        end
        
        if flag == 1
            startt = tic;
            if 24 < dict_sizes(d)
                res{7} = scaleup_NE_LLE(conf, low, 24);
            else
                res{7} = scaleup_NE_LLE(conf, low, dict_sizes(d));
            end
            toc(startt)
            conf.countedtime(7,i) = toc(startt);    
        else
            res{7} = interpolated;
        end
        
        % A+
        if ~isempty(Aplus_PPs)
            fprintf('A+\n');
            conf.PPs = Aplus_PPs;
            startt = tic;
            res{8} = scaleup_ANR(conf, low);
            toc(startt)
            conf.countedtime(8,i) = toc(startt);    
        else
            res{8} = interpolated;
        end         
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ICR
        if ~isempty(App_PPs)
            fprintf('ICR\n');
            conf_MIKSVD.PPs = App_PPs;
            startt = tic;
            res{9} = scaleup_APP_Zhang_MIKSVD(conf_MIKSVD, low);
            toc(startt)
            conf.countedtime(9,i) = toc(startt);
            
        else
            res{9} = interpolated;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ICR_App_1 (also CRC-1) Y.-L. Zhang
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if ~isempty(App_PPs)   % App_PPs && ICR_App_1
            
            conf_MIKSVD.border = [0 0]; % border of the image (to ignore)
            
            fprintf('CRC-1\n');
            conf_MIKSVD.PPs = App_PPs;
            startt = tic;
            % scale up via c_i^1 and F_i^1, where LR is from Bicubic
            Ih_0 = scaleup_APP_Zhang_MIKSVD(conf_MIKSVD, low);
            % scale up via c_i^2 and F_i^2, where LR is from original HR

            conf_ICR_App_1.PPs = ICR_App_1_PPs;
            res{10} = scaleup_ICR_MIKSVD(conf_ICR_App_1, Ih_0, low);
            toc(startt)
            conf.countedtime(10,i) = toc(startt);
            
        else
            res{10} = interpolated;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ICR_App_2 (also CRC-2) Y.-L. Zhang
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if ~isempty(App_PPs)   % App_PPs && ICR_App_1
            
            conf_MIKSVD.border = [0 0]; % border of the image (to ignore)
            
            fprintf('CRC-2\n');
            conf_MIKSVD.PPs = App_PPs;
            startt = tic;
            % scale up via c_i^1 and F_i^1, where LR is from Bicubic
            Ih_0 = scaleup_APP_Zhang_MIKSVD(conf_MIKSVD, low);
            % scale up via c_i^2 and F_i^2, where LR is from original HR

            conf_ICR_App_1.PPs = ICR_App_1_PPs;
            Ih_1 = scaleup_ICR_MIKSVD(conf_ICR_App_1, Ih_0, low);
            
            conf_ICR_App_2.PPs = ICR_App_2_PPs;
            res{11} = scaleup_ICR_MIKSVD(conf_ICR_App_2, Ih_1, low);
            toc(startt)
            conf.countedtime(11,i) = toc(startt);
            
        else
            res{11} = interpolated;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        result = cat(3, img{1}, interpolated{1}, res{2}{1}, res{3}{1}, ...
            res{4}{1}, res{5}{1}, res{6}{1}, res{7}{1}, res{8}{1}, ...
            res{9}{1}, res{10}{1}, res{11}{1});
        
        result = shave(uint8(result * 255), conf.border * conf.scale);
        
        if ~isempty(imgCB{1})
            resultCB = interpolatedCB{1};
            resultCR = interpolatedCR{1};           
            resultCB = shave(uint8(resultCB * 255), conf.border * conf.scale);
            resultCR = shave(uint8(resultCR * 255), conf.border * conf.scale);
        end

        conf.results{i} = {};
        for j = 1:numel(conf.desc)            
            conf.results{i}{j} = fullfile(conf.result_dirImages, [n sprintf('_%s_x%d', conf.desc{j}, upscaling) x]);
            imwrite(result(:, :, j), conf.results{i}{j});

            conf.resultsRGB{i}{j} = fullfile(conf.result_dirImagesRGB, [n sprintf('_%s_x%d', conf.desc{j}, upscaling) x]);
            if ~isempty(imgCB{1})
                rgbImg = cat(3,result(:,:,j),resultCB,resultCR);
                rgbImg = ycbcr2rgb(rgbImg);
            else
                %rgbImg = cat(3,result(:,:,j),result(:,:,j),result(:,:,j));
                rgbImg = result(:,:,j);
            end
            
            imwrite(rgbImg, conf.resultsRGB{i}{j});
        end        
        conf.filenames{i} = f;
    end   
    conf.duration = cputime - t;

    % Test performance
    % PSNR
    run_comparisonRGB_PSNR(conf); % provides color images and HTML summary
end
