# CRC-SISR
Matlab code for Collaborative Representation Cascade for Single-Image Super-Resolution
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

% Usage
Run 'Demo_CRC.m'.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Here, we can change upscaling factors (x2/3/4), test sets, and image
%%% patterns
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
upscaling = 2;           % the magnification factor x2, x3, x4...
input_dir = 'Set10Dong'; % Set5, Set14, Set10Dong, SetSRF
pattern = '*.bmp';       % Pattern to process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
