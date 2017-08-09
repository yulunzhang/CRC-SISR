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


















A+: Adjusted Anchored Neighborhood Regression for Fast Super-Resolution
========================================================================

Please cite:
------------
[1] Radu Timofte, Vincent De Smet, Luc Van Gool:
A+: Adjusted Anchored Neighborhood Regression for Fast Super-Resolution, ACCV 2014.

[2] Radu Timofte, Vincent De Smet, Luc Van Gool:
Anchored Neighborhood Regression for Fast Example-Based Super-Resolution, ICCV 2013.

The source codes are freely available for research and study purposes. Enjoy!

Codes written & compiled by:
----------------------------
Radu Timofte
Computer Vision Lab
ETH Zurich, Switzerland
radu.timofte@vision.ee.ethz.ch
http://www.vision.ee.ethz.ch/~timofter/

Packages and codes included and/or adapted:
-------------------------------------------
* Codes by Roman Zeyde are the basis of ours, are used for
training the dictionaries and feature representation
[http://www.cs.technion.ac.il/~elad/Various/Single_Image_SR.zip]

* OMPBox v9+ and KSVDBox v12+ by Ron Rubinstein are used 
for dictionary training and sparse coding in Roman Zeyde's code.
[http://www.cs.technion.ac.il/~ronrubin/software.html]

* Training and test files of Yang et al.'s Super Resolution algorithm.
Image Super-resolution as Sparse Representation of Raw Image Patches, (CVPR) 2008.
[CVPR08-SR/]

Usage
-----
% ACCV2014 paper (A+)

>> go_run_Set5_x2_Aplus1024; % demo running the magnification x2 experiment on Set5 -- Tables 2 and 3 from our ACCV2014 paper
>> go_run_Set5_x3_Aplus1024; % demo running the magnification x3 experiment on Set5 -- Tables 2 and 3 from our ACCV2014 paper
>> go_run_Set5_x4_Aplus1024; % demo running the magnification x4 experiment on Set5 -- Tables 2 and 3 from our ACCV2014 paper
>> go_run_Set14_x2_Aplus1024; % demo running the magnification x2 experiment on Set14 -- Tables 3 from our ACCV2014 paper
>> go_run_Set14_x3_Aplus1024; % demo running the magnification x3 experiment on Set14 -- Tables 1 and 3 from our ACCV2014 paper
>> go_run_Set14_x4_Aplus1024; % demo running the magnification x4 experiment on Set14 -- Tables 3 from our ACCV2014 paper
>> go_run_B100_x2_Aplus1024; % demo running the magnification x2 experiment on B100 -- Table 3 from our ACCV2014 paper
>> go_run_B100_x3_Aplus1024; % demo running the magnification x3 experiment on B100 -- Table 3 from our ACCV2014 paper
>> go_run_B100_x4_Aplus1024; % demo running the magnification x4 experiment on B100 -- Table 3 from our ACCV2014 paper

% ICCV2013 paper (ANR)

>> go_run_upscaling_experiment; % demo running one experiment setting using an magnification factor and set of images

>> go_run_Set14;   % demo running the magnification x3 experiment on Set14 -- results in Table 1 from our ICCV2013 paper
>> go_run_Set5_x2; % demo running the magnification x2 experiment on Set5 -- Table 2 from our ICCV2013 paper
>> go_run_Set5_x3; % demo running the magnification x3 experiment on Set5 -- Table 2 from our ICCV2013 paper
>> go_run_Set5_x4; % demo running the magnification x4 experiment on Set5 -- Table 2 from our ICCV2013 paper

Written
-------
30.03.2013, Radu Timofte @ KU Leuven

Revised versions
----------------
03.10.2013, Radu Timofte @ ETH Zurich (public release)
01.02.2014, Radu Timofte @ ETH Zurich (added A+)
10.06.2014, Radu Timofte @ ETH Zurich (added B100)
01.09.2014, Radu Timofte @ ETH Zurich (final A+)
