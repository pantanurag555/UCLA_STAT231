% prep
clear all;
close all;

% flags
flag_compile_libsvm_c = 1;
flag_compile_libsvm_mex = 1;

% compile libsvm
if flag_compile_libsvm_c
    parent = cd('libsvm-3.21');
    [status,cmdout] = system('make');
    cd(parent);
    disp(status);
    disp(cmdout);
end

if flag_compile_libsvm_mex
    parent = cd('libsvm-3.21/matlab');
    make;
    cd(parent);
end

% setup
diary('P1_1.out');
rng(123);
addpath('libsvm-3.21/matlab');

% data
disp('loading data ...');
load('train-anno.mat', 'face_landmark', 'trait_annotation');
features = face_landmark;
labels = trait_annotation;

% predict
disp('cross validation ...');
%% todo: perform k-fold cross-validation