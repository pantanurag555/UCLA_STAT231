addpath('./libsvm_matlab/');
mex HoGfeatures.cc

im = imread('test.jpg');
im = double(im);
hogfeat = HoGfeatures(im);
imshow(drawHOGtemplates(hogfeat, [29 29]));

