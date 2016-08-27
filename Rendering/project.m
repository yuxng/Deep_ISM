% project 3D points to the image plane
% x3d is a 3 x N matrix with each column a 3D point
function x2d = project(x3d, meta_data)

% projection matrix
P = meta_data.projection_matrix;

% homogeneous coordinates
num = size(x3d, 2);
x3d = [x3d; ones(1, num)];

% projection
x2d = P * x3d;
x2d(1, :) = x2d(1, :) ./ x2d(3, :);
x2d(2, :) = x2d(2, :) ./ x2d(3, :);
x2d = x2d(1:2, :);