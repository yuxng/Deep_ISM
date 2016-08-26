% backproject depth map into 3D points
function points = backproject(depth, meta_data)

% convert depth map
depth = double(depth) / double(meta_data.factor_depth);
width = size(depth, 2);
height = size(depth, 1);

% projection matrix
P = meta_data.projection_matrix;

% camera location
C = meta_data.camera_location;
Cmat = repmat(C', [1 height*width]);

% construction the 2D points
x2d = zeros(height, width, 3);
[x, y] = meshgrid(1:width, 1:height);
x2d(:,:,1) = x;
x2d(:,:,2) = y;
x2d(:,:,3) = ones(height, width);
x2d = reshape(x2d, [height * width, 3]);

% backprojection
x3d = pinv(P) * x2d';
x3d(1, :) = x3d(1, :) ./ x3d(4, :);
x3d(2, :) = x3d(2, :) ./ x3d(4, :);
x3d(3, :) = x3d(3, :) ./ x3d(4, :);
x3d = x3d(1:3, :);

% compute the ray
R = x3d - Cmat;

% compute the norm
N = sqrt(sum(R.^2, 1));
        
% normalization
R = R ./ repmat(N, [3,1]);

% compute the 3D points
X = Cmat + repmat(reshape(depth, [1, width*height]), [3, 1]) .* R;
points = zeros(height, width, 3);
points(:, :, 1) = reshape(X(1,:), [height, width]);
points(:, :, 2) = reshape(X(3,:), [height, width]);
points(:, :, 3) = reshape(X(2,:), [height, width]);