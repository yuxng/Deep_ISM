function generate_camera_trajectory(index_scene, index_model)

is_show = 1;

opt = globals();
scene = opt.scenes{index_scene};
model = opt.models{index_scene}{index_model};
fprintf('scene %s, model %s\n', scene, model);

filename = sprintf('data/%s_%s/camera_trajectory.txt', scene, model);
if exist(filename, 'file')
    return;
end

delta = 2.5;
num_frame = 200;
RTs = cell(num_frame, 1);

obj_file = fullfile(opt.root, scene, [model '.obj']);

% load obj file
mat_file = sprintf('cache/%s_%s.mat', scene, model);
if exist(mat_file, 'file')
    object = load(mat_file);
    obj = object.obj;
else
    obj = load_obj_file(obj_file);
    save(mat_file, 'obj');
end
vertices = obj.v';

% extract walls
Wall = extract_walls(obj);
if isempty(Wall.vertices)
    return;
end
All.vertices = obj.v';
All.faces = obj.f3';

X = Wall.vertices(:,1);
Xmin = min(X);
Xmax = max(X);

Y = Wall.vertices(:,3);
Ymin = min(Y);
Ymax = max(Y);

Z = Wall.vertices(:,2);
Zmin = 1.5;
Zmax = 1.8;
fprintf('3D model limit: x (%.2f, %.2f), y (%.2f, %.2f), z (%.2f, %.2f)\n', ...
    Xmin, Xmax, Ymin, Ymax, min(Z), max(Z));

while(1)
    % sample a point to look at on the ground
    P = zeros(3, 1);
    P(1) = mean(X) + rand(1) * (Xmax - Xmin) / 8;
    P(3) = mean(Y) + rand(1) * (Ymax - Ymin) / 8;    

    % sample the initial camera location
    T = zeros(3, 1);
    T(2) = Zmin + rand(1) * (Zmax - Zmin);
    while(1)
        T(1) = Xmin + rand(1) * (Xmax - Xmin);
        T(3) = Ymin + rand(1) * (Ymax - Ymin);
        plot3(T(1), T(3), T(2), 'ro', 'LineWidth', 1);
        distance = point2trimesh(All, 'QueryPoints', T', 'UseSubSurface', false);
        
        if T(1) > Xmin + 0.2 && T(1) < Xmax - 0.2 && ...
            T(3) > Ymin + 0.2 && T(3) < Ymax - 0.2 && distance > 0.2
            break;
        end
    end
    
    % check the angle between T and P
    V = T - P;
    angle = asin(V(2) / norm(V));
    if abs(angle) < pi/4
        break;
    end
end

fprintf('camera location: %.2f, %.2f, %.2f\n', T(1), T(2), T(3));
fprintf('camera look at: %.2f, %.2f, %.2f\n', P(1), P(2), P(3));

RT_wc = compute_SE3(T, P);
RTs{1} = RT_wc;

% draw
if is_show
    figure(1);
    index = randperm(size(vertices, 1));
    if numel(index) > 10000
        index = index(1:10000);
    end
    plot3(vertices(index,1), vertices(index,3), vertices(index,2), '.');
    hold on;
    trimesh(Wall.faces, Wall.vertices(:,1), Wall.vertices(:,3), Wall.vertices(:,2), 'EdgeColor', 'b', 'FaceAlpha', 0);
    draw_camera(RT_wc, T, P);
    axis equal;
    xlabel('x');
    ylabel('y');
    zlabel('z');
end

% generate following frames
theta = delta * pi / 180;
for i = 2:num_frame
    while (1)
        R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
        % rotate the camera around p for theta
        x = R * [T(1) - P(1); T(3) - P(3)] + [P(1); P(3)];
        if x(1) < Xmin
            x(1) = Xmin + 0.01;
        end
        if x(1) > Xmax
            x(1) = Xmax - 0.01;
        end

        if x(2) < Ymin
            x(2) = Ymin - 0.01;
        end
        if x(2) > Ymax
            x(2) = Ymax + 0.01;
        end    

        Tnew = T;
        Tnew(1) = x(1) + randn(1) * 0.01;
        Tnew(2) = T(2) + randn(1) * 0.01;
        Tnew(3) = x(2) + randn(1) * 0.01;

        % compute distance to wall
        distance = point2trimesh(All, 'QueryPoints', Tnew', 'UseSubSurface', false);
        disp(distance);
        if distance < 0.2
            theta = -1 * theta;
        else
            break;
        end
    end
    
    % perturb P
    while(1)
        Pnew = P;
        Pnew(1) = P(1) + randn(1) * 0.1;
        Pnew(3) = P(3) + randn(1) * 0.1;
        if Pnew(1) > Xmin + 0.1 && Pnew(1) < Xmax - 0.1 && ...
            Pnew(3) > Ymin + 0.1 && Pnew(3) < Ymax - 0.1
            P = Pnew;
            break;
        end        
    end
    
    RT_wc = compute_SE3(Tnew, P);
    T = Tnew;
    
    RTs{i} = RT_wc;
    
    if is_show
        draw_camera(RT_wc, Tnew, P);
        pause(0.1);
    end
end

if is_show
    hold off;
end

% check the output dir
outdir = sprintf('data/%s_%s', scene, model);
if exist(outdir, 'dir') == 0
    mkdir(outdir);
end

% write RTs to file
filename = sprintf('%s/camera_trajectory.txt', outdir);
fid = fopen(filename, 'w');
for k = 1:num_frame
    RT_wc = RTs{k};
    for i = 1:size(RT_wc, 1)
        for j = 1:size(RT_wc, 2)
            fprintf(fid, '%f ', RT_wc(i,j));
        end
        fprintf(fid, '\n');
    end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('finish %s\n', filename);

function draw_camera(RT_wc, T, P)

plot3(T(1), T(3), T(2), 'ro', 'LineWidth', 1);
plot3(P(1), P(3), P(2), 'go', 'LineWidth', 1);
% draw the axis
x = apply_RT(RT_wc, [0.4; 0; 0]);
z = apply_RT(RT_wc, [0; 0.4; 0]);
y = apply_RT(RT_wc, [0; 0; 0.4]);
quiver3(T(1), T(3), T(2), x(1)-T(1), x(3)-T(3), x(2)-T(2), 'r');
quiver3(T(1), T(3), T(2), y(1)-T(1), y(3)-T(3), y(2)-T(2), 'g');
quiver3(T(1), T(3), T(2), z(1)-T(1), z(3)-T(3), z(2)-T(2), 'b');

function RT_wc = compute_SE3(T, P)

% compute the rotation matrix, camera to world
a = [0 0 1];
b = P - T;
R = rotation_matrix(a, b);

% world to camera
RT_cw = [R' -R'*T];

% find inplane rotation to align horizontal line
theta = compute_inplane_rotation(RT_cw, P);
Rz = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
RT_cw = Rz * RT_cw;

% camera to world
R_cw = RT_cw(:,1:3);
T_cw = RT_cw(:,4);
RT_wc = [R_cw' -R_cw'*T_cw];

% rotation matrix to align vector a to vector b
function R = rotation_matrix(a, b)

% normalization
a = a / norm(a);
b = b / norm(b);

v = cross(a, b);
s = norm(v);
c = dot(a, b);

V =[0 -v(3) v(2) ; v(3) 0 -v(1) ; -v(2) v(1) 0 ];
R = eye(3) + V + V * V * (1 - c) / (s * s);

function b = apply_RT(RT, a)

b = RT * [a; 1];

function error = compute_error(x, P, P1, RT_cw)

Rz = [cos(x) -sin(x) 0; sin(x) cos(x) 0; 0 0 1];

% projection
p = Rz * RT_cw * [P; 1];
p1 = Rz * RT_cw * [P1; 1];
error = abs(p1(1) - p(1));

function theta = compute_inplane_rotation(RT_cw, P)

P1 = P;
P1(2) = P1(2) + 1;

options = optimset('Algorithm', 'interior-point');
[theta, fval] = fmincon(@(x)compute_error(x, P, P1, RT_cw),...
    0, [], [], [], [], -pi, pi, [], options);

Rz = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];

% projection to check if camera is upright
p = Rz * RT_cw * [P; 1];
p1 = Rz * RT_cw * [P1; 1];
if p1(2) < p(2)
    theta = theta + pi;
end