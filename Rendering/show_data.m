function show_data

% data directory
root = './data';
shapenet = '/var/Projects/ShapeNetCore.v1';

% synset id
synset = '02958343';

% list models
subdirs = dir(fullfile(root, synset));

% for each model
for i = 3:numel(subdirs)
    model = subdirs(i).name;
    
    % load shapenet model
    filename = fullfile(shapenet, synset, model, 'model.obj');
    fprintf('loading model %s\n', filename);
    obj = load_obj_file(filename);
    fprintf('model loaded\n');
    
    % display the obj model
    subplot(2, 2, 1);
    faces = obj.f3';
    vertices = obj.v';
    trimesh(faces, vertices(:,1), vertices(:,3), vertices(:,2), 'EdgeColor', 'b');
    axis equal;
    xlabel('x');
    ylabel('y');
    zlabel('z');
    
    % process renderings
    meta_files = dir(fullfile(root, synset, model, '*.mat'));
    for j = 1:numel(meta_files)
        id = meta_files(j).name(1:2);
        meta_data = load(fullfile(root, synset, model, meta_files(j).name));
        subplot(2, 2, 1);
        view(meta_data.azimuth + 90, meta_data.elevation);
        
        % read rgba image
        filename = fullfile(root, synset, model, [id '_rgba.png']);
        [I, ~, alpha] = imread(filename);
        subplot(2, 2, 2);
        imshow(I);
        
        % read depth image
        filename = fullfile(root, synset, model, [id '_depth.png']);
        depth = imread(filename);
        subplot(2, 2, 3);
        imshow(depth);
        
        % show the 3D points
        points = backproject(depth, meta_data);
        subplot(2, 2, 4);
        plot3(points(:, :, 1), points(:, :, 3), points(:, :, 2), 'b.');
        view(meta_data.azimuth + 90 + 30, meta_data.elevation + 15);
        axis equal;
        xlabel('x');
        ylabel('y');
        zlabel('z');
        
        % randomly show a point
        [row, col] = find(depth > 0);
        ind = randperm(numel(row));
        x = col(ind(1));
        y = row(ind(1));
        subplot(2, 2, 1);
        hold on;
        plot3(points(y, x, 1), points(y, x, 3), points(y, x, 2), 'ro', 'LineWidth', 2);
        hold off;
        
        subplot(2, 2, 2);
        hold on;
        plot(x, y, 'ro');
        hold off;
        
        subplot(2, 2, 3);
        hold on;
        plot(x, y, 'ro');
        hold off;
        
        subplot(2, 2, 4);
        hold on;
        plot3(points(y, x, 1), points(y, x, 3), points(y, x, 2), 'ro', 'LineWidth', 2);
        hold off;        
    end
end