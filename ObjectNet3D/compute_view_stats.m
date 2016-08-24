function compute_view_stats

opt = globals();

% read classes
filename = fullfile(opt.root, 'Image_sets', 'classes.txt');
classes = textread(filename, '%s');
num_class = numel(classes);

% read ids
filename = fullfile(opt.root, 'Image_sets', 'trainval.txt');
ids = textread(filename, '%s');
M = numel(ids);

azimuths = cell(num_class, 1);
elevations = cell(num_class, 1);
tilts = cell(num_class, 1);
distances = cell(num_class, 1);

% for each annotation file
for i = 1:M
    
    % load annotation
    filename = fullfile(opt.root, 'Annotations', sprintf('%s.mat', ids{i}));
    annotation = load(filename);
    objects = annotation.record.objects;
    
    % for each object
    for k = 1:numel(objects)
        object = objects(k);
        
        % class index
        cls_ind = find(strcmp(object.class, classes) == 1);
        
        % viewpoint information
        viewpoint = object.viewpoint;
        if isfield(viewpoint, 'azimuth') == 0 || isempty(viewpoint.azimuth) == 1
            azimuth = viewpoint.azimuth_coarse;
        else
            azimuth = viewpoint.azimuth;
        end
        if isfield(viewpoint, 'elevation') == 0 || isempty(viewpoint.elevation) == 1
            elevation = viewpoint.elevation_coarse;
        else
            elevation = viewpoint.elevation;
        end

        theta = viewpoint.theta;
        distance = viewpoint.distance;
            
        % add viewpoint information
        azimuths{cls_ind} = [azimuths{cls_ind} azimuth];
        elevations{cls_ind} = [elevations{cls_ind} elevation];
        tilts{cls_ind} = [tilts{cls_ind} theta];
        distances{cls_ind} = [distances{cls_ind} distance];
    end
end

save('objectnet3d_viewpoint_stats.mat', 'classes', 'azimuths', 'elevations', 'tilts', 'distances');