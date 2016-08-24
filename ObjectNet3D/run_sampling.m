% Generate view samples for PASCAL3D classes
function run_sampling

if exist('objectnet3d_viewpoint_stats.mat', 'file') == 0
    compute_view_stats;
end

% load viewpoint stats
object = load('objectnet3d_viewpoint_stats.mat');
classes = object.classes;
azimuths = object.azimuths;
elevations = object.elevations;
tilts = object.tilts;
distances = object.distances;

% aggeragate information
elevations_all = [];
tilts_all = [];
for k = 1:numel(classes)
    elevations_all = [elevations_all, elevations{k}];
    tilts_all = [tilts_all tilts{k}];
end

% KDE on real image samples and Generate samples from estimated distributions
elevation_avg = mean(elevations_all);
elevation_std = 20;
tilt_avg = mean(tilts_all);
tilt_std = 20;
num_samples = 10^5;
outlier_ratio = 0.2;

for k = 1:numel(classes)
    cls = classes{k};
    sample_viewpoints(cls, azimuths{k}, elevations{k}, tilts{k}, distances{k}, ...
        num_samples, outlier_ratio, elevation_avg, elevation_std, tilt_avg, tilt_std);
end