function process_scenenet

opt = globals();

for i = 1:numel(opt.scenes)
    for j = 1:numel(opt.models{i})
        generate_camera_trajectory(i, j);
    end
end