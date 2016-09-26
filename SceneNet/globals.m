function opt = globals()

opt.root = '/var/Projects/SceneNet';

opt.scenes = {'1Bathroom', '1Bedroom', '1Kitchen', '1Living-room', '1Office'};
opt.models = cell(1, numel(opt.scenes));

% list 3D models
for i = 1:numel(opt.scenes)
    scene = opt.scenes{i};
    files = dir(fullfile(opt.root, scene, '*.obj'));
    num = numel(files);
    models = cell(num, 1);
    for j = 1:num
        name = files(j).name;
        pos = strfind(name, '.');
        models{j} = name(1:pos-1);
    end
    opt.models{i} = models;
end