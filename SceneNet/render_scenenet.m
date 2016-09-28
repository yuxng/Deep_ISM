function render_scenenet

opt = globals();
tool = '/var/Projects/SceneNetv1.0/build/opengl_depth_rendering';

for i = 1:numel(opt.scenes)
    scene = opt.scenes{i};
    for j = 1:numel(opt.models{i})
        model = opt.models{i}{j};
        cmd = sprintf('%s %s/%s/%s.obj', tool, opt.root, scene, model);
        fprintf('%s\n', cmd);
%         filename = sprintf('data/%s_%s/depth_000000.png', scene, model);
%         if exist(filename, 'file')
%             continue;
%         end
        system(cmd);
    end
end

% index_scene = 1;
% index_model = 2;

% scene = opt.scenes{index_scene};
% model = opt.models{index_scene}{index_model};
% cmd = sprintf('%s %s/%s/%s.obj', tool, opt.root, scene, model);
% system(cmd);