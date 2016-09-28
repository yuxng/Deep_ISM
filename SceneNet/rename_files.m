function rename_files

N = 200;

opt = globals();
scenes = opt.scenes;

% for each scene
for i = 1:numel(scenes)
    scene = scenes{i};
    
    % list models
    models = opt.models{i};

    for j = 1:numel(models)
        model = models{j};
        fprintf('%s_%s\n', scene, model);
        for k = 0:N-1
            filename_old = sprintf('data/%s_%s/depth_%06d.png', scene, model, k);
            if exist(filename_old, 'file')
                filename_new = sprintf('data/%s_%s/%06d_depth.png', scene, model, k);
                movefile(filename_old, filename_new);
            end
            
            filename_old = sprintf('data/%s_%s/label_%06d.png', scene, model, k);
            if exist(filename_old, 'file')
                filename_new = sprintf('data/%s_%s/%06d_label.png', scene, model, k);
                movefile(filename_old, filename_new);
            end
            
            filename_old = sprintf('data/%s_%s/scene_%06d.png', scene, model, k);
            if exist(filename_old, 'file')
                filename_new = sprintf('data/%s_%s/%06d_scene.png', scene, model, k);
                movefile(filename_old, filename_new);
            end                
            
            filename_old = sprintf('data/%s_%s/pose_%06d.txt', scene, model, k);
            if exist(filename_old, 'file')
                filename_new = sprintf('data/%s_%s/%06d_pose.txt', scene, model, k);
                movefile(filename_old, filename_new);
            end            
        end
    end
end