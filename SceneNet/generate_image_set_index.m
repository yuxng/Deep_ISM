function generate_image_set_index

N = 200;
opt = globals();
scenes = opt.scenes;

fid_train = fopen('train.txt', 'w');
fid_val = fopen('val.txt', 'w');

% for each scene
for i = 1:numel(scenes)
    scene = scenes{i};
    
    % list models
    models = opt.models{i};
    num = numel(models);
    index = randperm(num);
    
    % half and half
    mid = round(num / 2);
    for j = 1:num
        if j <= mid
            fid = fid_train;
        else
            fid = fid_val;
        end
        model = models{index(j)};
        for k = 1:N
            image_index = sprintf('%s_%s/%06d', scene, model, k-1);
            fprintf(fid, '%s\n', image_index);
        end
    end
end

fclose(fid_train);
fclose(fid_val);