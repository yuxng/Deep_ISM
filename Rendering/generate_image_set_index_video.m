function generate_image_set_index_video

num_scene = 100;
num_view = 100;

fid_train = fopen('train.txt', 'w');
fid_val = fopen('val.txt', 'w');

% half and half
mid = round(num_scene / 2);
% index = randperm(num_scene);
% index_train = sort(index(1:mid));
% index_val = sort(index(mid+1:end));

index_train = 1:mid;
index_val = mid+1:num_scene;

% for each scene
for i = 1:numel(index_train)
    ind = index_train(i);
    for j = 1:num_view
        image_index = sprintf('%04d/%04d', ind-1, j-1);
        fprintf(fid_train, '%s\n', image_index);
    end

end

for i = 1:numel(index_val)
    ind = index_val(i);
    for j = 1:num_view
        image_index = sprintf('%04d/%04d', ind-1, j-1);
        fprintf(fid_val, '%s\n', image_index);
    end        
end

fclose(fid_train);
fclose(fid_val);
