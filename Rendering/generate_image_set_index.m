function generate_image_set_index

% data directory
root = './data';

% synset ids
synsets = {'02958343', '03001627', '03211117', '03797390', '04379243'};

fid_train = fopen('train.txt', 'w');
fid_val = fopen('val.txt', 'w');

% for each synset
for i = 1:numel(synsets)
    synset = synsets{i};
    
    % list models
    subdirs = dir(fullfile(root, synset));
    num = numel(subdirs) - 2;
    
    % half and half
    mid = round(num / 2);
    for j = 1:num
        if j <= mid
            fid = fid_train;
        else
            fid = fid_val;
        end
        model = subdirs(j+2).name;
        for k = 1:10
            image_index = fullfile(synset, model, sprintf('%02d', k-1));
            fprintf(fid, '%s\n', image_index);
        end
    end
end

fclose(fid_train);
fclose(fid_val);