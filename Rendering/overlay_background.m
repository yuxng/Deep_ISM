function overlay_background

% data directory
root = './data';
root_bg = '/var/Projects/Deep_ISM/sun2012pascalformat/JPEGImages';

% list sun images
files_bg = dir(fullfile(root_bg, '*.jpg'));
num_sun = numel(files_bg);

% synset id
synset = '03797390';

% list models
subdirs = dir(fullfile(root, synset));

% for each model
for i = 3:numel(subdirs)
    model = subdirs(i).name;
    fprintf('%s %s\n', synset, model);
    
    % process renderings
    meta_files = dir(fullfile(root, synset, model, '*.mat'));
    for j = 1:numel(meta_files)
        id = meta_files(j).name(1:2);
        
        % read rgba image
        filename = fullfile(root, synset, model, [id '_rgba.png']);
        [I, ~, alpha] = imread(filename);
        
        s = size(I);
        fh = s(1);
        fw = s(2);
        mask = double(alpha) / 255;
        mask = repmat(mask, [1 1 3]);
        
        % randomly sample a SUN image
        ind = randi(num_sun);
        bg = imread(fullfile(root_bg, files_bg(ind).name));
        s = size(bg);
        bh = s(1);
        bw = s(2);
        if length(s) < 3
            bg = repmat(bg, [1 1 3]);
        end
        
        if bh >= fh && bw >= fw
            by = randi(bh - fh + 1);
            bx = randi(bw - fw + 1);
            bgcrop = bg(by:(by+fh-1), bx:(bx+fw-1), :);
        else
            ratio = max(fh / bh, fw / bw);
            bg = imresize(bg, ratio);
            
            by = randi(size(bg,1) - fh + 1);
            bx = randi(size(bg,2) - fw + 1);
            bgcrop = bg(by:(by+fh-1), bx:(bx+fw-1), :);
        end

        I = uint8(double(I) .* mask + double(bgcrop) .* (1 - mask));
        
        % save image
        filename = fullfile(root, synset, model, [id '_bkgd.png']);
        imwrite(I, filename);
        
%         imshow(I);
%         pause;

    end        
end