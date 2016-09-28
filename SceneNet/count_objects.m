function count_objects

opt = globals();
classes = lower({ 'UNKNOWN', 'FLOOR', 'CEILING', 'WALL', 'BED', ...
    'CHAIR', 'FURNITURE', 'NIGHTSTAND', 'SHELF', 'CURTAIN', ...
    'PAINTING', 'PILLOWS', 'DOOR', 'WINDOW', 'TABLE', 'SOFA', ...
    'LAMP', 'VASE', 'PLANT', 'PLATE', 'STAND'});
count = zeros(numel(classes), 1);

for i = 1:numel(opt.scenes)
    scene = opt.scenes{i};
    for j = 1:numel(opt.models{i})
        model = opt.models{i}{j};
   
        % load obj file
        mat_file = sprintf('cache/%s_%s.mat', scene, model);
        if exist(mat_file, 'file')
            object = load(mat_file);
            obj = object.obj;
        else
            obj = load_obj_file(obj_file);
            save(mat_file, 'obj');
        end
        
        object_names = obj.object_names;
        for k = 1:numel(object_names)
            name = lower(object_names{k});
            
            if isempty(strfind(name, 'floor')) == 0 || ...
                    isempty(strfind(name, 'carpet')) == 0 || ...
                    isempty(strfind(name, 'ground')) == 0 || ...
                    isempty(strfind(name, 'rug')) == 0 || ...
                    isempty(strfind(name, 'mat_')) == 0
                class = 'floor';
                
            elseif isempty(strfind(name, 'ceiling')) == 0
                class = 'ceiling';
                
            elseif isempty(strfind(name, 'wall')) == 0 || ...
                    isempty(strfind(name, 'room_skeleton')) == 0
                class = 'wall';
            
            elseif isempty(strfind(name, 'bed')) == 0 || ...
                    isempty(strfind(name, 'duvet')) == 0
                class = 'bed';
                
            elseif isempty(strfind(name, 'chair')) == 0
                class = 'chair';
                
            elseif isempty(strfind(name, 'cupboard')) == 0 || ...
                    isempty(strfind(name, 'furniture')) == 0 || ...
                    isempty(strfind(name, 'chest')) == 0 || ...
                    isempty(strfind(name, 'drawers')) == 0 || ...
                    isempty(strfind(name, 'wardrobe')) == 0
                class = 'furniture';
                
            elseif isempty(strfind(name, 'nightstand')) == 0 || ...
                    isempty(strfind(name, 'night_stand')) == 0
                class = 'nightstand';
                
            elseif isempty(strfind(name, 'shelf')) == 0 || ...
                    isempty(strfind(name, 'shelves')) == 0
                class = 'shelf';
                
            elseif isempty(strfind(name, 'curtain')) == 0 || ...
                    isempty(strfind(name, 'blind')) == 0
                class = 'curtain';
                
            elseif isempty(strfind(name, 'painting')) == 0 || ...
                    isempty(strfind(name, 'paint')) == 0 || ...
                    isempty(strfind(name, 'mirror')) == 0 || ...
                    isempty(strfind(name, 'tv')) == 0
                class = 'painting';       
                
            elseif isempty(strfind(name, 'pillow')) == 0 || ...
                    isempty(strfind(name, 'cushion')) == 0 || ...
                    isempty(strfind(name, 'pilow')) == 0
                class = 'pillows';
                
            elseif isempty(strfind(name, 'door')) == 0
                class = 'door';                
                
            elseif isempty(strfind(name, 'window')) == 0
                class = 'window';                
                
            elseif isempty(strfind(name, 'table')) == 0
                class = 'table';
                
            elseif isempty(strfind(name, 'sofa')) == 0 || ...
                    isempty(strfind(name, 'bench')) == 0
                class = 'sofa';                
                
            elseif isempty(strfind(name, 'lamp')) == 0
                class = 'lamp';                
                
            elseif isempty(strfind(name, 'palm')) == 0 || ...
                    isempty(strfind(name, 'plant')) == 0
                class = 'plant';                
                
            elseif isempty(strfind(name, 'vase')) == 0
                class = 'vase';                
                
            elseif isempty(strfind(name, 'plate')) == 0
                class = 'plate';                
                
            elseif isempty(strfind(name, 'stand')) == 0
                class = 'stand';
                
            else
                class = 'unknown';
                
            end
            
            index = strcmp(class, classes);
            count(index) = count(index) + 1;
        end
        
    end
end

for i = 1:numel(classes)
    fprintf('%s: %d\n', classes{i}, count(i));
end