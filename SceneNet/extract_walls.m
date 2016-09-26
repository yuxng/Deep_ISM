function Wall = extract_walls(obj)

V = [];
F = [];
vnum = 0;
fnum = 0;
object_names = obj.object_names;

for i = 1:numel(object_names)
    name = object_names{i};
    
    if isempty(strfind(name, 'wall')) == 0 || ...
            isempty(strfind(name, 'Wall')) == 0 || ...
            isempty(strfind(name, 'room_skeleton')) == 0
        vindex = find(obj.v_index == i);
        num = numel(vindex);
        vnum_old = vnum;
        if isempty(V)
            V = obj.v(:, vindex)';
            vnum = num;
        else
            V(vnum+1:vnum+num, :) = obj.v(:, vindex)';
            vnum = vnum + num;
        end
        
        findex = find(obj.f3_index == i);
        num = numel(findex);
        if isempty(F)
            F = obj.f3(:, findex)' - min(vindex) + 1 + vnum_old;
            fnum = num;
        else
            F(fnum+1:fnum+num, :) = obj.f3(:, findex)' - min(vindex) + 1 + vnum_old;
            fnum = fnum + num;
        end
    end
end

Wall.faces = F;
Wall.vertices = V;

% figure(1);
% trimesh(F, V(:,1), V(:,3), V(:,2), 'EdgeColor', 'b');
% axis equal;
% hold off;