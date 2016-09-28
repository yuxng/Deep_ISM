function generate_videos

opt = globals();
ffmpeg = '/var/Softwares/ffmpeg-3.1.3-64bit-static/ffmpeg';
frame_rate = 8;

for i = 1:numel(opt.scenes)
    scene = opt.scenes{i};
    for j = 1:numel(opt.models{i})
        model = opt.models{i}{j};
        % generate rgb video
        file_video = sprintf('video/%s_%s_scene.avi', scene, model);
        fprintf('save video to %s\n', file_video);
        images = sprintf('data/%s_%s/%%06d_scene.png', scene, model);
        
        cmd = sprintf('%s -r %d -start_number 0 -i %s -vcodec mpeg4 -b 800K %s', ...
            ffmpeg, frame_rate, images, file_video);
        system(cmd);        
        
        % generate depth video
        file_video = sprintf('video/%s_%s_depth.avi', scene, model);
        fprintf('save video to %s\n', file_video);
        images = sprintf('data/%s_%s/%%06d_depth.png', scene, model);
        
        cmd = sprintf('%s -r %d -start_number 0 -i %s -vcodec mpeg4 -b 800K %s', ...
            ffmpeg, frame_rate, images, file_video);
        system(cmd);
    end
end