function generate_videos

num_scene = 100;
ffmpeg = '/var/Softwares/ffmpeg-3.1.3-64bit-static/ffmpeg';
frame_rate = 8;

for i = 1:num_scene
    
        % generate rgb video
        file_video = sprintf('video/%04d_rgba.avi', i-1);
        fprintf('save video to %s\n', file_video);
        images = sprintf('data/%04d/%%02d_rgba.png', i-1);
        
        cmd = sprintf('%s -r %d -start_number 0 -i %s -vcodec mpeg4 -b 800K %s', ...
            ffmpeg, frame_rate, images, file_video);
        system(cmd);        
        
        % generate depth video
%         file_video = sprintf('video/%04d_depth.avi', i-1);
%         fprintf('save video to %s\n', file_video);
%         images = sprintf('data/%04d/%%02d_depth.png', i-1);
%         
%         cmd = sprintf('%s -r %d -start_number 0 -i %s -vcodec mpeg4 -b 800K %s', ...
%             ffmpeg, frame_rate, images, file_video);
%         system(cmd);
end