function merge_audio_video_ffmpeg(videoIn, audioIn, outFile, ~)

    cmd = sprintf([ ...
        'ffmpeg -y -i "%s" -i "%s" ' , ...
        '-filter:v "scale=trunc(iw/2)*2:trunc(ih/2)*2" ' , ...  % enforce even width/height
        '-c:v libx264 -pix_fmt yuv420p ' , ...
        '-c:a aac -b:a 192k -shortest "%s"' ], ...
        videoIn, audioIn, outFile);

    status = system(cmd);
    if status ~= 0
        error('Failed to merge audio and video to %s', outFile);
    end
end
