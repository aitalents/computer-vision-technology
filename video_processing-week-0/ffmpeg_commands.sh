# запись rtsp потока в файл
ffmpeg -i rtsp://10.10.67.125:8554/test -b 900k -vcodec copy -r 60 -t 10 -rtsp_transport tcp -y test_out.avi

# запись из поток с изменениями в кадре
ffmpeg -i rtsp://10.10.67.125:8554/test -vf "select=gt(scene\,0.1)" -b 900k -vcodec copy -r 60 -t 10 -rtsp_transport tcp -y test_out.avi

# convert all avi files to mp4
for i in *.avi; do ffmpeg -i "$i" -c:v copy -c:a copy "${i%.*}.mp4"; done

# extract frames from all video files in folder
cd dataset_videos_2/processed/ # dirr with cropped videos
for d in */
do
    ( cd "$d" && for i in *.mp4; do ffmpeg -i "$i" -vf "select=not(mod(n\,25))" -vsync vfr "../all_frames/${d//\//}_%04d.jpg"; done )
done