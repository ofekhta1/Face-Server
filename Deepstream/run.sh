xhost +
docker run --gpus all -it --rm --net=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -v ./:/opt/face_pipeline -e DISPLAY=$DISPLAY -w /opt/face_pipeline dangrin/face-deepstream:v1.0 