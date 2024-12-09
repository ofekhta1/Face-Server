FROM nvcr.io/nvidia/deepstream:7.1-gc-triton-devel
WORKDIR /app
COPY Shared /common/Shared
COPY setup.py /common/

# To get video driver libraries at runtime (libnvidia-encode.so/libnvcuvid.so)
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video
ENV PYTHONPATH $PYTHONPATH:/common

RUN /opt/nvidia/deepstream/deepstream/user_additional_install.sh
RUN /opt/nvidia/deepstream/deepstream/user_deepstream_python_apps_install.sh -b
RUN apt install -y ffmpeg mlocate
# FFmpeg dependencies
RUN wget -qO /tmp/libmp3lame0.deb https://ftp.debian.org/debian/pool/main/l/lame/libmp3lame0_3.100-3_amd64.deb && dpkg -i /tmp/libmp3lame0.deb 
RUN wget -qO /tmp/libxvidcore4.deb http://ftp.us.debian.org/debian/pool/main/x/xvidcore/libxvidcore4_1.3.7-1+b2_amd64.deb && dpkg -i /tmp/libxvidcore4.deb
RUN wget -qO /tmp/libflac8.deb http://security.debian.org/debian-security/pool/updates/main/f/flac/libflac8_1.3.2-3+deb10u3_amd64.deb && dpkg -i /tmp/libflac8.deb

RUN pip install opencv-python pydantic pydantic_settings scikit-learn insightface dependency_injector\
 gst-python-stubs  onnxruntime-gpu tritonclient[http] faiss-gpu pymilvus aiofiles minio httpx redis\
 prometheus-client pika
RUN pip install -e /common/