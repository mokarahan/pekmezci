docker run -it --rm \
    --gpus all \
    --ipc=host \
    --shm-size=2g \
    -v $(pwd)/../:/workspace \
    -v /DATA:/DATA \
    -v /LOG:/LOG \
    p600-pytorch
