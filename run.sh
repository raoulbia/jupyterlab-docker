
docker build --rm -t notebooks -f config/notebooks.Dockerfile .
docker run --rm --privileged -ti -v ${PWD}:/usr/local/bin/notebooks -p 8888:8888 notebooks