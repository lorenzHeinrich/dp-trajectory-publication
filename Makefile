IMAGE_NAME = trajectory_clustering
CONTAINER_NAME = trajectory_clustering_container
script = main.py

all: run

build:
	docker build -t $(IMAGE_NAME):latest .

run: build
	docker run -it --rm \
		-e DISPLAY=${DISPLAY} \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e MPLCONFIGDIR=/tmp/matplotlib \
		$(IMAGE_NAME):latest \
		python $(script)

clean:
	docker rmi $(IMAGE_NAME):latest

.PHONY: all build run
