# Run detached (-d) so training survives terminal/SSH disconnection.
# View logs:  docker compose -f docker/docker-compose-train-cpu.yml logs -f
# Stop:       docker compose -f docker/docker-compose-train-cpu.yml down
mkdir -p ./datasets   # ensure the volume mount directory exists
docker compose -f docker/docker-compose-train-cpu.yml up --build --force-recreate -d
sudo docker logs yolo_train_cpu -f
