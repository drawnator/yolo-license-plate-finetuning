# Run detached (-d) so training survives terminal/SSH disconnection.
# View logs:  docker compose -f docker/docker-compose-train.yml logs -f
# Stop:       docker compose -f docker/docker-compose-train.yml down
mkdir -p ./datasets   # ensure the volume mount directory exists
docker compose -f docker/docker-compose-train.yml up --build --force-recreate -d

