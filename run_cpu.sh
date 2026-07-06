# Run detached (-d) so training survives terminal/SSH disconnection.
# View logs:  docker compose -f docker/docker-compose-train-cpu.yml logs -f
# Stop:       docker compose -f docker/docker-compose-train-cpu.yml down
docker compose -f docker/docker-compose-train-cpu.yml up --build --force-recreate -d
