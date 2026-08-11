# Run detached (-d) so training survives terminal/SSH disconnection.
# View logs:  docker compose -f docker/docker-compose-train.yml logs -f
# Stop:       docker compose -f docker/docker-compose-train.yml down
# Ensure volume mount directories exist on host so Docker doesn't
# create them as root-owned.
mkdir -p ./datasets
mkdir -p ./runs
mkdir -p ./angelicam
mkdir -p ./datasets/label_sets

docker compose -f docker/docker-compose-train.yml up --build --force-recreate -d

