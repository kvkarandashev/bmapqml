from bmapqml.dockerfilemaker import prepare_dockerfile
import sys

if len(sys.argv) > 1:
    docker_name = sys.argv[1]
else:
    docker_name = "base_chemxpl"

prepare_dockerfile(docker_name, dockerspec_dir="../../bmapqml/dockerfilemaker")
