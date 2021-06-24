#!/usr/bin/env bash

# Exit on first error
set -e

CUR_DIR="$(pwd)"

print_help() {
    echo "usage: build.sh install|test|help"
    echo "  install     install dependencies"
    echo "  lint        run linter"
    echo "  test        run test in each sub module"
    echo "  help        print this message"
    echo ""
    exit 1
}

install() {
  pip install -e . --extra-index-url https://$GEMFURY_TOKEN@pypi.fury.io/cytora/
  pip install -e .[dev] --extra-index-url https://$GEMFURY_TOKEN@pypi.fury.io/cytora/
}

lint() {
  pylint --rcfile=setup.cfg service
}

tests() {
  pytest tests
}

container() {
  docker build --tag universal-resolver:latest --build-arg GEMFURY_TOKEN=$GEMFURY_TOKEN --build-arg AWS_KEY=$AWS_KEY --build-arg AWS_SECRET=$AWS_SECRET --build-arg AWS_S3_BUCKET=$AWS_S3_BUCKET .
}

case "$1" in
    install)
        install
    ;;
    lint)
        lint
    ;;
    test)
        tests
    ;;
    container)
        container
    ;;
    *)
        print_help
    ;;
esac

exit 0
