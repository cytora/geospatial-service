#!/usr/bin/env bash
set -ex

if [[ -f "env.sh" ]]; then
    source "env.sh"
fi

run-service
