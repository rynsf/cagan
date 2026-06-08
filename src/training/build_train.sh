#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
sh "$SCRIPT_DIR/../../scripts/build_training.sh"
