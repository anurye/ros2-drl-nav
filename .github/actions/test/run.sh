#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash

./setup.sh
./build.sh
./test.sh
