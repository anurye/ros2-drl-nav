#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash

./setup.sh

case "$LINTER" in
  "uncrustify"|"cppcheck"|"cpplint")
    if [ -z "$(find src/ -name '*.cpp' -o -name '*.hpp' -o -name '*.c' -o -name '*.h')" ]; then
      echo "No C/C++ files found. Skipping $LINTER."
      exit 0
    fi
    ;;
  "flake8"|"pep257")
    if [ -z "$(find src/ -name '*.py')" ]; then
      echo "No Python files found. Skipping $LINTER."
      exit 0
    fi
    ;;
esac

if [ "$LINTER" = "flake8" ]; then
  ament_flake8 --config ament_flake8.ini src/
else
  ament_${LINTER} src/
fi
