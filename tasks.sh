#!/usr/bin/env bash

function test-build {
  maturin build --features python -o wheels -i $(which python)
}

function test-install {
  test-build
  cd wheels && pip install --force-reinstall -U knn_rs-*.whl && cd ..
}

function release-build {
  maturin build --release --features python -o wheels -i $(which python)
}

function release-install {
  release-build
  cd wheels && pip install --force-reinstall -U knn_rs-*.whl && cd ..
}

function build-run-tests {
  test-install
  pytest tests
}

function test {
  maturin develop
  pytest tests
}

"$@"
