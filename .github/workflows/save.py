name: CI

on: 
  push:
  pull_request:

jobs:
  build:
    docker:
      - image: dolfinx/dolfinx:stable
    steps:
      - checkout
      - run: pip install pytest pytest-cov
      - run:
          name: Run tests
          command: |
            python3 -m pytest test/ --cov festim --cov-report xml --cov-report term
      - run: 
          name: Upload to Codecov
          command: |
            curl -Os https://uploader.codecov.io/latest/linux/codecov

            chmod +x codecov
            ./codecov -t ${{ secrets.CODECOV_TOKEN }}

