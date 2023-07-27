name: CI

on:
    push:
        branches: [develop]
    pull_request:
        branches: [develop]

    jobs:
        tests:
            name: "Python ${{ matrix.python-version }}"
            runs - on: ${{matrix.os}}

        strategy:
            matrix:
                os: [macos - 12, ubuntu - latest]
                python - version: ["3.8", "3.9", "3.10", "3.11"]

        steps:
        - uses: "actions/checkout@v3"
            with:
                fetch - depth: 0

        # Setup env
        - uses: "actions/setup-python@v3"
            with:
                python - version: "${{ matrix.python-version }}"