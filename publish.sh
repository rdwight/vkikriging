#!/bin/bash
set -e  # Stop script on any failing command

# Publish changes to `vkikriging` to Github (source), and the Python Package
# Index (PyPI).  Latter allows installing with `pip install vkikriging`.

# Format all Python code with `black`
find . -name "*.py" | xargs black --skip-string-normalization

# Update distribution tools
python3 -m pip install --user --upgrade setuptools wheel
python3 -m pip install --user --upgrade twine

# Make distribution - creates directories: build, dist, vkikriging.egg-info
python3 setup.py sdist bdist_wheel

# Upload 
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
# Cleanup dist directories

rm -rf build dist vkikriging.egg-info

# Push changes to Github
git push origin master
