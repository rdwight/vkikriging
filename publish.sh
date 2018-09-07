#!/bin/bash
set -e  # Stop script on any failing command

# Publish changes to `vkikriging` to Github (source), and the Python Package
# Index (PyPI).  Latter allows installing with `pip install vkikriging`.

# Format all Python code with ``black``
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

# Make documentation, publish on webpage
cd docs
make html; make html
cp Notes_v3_2018-08.pdf _build/html
ssh rdwight@lamp06.tudelft.nl "rm -rf public_html/vkikriging"
scp -r _build/html rdwight@lamp06.tudelft.nl:public_html/vkikriging
ssh rdwight@lamp06.tudelft.nl "chmod -R 755 public_html"

# Push changes to Github
git push origin master

