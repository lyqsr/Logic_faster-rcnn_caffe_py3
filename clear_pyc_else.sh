find . -name '*.pyc' -type f -print -exec rm -rf {} \;
find . -name '*.py~' -type f -print -exec rm -rf {} \;
find . -name '__pycache__' -type d -exec rm -rf {} \;
