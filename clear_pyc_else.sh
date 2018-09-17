echo "hidden folder or file st "
find . -name ".*"
echo "hidden folder or file ed "


echo ""
echo ""


find . -name '*.pyc' -type f -print -exec rm -rf {} \;
find . -name '*.py~' -type f -print -exec rm -rf {} \;
find . -name '__pycache__' -type d -exec rm -rf {} \;


