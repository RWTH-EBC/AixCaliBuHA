pip install coverage
cd ..
cd ..
coverage run -m unittest discover --verbose . "test_*.py"
coverage html
htmlcov\index.html