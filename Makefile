# Makefile for the 'resuber' package.
# inspired from https://github.com/xolox/python-humanfriendly/blob/master/Makefile

PACKAGE_NAME=$(python3 setup.py --name)
PACKAGE_VERSION=$(python3 setup.py --version)

default:
	@echo "Makefile for $(PACKAGE_NAME) $(PACKAGE_VERSION)"
	@echo
	@echo 'Usage:'
	@echo
	@echo '    make install    build the package'
	@echo '    make test       run the test suite, report coverage'
	@echo '    make docs       update documentation using Sphinx'
	@echo '    make publish    publish changes to GitHub/PyPI'
	@echo '    make clean      cleanup all temporary files'
	@echo

install:
	@$(MAKE) clean
	@python3 -m pip install -r requirements.txt
	@python3 setup.py sdist bdist_wheel
	@python3 -m pip install -e .

test: 
	@pytest

publish:
	@$(MAKE) install
	@git add resuber/__init__.py
	@git commit resuber/__init__.py -m "Release v${PACKAGE_VERSION}"
	@git tag v${PACKAGE_VERSION}
	@git push origin && git push --tags origin
	@python3 -m pip install twine wheel setuptools
	@python3 -m twine upload dist/*
	@$(MAKE) clean
	@echo "Do not forget to create the release on github!"

publish:
	@git push origin && git push --tags origin
	@python3 -m pip install twine wheel setuptools
	@$(MAKE) install
	@python3 -m twine upload dist/*
	@$(MAKE) clean

clean:
	@rm -Rf *.egg *.egg-info .cache .coverage .tox build dist docs/build htmlcov
	@find -depth -type d -name __pycache__ -exec rm -Rf {} \;
	@find -type f -name '*.pyc' -delete
