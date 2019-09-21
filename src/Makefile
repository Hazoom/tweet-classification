all: pylint pycodestyle

pylint:
	pylint src \
		--disable=missing-docstring \
		--disable=too-few-public-methods \
		--disable=eval-used \
		--max-line-length=119 \
		--max-args=8 \
		--good-names=s3,e \
		--ignore-imports=yes \
		--disable=duplicate-code \
		--disable=no-else-return \
		--function-rgx='[a-z_][a-z0-9_]{2,60}$$' \
		--method-rgx='[a-z_][a-z0-9_]{2,60}$$'

pycodestyle:
	pycodestyle *.py --max-line-length=119
