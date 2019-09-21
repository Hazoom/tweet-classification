all: python_build

include dependencies.Makefile

python_build:
	$(MAKE) -C src
