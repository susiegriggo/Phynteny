
all:
	python3 -m pip install ../genbank/ --user

clean:
	rm -fr build/
	rm -fr dist/
	rm -fr genbank.egg-info/
	python3 -m pip uninstall -y genbank
