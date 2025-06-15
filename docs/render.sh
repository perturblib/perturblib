# call this script from the root folder
rm -rf docs/rendered_website
rm -rf docs/source/api
poetry run sphinx-build -b html docs/source/ docs/rendered_website
