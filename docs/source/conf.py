"""Copyright (C) 2025  GlaxoSmithKline plc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pathlib
import sys

print(pathlib.Path(__file__).parents[2].resolve())
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",  # nicely render types
    "sphinx.ext.viewcode",  # for show the code next to class/function description
    "nbsphinx",  # for notebook rendering
]

templates_path = ["_templates"]
html_static_path = ["_static"]

add_module_names = False  # in the detailed module description, omit the module
autodoc_default_options = {
    "inherited-members": False,  # do not show inherited methods' documentation
    "show-inheritance": True,  # show the parent of the class
    "exclude-members": "training_step, validation_step, configure_optimizers, forward",  # pytorch-lightning modules
}

html_theme = "pydata_sphinx_theme"
html_context = {"default_mode": "light"}  # light theme
html_theme_options = {
    "navbar_end": ["navbar-icon-links"],
    "logo": {
        "image_light": "../_images/plib_logo.svg",
        "image_dark": "../_images/plib_logo.svg",
    },
}  # omit theme switcher
html_css_files = ["custom.css"]
html_logo = "_images/plib_logo.svg"
html_favicon = "_images/plib_favicon.svg"


nbsphinx_prolog = """
.. raw:: html

    <style>
        div.nbinput.container {
            padding-top: 20px;
        }
        div.nblast.container {
            padding-bottom: 20px;
        }
    </style>
"""
