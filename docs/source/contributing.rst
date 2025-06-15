.. _contributing:

Contributing
============

Authors
-------
* Ðorđe Miladinović (djordje.x.miladinovic@gsk.com)
* Andreas Georgiou (andreas.x.georgiou@gsk.com)

Other contributors
------------------
* Tobias Höppe (tobias.x.hoeppe@gsk.com)
* Lachlan Stuart (lachlan.n.stuart@gsk.com)

General guidelines
------------------

Thank you for your interest in :code:`perturb-lib`! There are multiple ways to contribute:

- Address an outstanding issue:
    * Specify an issue from the list of issues.
    * Lay out the implementation plan in that issue.
    * Code up and submit the PR.

- Add a new feature:
    * Create an issue for the intended feature.
    * Start a discussion on the necessity of that issue until there is a team-level agreement.
    * Lay out the implementation plan in that issue.
    * Code up and submit the PR.

- Add a new module:
    * See module-specific guidelines further below.
    * Understand existing template for new module registration.
    * Code up and submit the PR.

Getting started
---------------

To install :code:`perturb-lib` in the development mode, follow the instructions given in :ref:`installation`.

Coding style
------------

:code:`perturb-lib` uses `Google style <https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html>`_ for formatting docstrings. All code is formatted via :code:`ruff` as specified in :code:`pyproject.toml`.

Adding new modules
------------------

New modules can be datasets, models, etc., essentially a new member of one of the existing collections. Before adding any kind of new module, please refer to the source code to understand the registration-based
template for registering new modules to :code:`perturb-lib`. Independently of the module, one must ensure that the new module
is well documented. The documentation should be rendered too as described at :ref:`render-docs`.


Executing unit tests
--------------------

All tests can be executed as follows:

.. code-block:: bash

    poetry run pytest


.. _render-docs:

Rendering documentation
-----------------------

.. code-block:: bash

    source docs/render.sh

Before submitting the PR, you can verify the looks by opening :code:`docs/rendered_website/index.html` locally.
