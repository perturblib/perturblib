.. _installation:

Installation
============


For developers
--------------

First, ensure that you have poetry installed. The recommended way to install poetry is using :code:`pipx`:

.. code-block:: python

    pipx install poetry

For :code:`pipx` installation see `here <https://pipx.pypa.io/stable/installation/>`_. Optionally, if working on HPC,
to manage storage in an efficient manner, ensure caches for different libraries are pointed to your scratch space.
This can be done by setting the environment variables for :code:`perturb-lib`, :code:`poetry`, and :code:`pip` as follows:

.. code-block:: bash

    SCRATCH_DIR=/your/path/to/scratch/space
    export PERTURB_LIB_CACHE_DIR=$SCRATCH_DIR/plib_cache
    export POETRY_CACHE_DIR=$SCRATCH_DIR/poetry_cache
    export PIP_CACHE_DIR=$SCRATCH_DIR/pip_cache


This can be automated by placing the code into :code:`.bash_profile`. Next, install :code:`perturb-lib` dependencies in the development mode. All the requirements are listed in :code:`pyproject.toml`. It is recommended (but not necessary) to install all dependency groups.

.. code-block:: bash

    poetry install --with dev,docs,benchmarking

Installing pre-commit hooks can be done as follows:

.. code-block:: bash

    poetry run pre-commit install

Finally, execute commands through poetry environment Simply use prefix :code:`poetry run` in front of python commands. For example, tests can be performed by running:

.. code-block:: bash

    poetry run pytest

Alternatively, you can first enter poetry environment first using :code:`poetry shell` and then run python commands as usual.
