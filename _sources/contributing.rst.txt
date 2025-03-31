============
Contributing
============

We welcome contributions to the city2graph project! This document provides guidelines for contributing to the project.

Setting Up Development Environment
---------------------------------

1. Fork the repository on GitHub.
2. Clone your fork locally:

   .. code-block:: bash

       git clone https://github.com/yourusername/city2graph.git
       cd city2graph

3. Create a conda environment using the provided environment file:

   .. code-block:: bash

       conda env create -f environment.yml
       conda activate city2graph

4. Install the package in development mode:

   .. code-block:: bash

       pip install -e .

5. Install pre-commit hooks:

   .. code-block:: bash

       pip install pre-commit
       pre-commit install

Making Changes
-------------

1. Create a new branch for your changes:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. Make your changes to the codebase.
3. Run the tests to ensure your changes don't break existing functionality:

   .. code-block:: bash

       pytest

4. Update or add documentation as needed.
5. Commit your changes with a descriptive commit message.

Code Style
---------

We follow PEP 8 style guidelines for Python code. Some key points:

* Use 4 spaces for indentation.
* Maximum line length of 88 characters (using Black formatter).
* Use docstrings for all public modules, functions, classes, and methods.
* Use type hints where appropriate.

Documentation
------------

When contributing new features or making significant changes, please update the documentation:

1. Add docstrings to all public functions, classes, and methods.
2. Update the relevant documentation files in the ``docs/source`` directory.
3. If adding a new feature, consider adding an example to ``docs/source/examples.rst``.

Pull Requests
------------

1. Push your changes to your fork:

   .. code-block:: bash

       git push origin feature/your-feature-name

2. Open a pull request on GitHub.
3. Describe your changes in the pull request description.
4. Reference any related issues that your pull request addresses.

Your pull request will be reviewed, and you may be asked to make changes before it's merged.

Building Documentation
--------------------

To build and preview the documentation locally:

1. Create and activate the documentation environment:

   .. code-block:: bash

       conda env create -f docs_environment.yml
       conda activate city2graph_docs

2. Build the documentation:

   .. code-block:: bash

       cd docs
       make html

3. Open ``docs/build/html/index.html`` in your browser to view the documentation.