.. currentmodule:: candle

.. _contributing:

*******************
Contributor's Guide
*******************

Welcome to the Contributor's Guide for candle!

Welcome to the team! If you are reading this document, we hope that
you are already or soon-to-be a candle library contributor, please keep reading!

1. Overview
===========

The CANDLE library enables users to run their own deep learning code on a set of
supported high-performance computers. With the current version of CANDLE,
users will be able to run a hyperparameter optimization task (mlrMBO workflow)
or a parallel execution task (upf workflow).

We invite other community members to become part of this
collaboration at any level of contribution.

1.1. Many Ways to Contribute
----------------------------

There are many different ways to contribute to candle_lib. Anyone can, for
example:

* Adding new functionality
  
  *  python function <fname> in candle/ folder
  *  Add the this file in docs/api_<fname>/index.rst
  *  Specify the API functions for inclusion in ReadTheDocs (RTD)

* Extending functionality of existing modules (say candle/file_utils.py)
  
  * Add the API functions in the corresponding index.rst file (eg. docs/api_file_utils/index.py)

* Write or revise documentation (including this document)

  * Documentation can be built locally by run running 'make html' from the docs folder
  * Python dependencies can for this documentation are listed in docs/NOTES.txt
  * After building documentation can be visualized locally using any browser from file docs/_build/html/index.html
  
* Develop example notebooks that demonstrate how a particular function
  is used
* Answer a support question
* Request a feature or report a bug

All of these activities are signicant contributions to the on-going
development and maintenance of candle lib.



1.2 Getting Started with GitHub and Git
----------------------------------------

Contributing to candle_lib requires using GitHub, and contributing to a GitHub
repository follows almost the same process by any open source Python project
maintained on GitHub. However, it can still seem complex and somewhat varied from
one project to another. As such, we will refer the reader to comprehensive
resources for basic learning and detailed information about GitHub (such as the
`Getting Started with Github <https://docs.github.com/en/get-started>`_ guide).


  - Assertions are used for testing various cases such as array comparison.

  - Please see previously implemented test cases for reference of the
    recommended testing approach

.. attention::
    Our test suite that includes all the unit tests is executed automatically
    for PRs with the help of GitHub Actions workflows to ensure new code passes
    tests. Hence, please check `3.7.4.2. GitHub Actions checks`_ to make sure your
    PR tests are all passing before asking others to review your work.

1.3. GitHub Actions checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

candle_lib employs a number of GitHub Actions workflows (please refer to the `GitHub
Actions <https://docs.github.com/en/actions>`_ guide for detailed information) to make
sure our PRs, branches, etc. pass certain test scenarios such as the pre-commit hooks,
code test suite, and documentation generation. The pre-commit hooks workflow ensures
the code being proposed to be merged is complying with code standards.

.. note::
    All PRs must be made to the ``develop`` branch.
    We require PRs to pass all of these checks before getting merged in order to
    always ensure our ``develop`` branch stability.

These checks can be extremely helpful for contributors to be sure about they are
changing things in correct directions and their PRs are ready to be reviewed and
merged.
