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
example,

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



$ python -m pytest test/<test_script_name>.py

Not using ``pytest`` for implementation allows the unit tests to be also run
by using (a number of benefits/conveniences coming from using ``pytest`` can be
seen `here <https://docs.pytest.org/en/7.1.x/how-to/unittest.html#how-to-use-unittest-based-tests-with-pytest>`_
though)::

    $ python -m unittest tests/<test_script_name>.py

Also, all of the test scripts can be run at once with the following command::

    $ pytest test

* Python's unit testing framework, `unittest
  <https://docs.python.org/3/library/unittest.html>`_ is used for implementation of
  the test scripts.

* Reference results (i.e. expected output or ground truth for not all but the most cases)
should not be magic values (i.e. they need to be justified and/or documented).

* Recommended, but not mandatory, implementation approach is as follows:

  - Common data structures, variables and functions,  as well as
    expected outputs, which could be used by multiple test methods throughout
    the test script, are defined either under a base test class or in the very
    beginning of the test script for being used by multiple unit test cases.

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
    We require PRs to pass all of these checks before getting merged in order to
    always ensure our ``main`` branch stability.

These checks can be extremely helpful for contributors to be sure about they are
changing things in correct directions and their PRs are ready to be reviewed and
merged.
