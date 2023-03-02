.. currentmodule:: candle

.. _installation:

Installation
============

This installation guide includes only the candle installation instructions.


Installing candle lib from source (Github)
------------------------------------------

Installing candle from source code is a fairly straightforward task, but
doing so should not be necessary for most users. If you `are` interested in
installing candle from source, you will first need to get the latest version
of the code::

    git clone https://github.com/ECP-CANDLE/candle_lib.git
    cd candle_lib
    pip install .


Installing develop version using pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One line installation using pip command::

    pip install git+https://github.com/ECP-CANDLE/candle_lib@develop


Testing candle library source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A candle code base can be tested from the root directory of the source
code repository using the following command (Explicit installation of the
`pytest <https://docs.pytest.org/en/stable/>`_ package may be required, please
see above)::

    python -m pytest
