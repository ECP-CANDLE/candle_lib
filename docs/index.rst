.. CANDLE lib documentation master file, created by
   sphinx-quickstart on Feb 9th 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: candle

.. meta::
   :description: candle library for setting up AI and deep-learning cancer problems
   :keywords: cancer deep learning drug response gene


Candle Library Documentation
============================

- The candle library was originally part of `CANDLE Benchmarks <https://github.com/ECP-CANDLE/Benchmarks.git>`_
- It provides various standard methods to allow/enforce consistency across models
  
  - Standardized model definition files and command line interface
  - Interoperability with `CANDLE Supervisor <https://github.com/ECP-CANDLE/Supervisor.git>`_ for automated workflows
  
    - Standard run() method
    - Standard hyperparameters
 
  - Functionality to incorporate standard methods at all stages of the workflow
  
    - data preprocessing, partitioning, filtering
    - noise injection, abstention
    - checkpointing, restart, logging

- Being extended to provide additional functionality required for `IMPROVE <https://jdacs4c-improve.github.io/docs/>`_


.. grid:: 1 1 2 2
    :gutter: 2

    .. grid-item-card:: Getting Started
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/images/icons/tips.svg
        :link: quickstart
        :link-type: doc

        A good place to start for new users

    .. grid-item-card::  Examples
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/images/icons/science.svg
        :link: examples
        :link-type: doc

        A gallery of examples using candle lib

    .. grid-item-card::  Installation
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/images/icons/download.svg
        :link: installation
        :link-type: doc

        Installation instructions for the candle library

    .. grid-item-card::  API
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/images/icons/code.svg
        :link: api
        :link-type: doc

        See the complete candle lib API


.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: For users

    Installation <installation>
    Getting Started <quickstart>
    Usage Examples <examples>
    API Reference <api>
    Tutorials <tutorials>
    .. Cite CANDLE LIB <citation>

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: For developers

    Contributor's Guide <contributing>
    GitHub Issues <https://github.com/ECP-CANDLE/candle_lib/issues>

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Related

    CANDLE Benchmarks <https://github.com/ECP-CANDLE/Benchmarks.git>
    CANDLE Supervisor <https://github.com/ECP-CANDLE/Supervisor.git>
    IMPROVE Singularity <https://github.com/JDACS4C-IMPROVE/Singularity>

--------------------

Supported By
============

.. raw:: html

   <table>
     <tr style="height:80px">
       <td><a href="https://www.energy.gov/science/office-science"><img src="_static/images/logos/DOE_vertical.png" alt="DOE Logo" width="250"/></a></td>
       <td>The U.S. Department of Energy (DOE) Office of Science.</td>
     </tr>
   </table>
