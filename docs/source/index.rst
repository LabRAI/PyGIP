.. raw:: html

   <div style="margin-top: 50px; text-align: center;">
     <img src="_static/icon.png" alt="PyGIP Icon" style="width: 600px; height: auto;">
   </div>

.. image:: https://img.shields.io/pypi/v/PyGIP
   :target: https://pypi.org/project/PyGIP
   :alt: PyPI Version

.. image:: https://img.shields.io/github/actions/workflow/status/LabRAI/PyGIP/docs.yml
   :target: https://github.com/LabRAI/PyGIP/actions
   :alt: Build Status

.. image:: https://img.shields.io/github/license/LabRAI/PyGIP.svg
   :target: https://github.com/LabRAI/PyGIP/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/pypi/dm/pygip
   :target: https://github.com/LabRAI/PyGIP
   :alt: PyPI Downloads

.. image:: https://img.shields.io/github/issues/LabRAI/PyGIP
   :target: https://github.com/LabRAI/PyGIP
   :alt: Issues

.. image:: https://img.shields.io/github/issues-pr/LabRAI/PyGIP
   :target: https://github.com/LabRAI/PyGIP
   :alt: Pull Requests

.. image:: https://img.shields.io/github/stars/LabRAI/PyGIP
   :target: https://github.com/LabRAI/PyGIP
   :alt: Stars

.. image:: https://img.shields.io/github/forks/LabRAI/PyGIP
   :target: https://github.com/LabRAI/PyGIP
   :alt: GitHub forks

.. image:: _static/github.svg
   :target: https://github.com/LabRAI/PyGIP
   :alt: GitHub

----

**PyGIP** is a comprehensive Python library focused on model extraction attacks and defenses in Graph Neural Networks (GNNs). Built on PyTorch, PyTorch Geometric, and DGL, the library offers a robust framework for understanding, implementing, and defending against attacks targeting GNN models.

**PyGIP is featured for:**

- **Extensive Attack Implementations**: Multiple strategies for GNN model extraction attacks, including fidelity and accuracy evaluation.
- **Defensive Techniques**: Tools for creating robust defense mechanisms, such as watermarking graphs and inserting synthetic nodes.
- **Unified API**: Intuitive APIs for both attacks and defenses.
- **Integration with PyTorch/DGL**: Seamlessly integrates with PyTorch Geometric and DGL for scalable graph processing.
- **Customizable**: Supports user-defined attack and defense configurations.

**Quick Start Example:**

Model Extraction Attack Example with 5 Lines of Code:

.. code-block:: python

    from datasets import Cora
    from models.attack import ModelExtractionAttack0

    # Load the Cora dataset
    dataset = Cora()

    # Initialize the attack with a sampling ratio of 0.25
    mea = ModelExtractionAttack0(dataset, 0.25)

    # Execute the attack
    mea.attack()

Attack Modules
---------------

.. list-table::
   :header-rows: 1

   * - Class Name
     - Reference

   * - :doc:`MEA <_autosummary/attack/pygip.models.attack.mea.MEA>`
     - Wu, Bang, et al. "Model extraction attacks on graph neural networks: Taxonomy and realisation." Proceedings of the 2022 ACM on Asia conference on computer and communications security. 2022.

   * - :doc:`AdvMEA <_autosummary/attack/pygip.models.attack.AdvMEA>`
     - DeFazio, David, and Arti Ramesh. "Adversarial model extraction on graph neural networks." arXiv preprint arXiv:1912.07721 (2019).

   * - :doc:`CEGA <_autosummary/attack/pygip.models.attack.CEGA>`
     - Wang, Zebin, et al. "CEGA: A Cost-Effective Approach for Graph-Based Model Extraction and Acquisition." arXiv preprint arXiv:2506.17709 (2025).

   * - :doc:`DataFreeMEA <_autosummary/attack/pygip.models.attack.DataFreeMEA>`
     - Zhuang, Yuanxin, et al. "Unveiling the Secrets without Data: Can Graph Neural Networks Be Exploited through {Data-Free} Model Extraction Attacks?." 33rd USENIX Security Symposium (USENIX Security 24). 2024.

   * - :doc:`Realistic <_autosummary/attack/pygip.models.attack.Realistic>`
     - Guan, Faqian, et al. "A realistic model extraction attack against graph neural networks." Knowledge-Based Systems 300 (2024): 112144.


Defense Modules
---------------

.. list-table::
   :header-rows: 1

   * - Class Name
     - Reference

   * - :doc:`RandomWM <_autosummary/defense/pygip.models.defense.RandomWM>`
     - Zhao, Xiangyu, Hanzhou Wu, and Xinpeng Zhang. "Watermarking graph neural networks by random graphs." 2021 9th International Symposium on Digital Forensics and Security (ISDFS). IEEE, 2021.

   * - :doc:`BackdoorWM <_autosummary/defense/pygip.models.defense.BackdoorWM>`
     - Xu, Jing, et al. "Watermarking graph neural networks based on backdoor attacks." 2023 IEEE 8th European Symposium on Security and Privacy (EuroS&P). IEEE, 2023.

   * - :doc:`SurviveWM <_autosummary/defense/pygip.models.defense.SurviveWM>`
     - Wang, Haiming, et al. "Making Watermark Survive Model Extraction Attacks in Graph Neural Networks." ICC 2023-IEEE International Conference on Communications. IEEE, 2023.

   * - :doc:`ImperceptibleWM <_autosummary/defense/pygip.models.defense.ImperceptibleWM>`
     - Zhang, Linji, et al. "An imperceptible and owner-unique watermarking method for graph neural networks." Proceedings of the ACM Turing Award Celebration Conference-China 2024. 2024.

   * - :doc:`ATOM <_autosummary/defense/pygip.models.defense.atom.ATOM>`
     - Cheng, Zhan, et al. "Atom: A framework of detecting query-based model extraction attacks for graph neural networks." Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 2025.

   * - :doc:`Integrity <_autosummary/defense/pygip.models.defense.Integrity>`
     - Wu, Bang, et al. "Securing graph neural networks in mlaas: A comprehensive realization of query-based integrity verification." 2024 IEEE Symposium on Security and Privacy (SP). IEEE, 2024.


How to Cite
-----------

If you find it useful, please considering cite the following work:

.. code-block:: bibtex

   @article{li2025intellectual,
      title={Intellectual Property in Graph-Based Machine Learning as a Service: Attacks and Defenses},
      author={Li, Lincan and Shen, Bolin and Zhao, Chenxi and Sun, Yuxiang and Zhao, Kaixiang and Pan, Shirui and Dong, Yushun},
      journal={arXiv preprint arXiv:2508.19641},
      year={2025}
    }


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quick_start
   benchmark

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   pygip_datasets
   pygip_models_attack
   pygip_models_defense
   pygip_utils

.. toctree::
   :maxdepth: 2
   :caption: Additional Information
   :hidden:

   implementation
   cite
   team


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
