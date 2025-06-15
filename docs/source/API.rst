.. _API:

API
===


Data
----

.. currentmodule:: perturb_lib

.. rubric:: Basic routines
.. autosummary::
   :toctree: api
   :template: functiontemplate.rst

   list_contexts
   describe_context
   register_context
   load_anndata
   load_plibdata
   split_plibdata_2fold
   split_plibdata_3fold

.. rubric:: Data structures
.. autosummary::
   :toctree: api
   :template: classtemplate.rst
   :nosignatures:

   PlibData
   InMemoryPlibData
   OnDiskPlibData

Embeddings
----------

.. autosummary::
   :toctree: api
   :template: functiontemplate.rst

   list_embeddings
   describe_embedding
   register_embedding
   load_embedding

Models
------

.. rubric:: Basic routines
.. autosummary::
   :toctree: api
   :template: functiontemplate.rst

   list_models
   describe_model
   register_model
   load_model
   save_trained_model
   load_trained_model

.. rubric:: Base models
.. autosummary::
   :toctree: api
   :template: classtemplate.rst
   :nosignatures:

   ModelMixin
   SklearnModel


Evaluation
----------

.. rubric:: Basis routines
.. autosummary::
   :toctree: api
   :template: functiontemplate.rst

   list_evaluators
   describe_evaluator
   register_evaluator
   load_evaluator

.. rubric:: Mixin
.. autosummary::
   :toctree: api
   :template: classtemplate.rst
   :nosignatures:

   PlibEvaluatorMixin


Collections
-----------

.. currentmodule:: perturb_lib.models.collection
.. rubric:: Models
.. autosummary::
   :toctree: api
   :template: classtemplate.rst
   :nosignatures:

    baselines.NoPerturb
    baselines.GlobalMean
    baselines.ReadoutMean
    baselines.Catboost
    lpm.LPM


.. currentmodule:: perturb_lib.evaluators.collection
.. rubric:: Evaluators
.. autosummary::
   :toctree: api
   :template: classtemplate.rst
   :nosignatures:

   standard_ones.RMSE
   standard_ones.MAE
   standard_ones.R2
   standard_ones.Pearson
