*************
SmartSim API
*************

.. _experiment_api:

Experiment
==========

.. currentmodule:: smartsim.experiment

.. _exp_init:
.. autosummary::

   Experiment.__init__
   Experiment.start
   Experiment.stop
   Experiment.create_ensemble
   Experiment.create_model
   Experiment.create_database
   Experiment.create_run_settings
   Experiment.create_batch_settings
   Experiment.generate
   Experiment.poll
   Experiment.finished
   Experiment.get_status
   Experiment.reconnect_orchestrator
   Experiment.preview
   Experiment.summary
   Experiment.telemetry

.. autoclass:: Experiment
   :show-inheritance:
   :members:


.. _settings-info:

Settings
========

.. currentmodule:: smartsim.settings

Settings are provided to ``Application`` and ``Ensemble`` objects
to provide parameters for how a job should be executed. For
more information, see ``LaunchSettings``


Types of Settings:

.. autosummary::

    RunSettings
    DragonRunSettings

Settings objects can accept a container object that defines a container
runtime, image, and arguments to use for the workload. Below is a list of
supported container runtimes.

Types of Containers:

.. autosummary::

    Singularity


.. _ls_api:

LaunchSettings
-----------


When running SmartSim on laptops and single node workstations,
the base ``LaunchSettings`` object is used to parameterize jobs.
``LaunchSettings`` include a ``run_command`` parameter for local
launches that utilize a parallel launch binary like
``mpirun``, ``mpiexec``, and others.


.. autosummary::

    RunSettings.env_vars
    RunSettings.launch_args
    RunSettings.update_env
    
.. autoclass:: LaunchSettings
    :inherited-members:
    :undoc-members:
    :members:


.. _dragonsettings_api:

DragonRunSettings
-----------------

``DragonRunSettings`` can be used on systems that support Slurm or
PBS, if Dragon is available in the Python environment (see `_dragon_install`
for instructions on how to install it through ``smart``).

``DragonRunSettings`` can be used in interactive sessions (on allcation)
and within batch launches (i.e. ``sbatch`` or ``qsubbatch``,
for Slurm and PBS sessions, respectively).

.. autosummary::
    DragonRunSettings.set_nodes
    DragonRunSettings.set_tasks_per_node

.. autoclass:: DragonRunSettings
    :inherited-members:
    :undoc-members:
    :members:


.. _singularity_api:

Singularity
-----------


``Singularity`` is a type of ``Container`` that can be passed to a
``RunSettings`` class or child class to enable running the workload in a
container.

.. autoclass:: Singularity
    :inherited-members:
    :undoc-members:
    :members:

.. _featurestore_api:

FeatureStore
============

.. currentmodule:: smartsim.database

.. autosummary::

   FeatureStore.__init__
   FeatureStore.fs_identifier
   FeatureStore.num_shards
   FeatureStore.fs_nodes
   FeatureStore.hosts
   FeatureStore.reset_hosts
   FeatureStore.remove_stale_files
   FeatureStore.get_address
   FeatureStore.is_active
   FeatureStore.set_cpus
   FeatureStore.set_walltime
   FeatureStore.set_hosts
   FeatureStore.set_batch_arg
   FeatureStore.set_run_arg
   FeatureStore.enable_checkpoints
   FeatureStore.set_max_memory
   FeatureStore.set_eviction_strategy
   FeatureStore.set_max_clients
   FeatureStore.set_max_message_size
   FeatureStore.set_fs_conf
   FeatureStore.telemetry
   FeatureStore.checkpoint_file
   FeatureStore.batch

FeatureStore
------------

.. _featurestore_api:

.. autoclass:: FeatureStore
   :members:
   :inherited-members:
   :undoc-members:

.. _application_api:

Application
===========

.. currentmodule:: smartsim.entity

.. autosummary::

   Application.__init__
   Application.exe
   Application.exe_args
   Application.file_parameters
   Application.incoming_entities
   Application.key_prefixing_enabled
   Application.add_exe_args
   Application.as_executable_sequence

Application
-----

.. autoclass:: Application
   :members:
   :show-inheritance:
   :inherited-members:

Ensemble
========

.. currentmodule:: smartsim.builders

.. autosummary::

   Ensemble.__init__
   Ensemble.exe
   Ensemble.exe_args
   Ensemble.exe_arg_parameters
   Ensemble.files
   Ensemble.file_parameters
   Ensemble.max_permutations
   Ensemble.permutation_strategy
   Ensemble.replicas
   Ensemble.build_jobs

Ensemble
--------

.. _ensemble_api:

.. autoclass:: Ensemble
   :members:
   :show-inheritance:
   :inherited-members:

.. _ml_api:

Machine Learning
================


SmartSim includes built-in utilities for supporting TensorFlow, Keras, and Pytorch.

.. _smartsim_tf_api:

TensorFlow
----------

SmartSim includes built-in utilities for supporting TensorFlow and Keras in training and inference.

.. currentmodule:: smartsim.ml.tf.utils

.. automodule:: smartsim.ml.tf.utils
    :members:

.. currentmodule:: smartsim.ml.tf

.. autoclass:: StaticDataGenerator
   :show-inheritance:
   :inherited-members:
   :members:

.. autoclass:: DynamicDataGenerator
   :members:
   :show-inheritance:
   :inherited-members:

.. _smartsim_torch_api:

PyTorch
----------

SmartSim includes built-in utilities for supporting PyTorch in training and inference.

.. currentmodule:: smartsim.ml.torch

.. autoclass:: StaticDataGenerator
   :members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: DynamicDataGenerator
   :members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: DataLoader
   :members:
   :show-inheritance:
   :inherited-members:

.. _slurm_module_api:

Slurm
=====

.. currentmodule:: smartsim.wlm.slurm

.. autosummary::

    get_allocation
    release_allocation
    validate
    get_default_partition
    get_hosts
    get_queue
    get_tasks
    get_tasks_per_node

.. automodule:: smartsim.wlm.slurm
    :members:
