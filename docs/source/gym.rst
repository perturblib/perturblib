Gym
===


Intro
-----

:code:`perturb-gym` is a special part of :code:`perturb-lib` specifically designed for easily configurable training of perturbation models.
It is also a convenient tool for logging results and training details, as well as for aggregating results in a readable and processable format.

Training perturbation models using :code:`perturb-gym`
------------------------------------------------------

At the beginning, we enter poetry environment:

.. code-block:: bash

    poetry shell

Training configurations are defined in the form of :code:`.yaml` files.
Predefined ones are given in the :code:`perturb_gym/configs/collection` directory.
Configuration files allow specifying models to be trained, model arguments (hyperparameters), train/validation/test splits, as well as environment parameters such as random seed.
Each configuration file can be potentially used to specify a grid of training configurations.
An example :code:`.yaml` file structure is given below:

.. code-block:: yaml

    environment_configs:
      - seed: 17

    data_configs:
      - training_contexts: "Replogle22_K562"
        val_and_test_perturbations_selected_from: "Replogle22_K562"

    model_configs:
      - model_id: LPM
        model_args:
          optimizer_name: "AdamW"
          learning_rate: 0.002
          learning_rate_decay: 0.996
          num_layers: [1,2,3]
          hidden_dim: 256
          dropout: 0.0
          batch_size: 5000
          embedding_dim: 32
          lightning_trainer_pars:
            max_epochs: 1
        save_model_after_training: True

Training will be performed as specified in the corresponding configuration file.
In this example, 3 different training configurations are defined, where in each one different number of layers will be used.


Training is executed by passing the id of the configuration file to the training script.
For example, if the id of the configuration file is `example2`, the corresponding command
would be:

.. code-block:: bash

    python -m perturb_gym.training train_from_config_file example2

Multiple training configurations that are defined within the training configuration file can also be executed in parallel in case SLURM environment is available.
This can be done as follows:

.. code-block:: bash

    python -m perturb_gym.training train_from_config_file example2 use_slurm=True

If for user the existing collection of configuration files is not sufficient, the user can train perturbation models based on newly defined configuration file.
This can be done as follows

.. code-block:: bash

    python -m perturb_gym.training train_from_config_file "/path/to/yaml/config/file"

Results are by default generated in the cache folder but user can also specify a custom folder as follows:

.. code-block:: bash

    python -m perturb_gym.training train_from_config_file example1 results_dir="/path/to/results/dir"


Evaluating trained models and processing results
------------------------------------------------

Trained models can be (re-)evaluated by running:

.. code-block:: bash

    python -m perturb_gym.evaluation evaluate_all_trained_models "/path/to/dir/with/models"

This command will evaluate all models found in the given directory and all subdirectories.
The following command will process all the results and store them in a convenient :code:`pandas.DataFrame` format.

.. code-block:: bash

    python -m perturb_gym.evaluation process_results "/path/to/dir/with/models"
