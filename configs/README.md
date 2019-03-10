# Pythen Environments

## Setup

1. Install miniconda.
1. Update conda to latest.

   ```bash
   sudo /usr/local/miniconda2/bin/conda update -n base -c defaults conda
   ```

1. Remove possible conflicts.

   ```bash
   sudo rm -fr /usr/local/miniconda2/envs/py27
   ```

1. Install env and activate.

   ```bash
   conda env update -f <some-conda.yaml>
   source activate <env-name>
   ```

## Standard Envs

In standard env, we try to avoid any version restrictions.

Such envs are:

* `py27-std` py27-conda-std.yaml
* `py36` py36-conda.yaml

## Envs with Cyber compatibility

Cyber has strong restriction on some libs' versions. You should obey them if you
need to read, write Cyber records, or call Cyber functions.

Such envs are:

* `py27` py27-conda.yaml

## Ongoing efforts

1. py27-conda-std.yaml is in deprecation along with Python 2.7 retiring. Any new
   jobs should NOT depend on it.
1. py27-conda.yaml is in deprecation along with Cyber wrapper upgrade.
   * If the upcoming Python3 wrapper is compatible with the standard env， we'll
     unify everything into py36-conda.yaml.
   * If the upcoming Python3 wrapper still has strong restrictions on lib
     versions, we'll maintain a new py36-conda-cyber.yaml.
