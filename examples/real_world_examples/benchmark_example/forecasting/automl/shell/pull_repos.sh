#!/bin/bash

# Usage: sh ./shell/pull_repos.sh

echo Cloning repositories...

repo_dir=./repositories/

mkdir -p ${repo_dir}

# Clone repository or pull if already existing
if echo Adanet: && cd ${repo_dir}adanet; then git pull && cd ../../; else git clone https://github.com/tensorflow/adanet.git ${repo_dir}adanet; fi
if echo Auto-PyTorch: && cd ${repo_dir}Auto-PyTorch; then git pull && cd ../../; else git clone https://github.com/automl/Auto-PyTorch.git ${repo_dir}Auto-PyTorch; fi
if echo auto-sklearn: && cd ${repo_dir}auto-sklearn; then git pull && cd ../../; else git clone https://github.com/automl/auto-sklearn ${repo_dir}auto-sklearn; fi
if echo AutoGluon: && cd ${repo_dir}autogluon; then git pull && cd ../../; else git clone https://github.com/awslabs/autogluon ${repo_dir}autogluon; fi
if echo AutoKeras: && cd ${repo_dir}autokeras; then git pull && cd ../../; else git clone https://github.com/keras-team/autokeras.git ${repo_dir}autokeras; fi
if echo AutoTS: && cd ${repo_dir}AutoTS; then git pull && cd ../../; else git clone https://github.com/winedarksea/AutoTS ${repo_dir}AutoTS; fi
if echo ETNA: && cd ${repo_dir}etna; then git pull && cd ../../; else git clone https://github.com/tinkoff-ai/etna ${repo_dir}etna; fi
if echo EvalML: && cd ${repo_dir}evalml; then git pull && cd ../../; else git clone https://github.com/alteryx/evalml ${repo_dir}evalml; fi
if echo FEDOT: && cd ${repo_dir}FEDOT; then git pull && cd ../../; else git clone https://github.com/nccr-itmo/FEDOT ${repo_dir}FEDOT; fi
if echo FLAML: && cd ${repo_dir}FLAML; then git pull && cd ../../; else git clone https://github.com/microsoft/FLAML ${repo_dir}FLAML; fi
if echo H2O: && cd ${repo_dir}h2o-3; then git pull && cd ../../; else git clone https://github.com/h2oai/h2o-3 ${repo_dir}h2o-3; fi
if echo hyperopt-sklearn: && cd ${repo_dir}hyperopt-sklearn; then git pull && cd ../../; else git clone https://github.com/hyperopt/hyperopt-sklearn ${repo_dir}hyperopt-sklearn; fi
if echo Kats: && cd ${repo_dir}Kats; then git pull && cd ../../; else git clone https://github.com/facebookresearch/Kats ${repo_dir}Kats; fi
if echo LightAutoML: && cd ${repo_dir}LightAutoML; then git pull && cd ../../; else git clone https://github.com/AILab-MLTools/LightAutoML ${repo_dir}LightAutoML; fi
if echo Ludwig: && cd ${repo_dir}ludwig; then git pull && cd ../../; else git clone https://github.com/ludwig-ai/ludwig ${repo_dir}ludwig; fi
if echo Meta-AAD: && cd ${repo_dir}Meta-AAD; then git pull && cd ../../; else git clone https://github.com/daochenzha/Meta-AAD ${repo_dir}Meta-AAD; fi
if echo MetaOD: && cd ${repo_dir}MetaOD; then git pull && cd ../../; else git clone https://github.com/yzhao062/MetaOD ${repo_dir}MetaOD; fi
if echo MLBox: && cd ${repo_dir}MLBox; then git pull && cd ../../; else git clone https://github.com/AxeldeRomblay/MLBox ${repo_dir}MLBox; fi
if echo mljar: && cd ${repo_dir}mljar-supervised; then git pull && cd ../../; else git clone https://github.com/mljar/mljar-supervised ${repo_dir}mljar-supervised; fi
if echo PyCaret: && cd ${repo_dir}pycaret; then git pull && cd ../../; else git clone https://github.com/pycaret/pycaret ${repo_dir}pycaret; fi
if echo PyODDs: && cd ${repo_dir}pyodds; then git pull && cd ../../; else git clone https://github.com/datamllab/pyodds ${repo_dir}pyodds; fi
if echo TPOT: && cd ${repo_dir}tpot; then git pull && cd ../../; else git clone https://github.com/epistasislab/tpot ${repo_dir}tpot; fi

echo Repositories ready
