#!/bin/bash

sh ./shell/pull_repos.sh

echo "Python lines-of-code counting..."

cd repositories

echo "adanet" >> line_counts.txt && cd adanet && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "autogluon" >> line_counts.txt && cd autogluon && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "Auto-PyTorch" >> line_counts.txt && cd Auto-PyTorch && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "auto-sklearn" >> line_counts.txt && cd auto-sklearn && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "autokeras" >> line_counts.txt && cd autokeras && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "AutoTS" >> line_counts.txt && cd AutoTS && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "etna" >> line_counts.txt && cd etna && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "evalml" >> line_counts.txt && cd evalml && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "FEDOT" >> line_counts.txt && cd FEDOT && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "FLAML" >> line_counts.txt && cd FLAML && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "h2o-3" >> line_counts.txt && cd h2o-3 && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "hyperopt-sklearn" >> line_counts.txt && cd hyperopt-sklearn && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "kats" >> line_counts.txt && cd kats && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "LightAutoML" >> line_counts.txt && cd LightAutoML && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "ludwig" >> line_counts.txt && cd ludwig && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "Meta-AAD" >> line_counts.txt && cd Meta-AAD && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "MetaOD" >> line_counts.txt && cd MetaOD && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "mljar-supervised" >> line_counts.txt && cd mljar-supervised && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "pycaret" >> line_counts.txt && cd pycaret && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "pyodds" >> line_counts.txt && cd pyodds && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd .. && \
echo "tpot" >> line_counts.txt && cd tpot && git ls-files | grep '\.py' | xargs wc -l | grep total >> ../line_counts.txt && cd ..

echo "Line counting finished"
