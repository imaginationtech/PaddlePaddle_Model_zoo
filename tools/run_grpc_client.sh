ScriptsL=$(dirname $0)

export PYTHONPATH=${ScriptsL}/../python:$PYTHONPATH

python ${ScriptsL}/../python/engine/backend/pvr_grpc/pvr_infer_test_client.py
