# Evaluate saved Iris predictions

After [restoring the artifacts](<../../4 Trained Models/README.md>), run the following from the repository root to recompute anatomical and topology-control metrics:

```bash
./run.sh --device cpu aggregate \
  "5 Iris/6 Workspace/evaluation/core" \
  --sources "5 Iris/5 Outputs/campaigns/round4-core"
```

The command writes `aggregate.json`, `aggregate.md` and OOF arrays from saved predictions. Each complete OOF evaluation combines the five held-out folds for one partition/optimization setting.

To compare the core and full-graph campaigns, supply both source directories and a separate output:

```bash
./run.sh --device cpu aggregate \
  "5 Iris/6 Workspace/evaluation/core-full" \
  --sources "5 Iris/5 Outputs/campaigns/round4-core" \
            "5 Iris/5 Outputs/campaigns/round4-full"
```

The [aggregation protocol](<../../8 Documentation/repeated_evaluation.md>) defines paired differences and conditional bootstrap intervals. Permutation and transfer campaigns use the separate analyses described in the [Iris protocol](<../../8 Documentation/round4_protocol.md>).

For newly trained campaigns, replace each `--sources` path with its `5 Iris/6 Workspace/retrained/SESSION/round4-<campaign>` directory and choose a new output directory. Use [paper reproduction](<../../7 Paper Results/README.md>) for the final figures and tables.
