# Context-Aware AlphaGen

This package implements the new context-aware training line:

- `ASTGraphEncoder`: AST-GNN based structure encoder using `torch-geometric`
- `ContextAlphaEncoder`: behavior/stat alpha encoder
- `DeepSetsCombiner`: context-aware nonlinear combiner
- `ContextEvaluator`: marginal-contribution reward decomposition
- `ContextAlphaPool`: contextual pool for PPO

## Dependency

```bash
pip install torch-geometric==2.7.0
```

The old baseline line remains usable without importing this package.
