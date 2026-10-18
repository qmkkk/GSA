# Integrating a hierarchical network

GSA changes radius-query and 3NN execution. Keep the neural layers, checkpoint, sampling order, normalization, and interpolation arithmetic fixed when measuring its effect.

## QuerySession

Replace each query call with `ball_into` or `nn_into`, and keep a single session around one forward. Allocate int32 ID outputs and FP32 squared-distance outputs on the current CUDA device. Pass original-order support tensors. `examples/query_session.py` demonstrates a radius hit and a shared-prefix 3NN query.

## Fixed-graph adapter

For a validated FlashFPS-style nested hierarchy, `fixed_graph_forward` intercepts an operator binding object with these signatures:

```python
ops.ball_query_wrapper(B, N, Q, radius, K, queries, support, output_ids)
ops.three_nn_wrapper(B, Q, N, queries, support, squared_distances, output_ids)
```

Coordinates are `(1,N,3)` or `(1,Q,3)`, and outputs include the batch dimension. The model must resolve the wrappers through this binding object during the forward.

```python
from gsa import fixed_graph_forward

# First configure the model's sampling policy and reset its sampling cache.
stats = {}
output = fixed_graph_forward(
    model, data, ops=pointnet2_binding,
    model_forward=your_forward_with_sampling_cache_lifetime,
    policy="prefix", verify=True, stats=stats,
)
```

The first radius call uses the original input support and the ordered first sampled level as queries. Subsequent levels use ordered prefixes of that first sampled level with successive ratios of four. A reusable self-radius call populates 32 IDs per row. Later calls using the same complete support and radius can slice the cached query rows and ID columns. Radius-result reuse therefore requires equality of the complete ordered support, even when the index is shared with shorter supports.

`verify=True` checks these coordinate conditions while integrating a new graph. Once the graph's ordering and immutability have been independently established, `verify=False` selects the measured fixed-graph execution policy. `policy="scale"` builds separate indices for support sizes; `cache=False` disables radius-result reuse. The adapter restores the original operator wrappers and frees its trees when the forward returns or raises an exception.

Use one sequential forward at a time: this adapter temporarily replaces wrapper attributes. Configure the FlashFPS sample policy in the model before calling it; the adapter never chooses the sampling rate. For models whose ordered-prefix relationship is not established, use QuerySession's coordinate-value admission checks.

## Validation sequence

1. Run `python tests/check_queries.py` for 94 independent query/state checks.
2. Run `python tests/check_integration.py` for the fixed-graph call chain, API validation, and wrapper cleanup.
3. On each intended full model and dataset, compare the entire FP32 logit tensor and final predictions to the selected arithmetic reference.
4. Warm up each implementation and rotate paired measurement order. Include index construction, cache maintenance, coordinate checks, and freeing in the forward interval.

The packaged records identify the exact FP32 reference for every measurement. A generic upstream build may enable fused arithmetic and have different near-tie outcomes; keep that arithmetic choice explicit when comparing outputs.
