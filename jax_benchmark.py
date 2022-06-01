import time

import jax
import jax.numpy as jnp


trajectories_count = 1000
trajectories_points_count = 20
reference_segments_count = 20

key = jax.random.PRNGKey(0)
grid = jax.random.normal(key, shape=(10, 10))

trajectories = jax.random.normal(key, (trajectories_count * trajectories_points_count, 1, 2))

segments = jax.random.normal(key, (reference_segments_count, 2))

trajectories_tiled = jnp.reshape(jnp.tile(trajectories, (1, reference_segments_count, 1)), (trajectories_count * trajectories_points_count * reference_segments_count, 2))

segments_tiled = jnp.reshape(jnp.tile(segments, (trajectories_count * trajectories_points_count, 1, 1)), (trajectories_count * trajectories_points_count * reference_segments_count, 2))

def dist_fn(point, segment):
  return jnp.linalg.norm(point-segment)

vj_dist_fn = jax.jit(jax.vmap(dist_fn))

for i in range(5):

    tic = time.time()

    d = vj_dist_fn(trajectories_tiled, segments_tiled)
    d.block_until_ready()

    opt_time = time.time() - tic

    print("optimization time {:0.6f} sec".format(opt_time))

for i in range(5):

    tic = time.time()

    for i in range(trajectories_count):
      for j in range(trajectories_points_count):
        point = trajectories[i * j]
        for k in range(reference_segments_count):
          segment = segments[k]
          dist_fn(point, segment)

    opt_time = time.time() - tic

    print("optimization time {:0.6f} sec".format(opt_time))