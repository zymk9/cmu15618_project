# Yocto/Noise: Noise functions

Yocto/Noise provides a Perlin noise implementation.
Yocto/Noise is implemented in `yocto_noise.h`.
The noise implementation included here is derived from `stb_noise.h`,
although we expect this to be changed in the next releases.

## Noise functions

Use `perlin_noise(p, w)` to generate Perlin noise with optional wrapping.
Returned values are in the range [0, 1] following the Renderman convention and
to ensure that all noise functions return values in the same range.
For fractal variations, use `perlin_ridge(p, l, g, o, f, w)`,
`perlin_fbm(p, l, g, o, w)` and `perlin_turbulence(p, l, g, o, w)`.
Each fractal version is defined by its lacunarity `l`, its gain `g`, the
number of octaves `o` and possibly an offet.

```cpp
auto p = vec3f{0,0,0};
auto n = perlin_noise(p);
auto lacunarity = 2.0f, gain = 0.5.0f, , offset = 1.0f; auto octaves = 6;
auto n = perlin_ridge(p, lacunarity, gain, octaves, offset);
auto n = perlin_fbm(p, lacunarity, gain, octaves);
auto n = perlin_turbulence(p, lacunarity, gain, octaves);
```
