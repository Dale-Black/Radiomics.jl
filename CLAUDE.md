# Radiomics.jl - Claude Context

## Project Overview

This is a pure Julia port of [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics), a Python library for extracting radiomic features from medical images.

## Development Method

This project is developed using a **Ralph Loop** - an autonomous agent orchestration system. The loop configuration is in `ralph_loop/`.

## Key Files

- `ralph_loop/prd.json` - Product requirements with all user stories
- `ralph_loop/prompt.md` - Agent instructions (read every iteration)
- `ralph_loop/guardrails.md` - Rules, conventions, settled decisions
- `ralph_loop/progress.md` - Iteration log and research findings
- `ralph_loop/loop.sh` - Bash orchestration script

## Running the Ralph Loop

```bash
cd /Users/daleblack/Documents/dev/julia/Radiomics.jl
./ralph_loop/loop.sh 100  # Run up to 100 iterations
```

## Testing Strategy

Tests use PythonCall.jl to compare Julia implementation against PyRadiomics directly:

```julia
using PythonCall
using CondaPkg  # Manages Python environment with pyradiomics

# In tests, we compare:
julia_result = Radiomics.energy(image, mask)
python_result = pyradiomics_feature("firstorder", "Energy", image, mask)
@test julia_result ≈ python_result rtol=1e-10
```

## Feature Classes to Implement

1. **First Order** (19 features) - Statistical features from intensity histogram
2. **Shape 2D/3D** (30+ features) - Geometric features from ROI shape
3. **GLCM** (24 features) - Gray Level Co-occurrence Matrix texture
4. **GLRLM** (16 features) - Gray Level Run Length Matrix texture
5. **GLSZM** (16 features) - Gray Level Size Zone Matrix texture
6. **NGTDM** (5 features) - Neighboring Gray Tone Difference Matrix
7. **GLDM** (14 features) - Gray Level Dependence Matrix

## Current Status

Check `ralph_loop/prd.json` for current story statuses and `ralph_loop/progress.md` for detailed progress.

## Important Constraints

1. **Pure Julia** - No Python dependencies in final package (only test suite)
2. **1:1 Parity** - Every feature must match PyRadiomics output exactly
3. **Deterministic Tests** - Use seeded random arrays for reproducibility
4. **Git Commits** - All changes must be committed with story ID
