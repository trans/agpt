# Experiment Template

Copy this into `rnd/<experiment-name>/README.md` when starting a new experiment
or backfilling an old one.

```md
# <Experiment Title>

**Status**: active | closed | incomplete | log-only | archived

**Code**: branch `<branch-name>`; key commits: `<hash>`, `<hash>`

## Hypothesis

What are we testing? What should improve, fail, or become clearer if this idea
is right?

## Scope

What is in scope for this experiment, and what is explicitly out of scope?

## Setup

- corpus / dataset:
- model / trainer config:
- important flags / env vars:
- output location:

## Reproduce

If the recipe is short, include it here. Otherwise point to `reproduce.sh`,
`run.sh`, or named scripts in this directory.

```sh
# commands here
```

## Artifacts

List what is committed versus regenerated:

- kept in git:
- ignored / regenerated:
- canonical artifact, if any:

## Results

Summarize the important numbers, observations, or failure modes.

## Conclusion

State the conclusion plainly:

- did it help?
- did it fail?
- is it inconclusive?
- what should happen next?
```

## Notes

- `README.md` is required.
- `findings.md` is useful when the conclusion becomes too large for the README.
- Prefer small summary artifacts over committed bulk outputs.
