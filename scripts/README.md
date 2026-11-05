# Scripts: tests, correctness checks and the real-data recipe

## The test suite

`run_tests.sh` is the host-side suite in one command. It builds the regression tests of `tests/`, makes the small data sets they read, runs them, and checks the tools in `bin/` against the original code path, and the GPU path against the CPU path when a GPU is visible. The container image runs it at build time with `docker build --target test .`, and `gpu-ibs selftest` inside the image is the user-facing check (see [`docs/DOCKER.md`](../docs/DOCKER.md)).

## Test data

```bash
bash scripts/make_test_corpus.sh <dir> 25                        # small: 2.2 Mbp reference, 2,202,805 k-mers
bash scripts/make_test_corpus.sh <dir> 25 2000000000 1           # large: 2.0 Gbp, 2,000,199,169 k-mers (KMC_THREADS, KMC_MEM_GB for speed)
bash scripts/make_test_panel.sh <dir> <N> [k] [ref_bp] [seed]    # panel data: one reference, N databases
```

The generators need GNU awk (`AWK=gawk` by default; another awk gives a different data set for the same seed) and `kmc`, and they print the sha256 of each FASTA they write. With GNU Awk 5.3.2 the small set is `test_genome.fasta` sha256 `fd15402f92412fecf69fa77c5c3eb15563b46ec940c4f5202352b8b05deb992d` and `sample_reads.fasta` `1f3553683182a68f9ab5a97f0113aafb6097e47c2a5f7a0778ebb8f27ac531b0`. Compare data sets by these hashes, or by a sorted `kmc_dump`, not by the database files: KMC's `.kmc_pre` and `.kmc_suf` bytes differ from one run to the next on the same FASTA, while the k-mer set and every tool output stay the same.

## Correctness checks on a GPU node

Each of these SLURM jobs runs the tools on the GPU and on the CPU and requires byte-identical output. The `#SBATCH` headers ask only for generic resources; put your partition, account and GPU type on the command line, for example `sbatch -p <partition> --gres=gpu:a100:1 scripts/slurm_phase1.sbatch corpus/db corpus/refs 50000`.

| script | what it checks |
|---|---|
| `slurm_phase1.sbatch <db> <refs> [W]` | the GPU scan against the CPU scan |
| `slurm_phase2.sbatch <db> <refs> [W]` | the streamed GPU scan against the CPU and the synchronous GPU path |
| `slurm_phase3.sbatch <db> <refs> [W]` | every probe configuration (sorted queries, pinned look-up table, hash index); also prints `probe_bench`'s probe rate as a device-health check |
| `slurm_phase4.sbatch <db> <refs> [W] [db2]` | the mapper, the intersection counts, and the index split across several GPUs |
| `slurm_sif_gate.sbatch <db> <refs> [W]` | the container image: GPU inside the image = CPU inside the image = a native build |
| `slurm_multidb_gate.sbatch` | panel mode on the GPU (one and two devices) against single runs; it calls `gate_multidb.sh`, which runs the same check on any machine |
| `ibex/50_release_gate.sbatch` | all of the above in one job on two A100s of the KAUST Ibex cluster, plus a real sample when `BARLEY_DB` and `BARLEY_REFS` name one; `GATE_DIR` sets where its data goes |

`docs/PHASES.md` explains what each phase added and what its check guards.

## Real data

`realdata/00_setup.sh` to `realdata/04_run_2gpu.sbatch` take a public barley run from download to a two-GPU scan: build, download, KMC, a check of which GPU the database fits (`03_preflight.sh`, also behind `gpu-ibs fit` in the image), and the run. [`docs/REALDATA_WALKTHROUGH.md`](../docs/REALDATA_WALKTHROUGH.md) goes through them step by step.

They use the `conda` on `PATH` and read `CONDA_ENV`, the environment with the toolchain and `kmc` that `00_setup.sh` creates (default `gpuibs`).

## The original FastIBS installer

`build_img.sh` and `start_cont.sh` are the helpers of FastIBS's own `install.sh`; [`docs/INSTALL.md`](../docs/INSTALL.md) describes that route.
