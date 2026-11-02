#!/usr/bin/env bash
# Fetches the ann-benchmarks HDF5 files into data/ and converts them to the
# flat .annb layout the C++ harness memory-maps. The HDF5 originals stay in
# data/ (gitignored); a partial download resumes instead of restarting.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${repo_root}/data"
base_url="https://ann-benchmarks.com"
python_bin="${PYTHON:-${repo_root}/.venv/bin/python}"

# Sizes are checked because the site serves no published checksums, and a
# truncated HDF5 file otherwise fails much later with an opaque h5py error.
datasets=(
  "glove-25-angular 127359688"
  "sift-128-euclidean 525128288"
)

selected=("$@")
mkdir -p "${data_dir}"

for entry in "${datasets[@]}"; do
  read -r name size <<<"${entry}"
  if [[ ${#selected[@]} -gt 0 && ! " ${selected[*]} " == *" ${name} "* ]]; then
    continue
  fi
  hdf5="${data_dir}/${name}.hdf5"
  if [[ ! -f "${hdf5}" || "$(wc -c <"${hdf5}" | tr -d ' ')" != "${size}" ]]; then
    echo "downloading ${name}"
    curl --fail --location --continue-at - --output "${hdf5}" "${base_url}/${name}.hdf5"
  fi
  actual="$(wc -c <"${hdf5}" | tr -d ' ')"
  if [[ "${actual}" != "${size}" ]]; then
    echo "error: ${hdf5} is ${actual} bytes, expected ${size}" >&2
    exit 1
  fi
  "${python_bin}" -I "${repo_root}/scripts/hdf5_to_annb.py" "${hdf5}" "${data_dir}/${name}.annb"
done
