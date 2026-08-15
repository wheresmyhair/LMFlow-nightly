#!/usr/bin/env bash
set -euo pipefail

APPWORLD_REVISION="a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
APPWORLD_DATA_VERSION="0.2.0"
APPWORLD_REPOSITORY="https://github.com/StonyBrookNLP/appworld.git"

usage() {
  echo "usage: $0 --python PYTHON --root APPWORLD_ROOT [--source-dir SOURCE_DIR]" >&2
}

python_path=""
appworld_root=""
source_dir="${XDG_CACHE_HOME:-${HOME}/.cache}/lmflow-agent/appworld-${APPWORLD_REVISION:0:7}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      python_path="$2"
      shift 2
      ;;
    --root)
      appworld_root="$2"
      shift 2
      ;;
    --source-dir)
      source_dir="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${python_path}" || -z "${appworld_root}" ]]; then
  usage
  exit 2
fi
if [[ ! -x "${python_path}" ]]; then
  echo "python executable does not exist: ${python_path}" >&2
  exit 2
fi
for command_name in git git-lfs sha256sum; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "required command is unavailable: ${command_name}" >&2
    exit 2
  fi
done

if [[ ! -e "${source_dir}" ]]; then
  mkdir -p "$(dirname "${source_dir}")"
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout "${APPWORLD_REPOSITORY}" "${source_dir}"
fi
if [[ ! -d "${source_dir}/.git" ]]; then
  echo "source directory exists but is not an AppWorld Git checkout: ${source_dir}" >&2
  exit 2
fi

git -C "${source_dir}" fetch --depth 1 origin "${APPWORLD_REVISION}"
git -C "${source_dir}" checkout --detach "${APPWORLD_REVISION}"
git -C "${source_dir}" lfs pull --include="src/appworld/.source/apps.bundle,src/appworld/.source/tests.bundle,generate/.source/data.bundle,generate/.source/tasks.bundle"

actual_revision="$(git -C "${source_dir}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${APPWORLD_REVISION}" ]]; then
  echo "AppWorld revision mismatch: ${actual_revision}" >&2
  exit 1
fi

(
  cd "${source_dir}"
  sha256sum --check --strict <<'EOF'
88d21fc526c1655bb3eee4adfca78ccac793921e4506f28f734ecdb19af77a62  src/appworld/.source/apps.bundle
04aa898cb015c53468c355d5ded662757c5234835a4de5cf4f7d7947bef159ec  src/appworld/.source/tests.bundle
42c2a3c929c60cc891c94e8924d8ce39c2ffd779bd51ca819b26944483de6106  generate/.source/data.bundle
471f225cf4d85db1ba61d24e2fef881afb68f4f9ca0b8bf76cde712745d60433  generate/.source/tasks.bundle
EOF
)

uv_path="$(command -v uv || true)"
if [[ -z "${uv_path}" ]]; then
  echo "uv is required to install the pinned source checkout" >&2
  exit 2
fi
"${uv_path}" pip install \
  --python "${python_path}" \
  --no-deps \
  --no-build-isolation \
  "${source_dir}"

appworld_cli="$(dirname "${python_path}")/appworld"
if [[ ! -x "${appworld_cli}" ]]; then
  echo "AppWorld console command was not installed next to ${python_path}" >&2
  exit 1
fi

PATH="$(dirname "${python_path}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "${appworld_cli}" install
mkdir -p "${appworld_root}"
PATH="$(dirname "${python_path}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "${appworld_cli}" download data \
  --version "${APPWORLD_DATA_VERSION}" \
  --mode minimal \
  --without-setup \
  --root "${appworld_root}"

actual_data_version="$(tr -d '[:space:]' < "${appworld_root}/data/version.txt")"
if [[ "${actual_data_version}" != "${APPWORLD_DATA_VERSION}" ]]; then
  echo "AppWorld data version mismatch: ${actual_data_version}" >&2
  exit 1
fi
echo "AppWorld ${APPWORLD_REVISION} with data ${APPWORLD_DATA_VERSION} is ready."
echo "APPWORLD_SOURCE=${source_dir}"
echo "APPWORLD_ROOT=${appworld_root}"
