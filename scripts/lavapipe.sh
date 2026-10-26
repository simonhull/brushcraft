#!/usr/bin/env bash
# Prints the path of the lavapipe (software Vulkan) ICD manifest the GPU tests run on, or fails
# with how to install it. Usage: export VK_ICD_FILENAMES=$(scripts/lavapipe.sh)
# A VK_ICD_FILENAMES that already names a lavapipe manifest for this machine wins. Otherwise the
# standard ICD directories are searched for the manifest of this machine's architecture
# (lvp_icd.x86_64.json on Debian and Ubuntu), then the unsuffixed one (Arch's lvp_icd.json);
# another architecture's (the 32-bit lvp_icd.i686.json) is never taken. The gate never skips GPU
# tests, so a missing lavapipe is an error.
set -euo pipefail

arch=$(uname -m)
dirs=(/usr/share/vulkan/icd.d /etc/vulkan/icd.d)

# Whether $1 is a lavapipe manifest this machine can load: its own architecture's, or unsuffixed.
usable() {
  local name
  name=$(basename "$1")
  [[ -f $1 && ($name == "lvp_icd.$arch.json" || $name == lvp_icd.json) ]]
}

if [[ -n ${VK_ICD_FILENAMES:-} ]] && usable "$VK_ICD_FILENAMES"; then
  echo "$VK_ICD_FILENAMES"
  exit 0
fi

for name in "lvp_icd.$arch.json" lvp_icd.json; do
  for dir in "${dirs[@]}"; do
    if [[ -f $dir/$name ]]; then
      echo "$dir/$name"
      exit 0
    fi
  done
done

echo "lavapipe not found for $arch: the GPU tests run on it." >&2
echo "install lavapipe: sudo pacman -S vulkan-swrast (Arch) or mesa-vulkan-drivers (Debian/Ubuntu)" >&2
exit 1
