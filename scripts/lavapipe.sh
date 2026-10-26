#!/usr/bin/env bash
# Prints the path of the lavapipe (software Vulkan) ICD manifest the GPU tests run on, or fails
# with how to install it. Usage: export VK_ICD_FILENAMES=$(scripts/lavapipe.sh)
# A VK_ICD_FILENAMES that already names a lavapipe manifest wins; otherwise the standard ICD
# directories are searched. The gate never skips GPU tests, so a missing lavapipe is an error.
set -euo pipefail

if [[ -n ${VK_ICD_FILENAMES:-} && $(basename "$VK_ICD_FILENAMES") == lvp_icd*.json && -f $VK_ICD_FILENAMES ]]; then
  echo "$VK_ICD_FILENAMES"
  exit 0
fi

shopt -s nullglob
for icd in /usr/share/vulkan/icd.d/lvp_icd*.json /etc/vulkan/icd.d/lvp_icd*.json; do
  echo "$icd"
  exit 0
done

echo "lavapipe not found: the GPU tests run on it." >&2
echo "install lavapipe: sudo pacman -S vulkan-swrast (Arch) or mesa-vulkan-drivers (Debian/Ubuntu)" >&2
exit 1
