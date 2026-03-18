#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZONES_FILE="${SCRIPT_DIR}/n1_t4_zones.txt"

MACHINE_TYPE="n1-highmem-16"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT="1"
BOOT_DISK_SIZE="256GB"
DATA_DISK_TYPE="pd-ssd"
IMAGE_FAMILY="ubuntu-2404-lts-amd64"
IMAGE_PROJECT="ubuntu-os-cloud"
DATA_DEVICE_NAME="persistent-disk-1"
RETRY_DELAY_SECONDS=15
RESET_WAIT_SECONDS=300

PROJECT=""
VM_NAME=""
BOOT_SNAPSHOT_NAME="ge-ml-training-snapshot-2026-03-17"
DATA_SNAPSHOT_NAME="ge-ml-training-data-snapshot-2026-03-17"
DATA_DISK_NAME=""

OPS_AGENT_SCRIPT='
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
sudo bash add-google-cloud-ops-agent-repo.sh --also-install
'

DATA_DISK_SETUP_SCRIPT='
DISK_DEVICE="/dev/disk/by-id/google-persistent-disk-1"

if [[ ! -b "${DISK_DEVICE}" ]]; then
  echo "Data disk device not present yet; skipping mount setup."
else
  if blkid "${DISK_DEVICE}" &>/dev/null; then
    echo "Data disk already has a filesystem, skipping format"
  else
    echo "Formatting new data disk..."
    mkfs.ext4 -F "${DISK_DEVICE}"
  fi

  mkdir -p /mnt/data

  if ! mountpoint -q /mnt/data; then
    mount "${DISK_DEVICE}" /mnt/data
    echo "Data disk mounted at /mnt/data"
  fi

  if ! grep -q "^${DISK_DEVICE} /mnt/data " /etc/fstab; then
    echo "${DISK_DEVICE} /mnt/data ext4 defaults 0 2" >> /etc/fstab
    echo "Added data disk mount to /etc/fstab"
  fi
fi
'

NVIDIA_DRIVER_SCRIPT='
if test -f /opt/google/cuda-installer; then
  exit 0
fi

mkdir -p /opt/google/cuda-installer
cd /opt/google/cuda-installer/ || exit 1

curl -fSsL -O https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz
python3 cuda_installer.pyz install_cuda
'

usage() {
    cat << EOF
Usage: $0 <vm-name> [options]

Required:
    <vm-name>                         Name of the VM to create

Options:
    --boot-snapshot <snapshot-name>   Override boot snapshot (default: ge-ml-training-snapshot-2026-03-17)
    --data-snapshot <snapshot-name>   Override data snapshot (default: ge-ml-training-data-snapshot-2026-03-17)
    --data-disk-name <disk-name>      Restored data disk name (default: <vm-name>-data)
    --zones-file <path>               Zone list file (default: scripts/n1_t4_zones.txt)
    --delay-seconds <seconds>         Delay between trials (default: 15)
    --project <project-id>            GCP project; defaults to current gcloud config
    --machine-type <type>             Machine type (default: n1-highmem-16)
    --gpu-type <type>                 GPU type (default: nvidia-tesla-t4)
    --gpu-count <count>               GPU count (default: 1)
    -h, --help                        Show this help message

Behavior:
    - Iterates through zones in the zones file forever until one succeeds
    - Waits for each VM creation attempt to succeed or fail before moving on
    - Restores the data snapshot only after a zone successfully creates the GPU VM
    - Attaches the restored data disk, then resets the VM so the startup script can mount it

Examples:
    $0 my-vm
    $0 my-vm --data-snapshot my-fresh-data-snapshot
    $0 my-vm --boot-snapshot my-boot-snapshot --data-snapshot my-data-snapshot
EOF
}

log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '[%s] %s\n' "${timestamp}" "$*"
}

die() {
    log "ERROR: $*"
    exit 1
}

trim() {
    echo "$1" | xargs
}

run_gcloud() {
    if [[ -n "${PROJECT}" ]]; then
        gcloud --project="${PROJECT}" "$@"
    else
        gcloud "$@"
    fi
}

vm_exists_in_zone() {
    local zone="$1"

    run_gcloud compute instances describe "${VM_NAME}" --zone="${zone}" &>/dev/null
}

disk_exists_in_zone() {
    local zone="$1"

    run_gcloud compute disks describe "${DATA_DISK_NAME}" --zone="${zone}" &>/dev/null
}

boot_disk_exists_in_zone() {
    local zone="$1"

    run_gcloud compute disks describe "${VM_NAME}" --zone="${zone}" &>/dev/null
}

disk_attached_in_zone() {
    local zone="$1"
    local attached_disks

    attached_disks="$(run_gcloud compute instances describe "${VM_NAME}" \
        --zone="${zone}" \
        --format="value(disks[].source.basename())" 2>/dev/null || true)"

    grep -Fxq "${DATA_DISK_NAME}" <<< "${attached_disks}"
}

cleanup_zone_resources() {
    local zone="$1"

    if vm_exists_in_zone "${zone}"; then
        log "Deleting VM '${VM_NAME}' in zone '${zone}'..."
        run_gcloud compute instances delete "${VM_NAME}" --zone="${zone}" --quiet || true
    fi

    if disk_exists_in_zone "${zone}"; then
        log "Deleting restored data disk '${DATA_DISK_NAME}' in zone '${zone}'..."
        run_gcloud compute disks delete "${DATA_DISK_NAME}" --zone="${zone}" --quiet || true
    fi

    if boot_disk_exists_in_zone "${zone}"; then
        log "Deleting leftover boot disk '${VM_NAME}' in zone '${zone}'..."
        run_gcloud compute disks delete "${VM_NAME}" --zone="${zone}" --quiet || true
    fi
}

wait_for_running_after_reset() {
    local zone="$1"
    local start_time
    local status

    start_time="$(date +%s)"

    while true; do
        status="$(run_gcloud compute instances describe "${VM_NAME}" \
            --zone="${zone}" \
            --format="value(status)" 2>/dev/null || true)"

        if [[ "${status}" == "RUNNING" ]]; then
            log "VM is back in RUNNING state after reset."
            return 0
        fi

        if (( $(date +%s) - start_time >= RESET_WAIT_SECONDS )); then
            log "Timed out waiting for VM to return to RUNNING after reset."
            return 1
        fi

        sleep 5
    done
}

write_startup_script() {
    local startup_script_path="$1"

    cat > "${startup_script_path}" << EOF
#!/bin/bash
set -e
${OPS_AGENT_SCRIPT}
${DATA_DISK_SETUP_SCRIPT}
${NVIDIA_DRIVER_SCRIPT}
EOF
}

validate_preconditions() {
    local active_account=""
    local effective_project=""

    log "Validating prerequisites..."
    command -v gcloud >/dev/null 2>&1 || die "gcloud CLI is not installed."
    [[ -f "${ZONES_FILE}" ]] || die "Zones file not found: ${ZONES_FILE}"
    [[ -n "${VM_NAME}" ]] || die "VM name is required."
    [[ -n "${BOOT_SNAPSHOT_NAME}" ]] || die "Boot snapshot name is required."
    [[ -n "${DATA_SNAPSHOT_NAME}" ]] || die "Data snapshot name is required."
    [[ -n "${DATA_DISK_NAME}" ]] || die "Internal error: data disk name was not set."

    active_account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -n1)"
    [[ -n "${active_account}" ]] || die "No active gcloud account found. Run 'gcloud auth login' or 'gcloud auth application-default login'."

    if [[ -n "${PROJECT}" ]]; then
        effective_project="${PROJECT}"
    else
        effective_project="$(gcloud config get-value project 2>/dev/null | tr -d '\r')"
    fi
    [[ -n "${effective_project}" && "${effective_project}" != "(unset)" ]] || die "No GCP project is configured. Pass --project or run 'gcloud config set project <project-id>'."

    log "Using gcloud account: ${active_account}"
    log "Using project: ${effective_project}"
    log "Checking boot snapshot '${BOOT_SNAPSHOT_NAME}'..."
    if ! run_gcloud compute snapshots describe "${BOOT_SNAPSHOT_NAME}" >/dev/null 2>&1; then
        die "Boot snapshot '${BOOT_SNAPSHOT_NAME}' was not found or is not accessible."
    fi

    log "Checking data snapshot '${DATA_SNAPSHOT_NAME}'..."
    if ! run_gcloud compute snapshots describe "${DATA_SNAPSHOT_NAME}" >/dev/null 2>&1; then
        die "Data snapshot '${DATA_SNAPSHOT_NAME}' was not found or is not accessible."
    fi

    log "Preflight validation complete."
}

check_for_name_collisions() {
    local zone
    local collision_found=false

    log "Checking for existing resources with the same names..."

    while IFS= read -r raw_zone || [[ -n "${raw_zone}" ]]; do
        zone="$(trim "${raw_zone%%#*}")"
        [[ -z "${zone}" ]] && continue

        if vm_exists_in_zone "${zone}"; then
            log "Existing VM '${VM_NAME}' found in zone '${zone}'."
            collision_found=true
        fi

        if disk_exists_in_zone "${zone}"; then
            log "Existing disk '${DATA_DISK_NAME}' found in zone '${zone}'."
            collision_found=true
        fi

        if boot_disk_exists_in_zone "${zone}"; then
            log "Existing boot disk '${VM_NAME}' found in zone '${zone}'."
            collision_found=true
        fi
    done < "${ZONES_FILE}"

    if [[ "${collision_found}" == "true" ]]; then
        die "Found existing resources with the same names. Choose a new VM name / data disk name or clean them up first."
    fi

    log "No existing resource name collisions found."
}

create_vm_in_zone() {
    local zone="$1"
    local startup_script_path="$2"
    local create_cmd=(
        compute instances create "${VM_NAME}"
        --zone="${zone}"
        --machine-type="${MACHINE_TYPE}"
        --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}"
        --maintenance-policy=TERMINATE
        --metadata-from-file="startup-script=${startup_script_path}"
    )

    if [[ -n "${BOOT_SNAPSHOT_NAME}" ]]; then
        create_cmd+=(
            "--create-disk=name=${VM_NAME},boot=yes,auto-delete=yes,source-snapshot=${BOOT_SNAPSHOT_NAME}"
        )
    else
        create_cmd+=(
            --boot-disk-size="${BOOT_DISK_SIZE}"
            --image-family="${IMAGE_FAMILY}"
            --image-project="${IMAGE_PROJECT}"
        )
    fi

    run_gcloud "${create_cmd[@]}"
}

restore_and_attach_data_disk() {
    local zone="$1"

    log "Restoring data disk '${DATA_DISK_NAME}' from snapshot '${DATA_SNAPSHOT_NAME}' in zone '${zone}'..."
    run_gcloud compute disks create "${DATA_DISK_NAME}" \
        --zone="${zone}" \
        --source-snapshot="${DATA_SNAPSHOT_NAME}" \
        --type="${DATA_DISK_TYPE}"

    log "Attaching restored data disk to VM..."
    run_gcloud compute instances attach-disk "${VM_NAME}" \
        --zone="${zone}" \
        --disk="${DATA_DISK_NAME}" \
        --device-name="${DATA_DEVICE_NAME}"

    log "Resetting VM so the startup script can detect and mount the restored data disk..."
    run_gcloud compute instances reset "${VM_NAME}" --zone="${zone}" --quiet
    wait_for_running_after_reset "${zone}"
}

try_zone() {
    local zone="$1"
    local startup_script_path

    startup_script_path="$(mktemp)"
    write_startup_script "${startup_script_path}"

    log "Trying zone '${zone}'..."

    if ! create_vm_in_zone "${zone}" "${startup_script_path}"; then
        log "Failed to create GPU VM in zone '${zone}'."
        cleanup_zone_resources "${zone}"
        rm -f "${startup_script_path}"
        return 1
    fi

    log "Successfully created GPU VM in zone '${zone}'."

    if restore_and_attach_data_disk "${zone}"; then
        log "Successfully restored and attached data disk in zone '${zone}'."
        rm -f "${startup_script_path}"
        return 0
    fi

    log "Post-create disk restore/attach failed in zone '${zone}'. Cleaning up this attempt."
    cleanup_zone_resources "${zone}"
    rm -f "${startup_script_path}"
    return 1
}

parse_args() {
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi

    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    VM_NAME="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --data-snapshot)
                [[ -n "${2:-}" ]] || die "--data-snapshot requires a value."
                DATA_SNAPSHOT_NAME="$2"
                shift 2
                ;;
            --boot-snapshot)
                [[ -n "${2:-}" ]] || die "--boot-snapshot requires a value."
                BOOT_SNAPSHOT_NAME="$2"
                shift 2
                ;;
            --data-disk-name)
                [[ -n "${2:-}" ]] || die "--data-disk-name requires a value."
                DATA_DISK_NAME="$2"
                shift 2
                ;;
            --zones-file)
                [[ -n "${2:-}" ]] || die "--zones-file requires a value."
                ZONES_FILE="$2"
                shift 2
                ;;
            --delay-seconds)
                [[ -n "${2:-}" ]] || die "--delay-seconds requires a value."
                RETRY_DELAY_SECONDS="$2"
                shift 2
                ;;
            --project)
                [[ -n "${2:-}" ]] || die "--project requires a value."
                PROJECT="$2"
                shift 2
                ;;
            --machine-type)
                [[ -n "${2:-}" ]] || die "--machine-type requires a value."
                MACHINE_TYPE="$2"
                shift 2
                ;;
            --gpu-type)
                [[ -n "${2:-}" ]] || die "--gpu-type requires a value."
                GPU_TYPE="$2"
                shift 2
                ;;
            --gpu-count)
                [[ -n "${2:-}" ]] || die "--gpu-count requires a value."
                GPU_COUNT="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown option: $1"
                ;;
        esac
    done

    if [[ -z "${DATA_DISK_NAME}" ]]; then
        DATA_DISK_NAME="${VM_NAME}-data"
    fi
}

main() {
    local zone

    parse_args "$@"
    log "Starting script initialization for VM '${VM_NAME}'..."
    validate_preconditions
    check_for_name_collisions

    log "Starting GPU zone retry loop for VM '${VM_NAME}'."
    log "Using zones file: ${ZONES_FILE}"
    log "Data snapshot: ${DATA_SNAPSHOT_NAME}"

    if [[ -n "${BOOT_SNAPSHOT_NAME}" ]]; then
        log "Boot snapshot: ${BOOT_SNAPSHOT_NAME}"
    else
        log "Boot disk source: image ${IMAGE_PROJECT}/${IMAGE_FAMILY}"
    fi

    while true; do
        while IFS= read -r raw_zone || [[ -n "${raw_zone}" ]]; do
            zone="$(trim "${raw_zone%%#*}")"
            [[ -z "${zone}" ]] && continue

            if try_zone "${zone}"; then
                log "Success. VM '${VM_NAME}' is ready in zone '${zone}'."
                log "Attached restored data disk: '${DATA_DISK_NAME}'."
                exit 0
            fi

            log "Sleeping for ${RETRY_DELAY_SECONDS} seconds before the next zone trial..."
            sleep "${RETRY_DELAY_SECONDS}"
        done < "${ZONES_FILE}"

        log "Reached the end of the zone list without success. Restarting from the top."
        sleep "${RETRY_DELAY_SECONDS}"
    done
}

main "$@"
