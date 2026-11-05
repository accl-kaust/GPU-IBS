#!/bin/bash

# Function to display help message
display_help() {
    cat << EOF
Usage: $0 <accessions_file> <data_path> <window_size>

This script generates a SLURM job script and submits it for each accession.

Arguments:
  accessions_file File containing accessions for the jobs
  data_path       Path to the data directory on the host
  window_size     Window size for the FastIBS program

Environment (optional):
  GPU_GRES        If set (e.g. GPU_GRES=gpu:a100:1), each job also requests
                  that GPU resource with #SBATCH --gres; run.sh then starts the
                  container with --nv and fastibs takes the GPU path.

Example:  bash $0 accessions.txt ../FastIBSData/ 50000
EOF
}

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    display_help
    exit 1
fi

# Assign arguments to variables
accessions_file=$1
data_path=$2
window_size=$3
GPU_GRES=${GPU_GRES:-}

# Check if the accessions file exists
if [ ! -f "$accessions_file" ]; then
    echo "Error: Accessions file '$accessions_file' does not exist."
    exit 1
fi

# Loop over each accession in the file
while IFS= read -r accession || [ -n "$accession" ]
do
    [ -z "$accession" ] && continue
    # Optional GPU request; carries its own newline so that, when unset, the
    # generated script is byte-identical to the original FastIBS one.
    gres_line=""
    if [ -n "$GPU_GRES" ]; then
        gres_line="#SBATCH --gres=${GPU_GRES}"$'\n'
    fi
    # Generate a SLURM job script
    job_script="job_${accession}.sh"
    cat << EOF > $job_script
#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=FastIBS_${accession}
#SBATCH --output=FastIBS_${accession}.out
#SBATCH --error=FastIBS_${accession}.err
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=50
${gres_line}bash run.sh ${accession} ${data_path} ${window_size}
EOF

    # Submit the job - uncomment below to submit
    #sbatch $job_script
done < "$accessions_file"
