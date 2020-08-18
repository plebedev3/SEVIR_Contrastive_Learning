source /etc/profile

# Load the anaconda module
module load anaconda/2020a
conda activate rainymotion
# Call your script as you would from the command line, passing in $1 and $2 as arugments
# Note that $1 and $2 are the arguments passed into this script
python map_to_opt_flow.py $1 $2