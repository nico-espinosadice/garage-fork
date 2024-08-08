# import time

# # Delay for 2 hours (2 hours * 60 minutes/hour * 60 seconds/minute)
# time.sleep(2 * 60 * 60)

import os
from datetime import datetime
import argparse
import time
import socket
import uuid
import random

import shortuuid
su = shortuuid.ShortUUID()

## ==== SWEEP PARMS ====
SEED = [1, 2, 3, 4, 5] 
# SEED = [1]
ENV = ["antmaze_diverse"]
RESET_PROB = [0.5]
EXPERT_SIZE = [1_000]

SUBOPT_SIZE = [100_000]
# SUBOPT_TREMBLE = [0.05, 0.1, 0.25]
SUBOPT_TREMBLE = [0.25]

# BC_INIT_STEPS = list()
BC_INIT_STEPS = [15_000]
ALGORITHM = ["mm"]
# ALGORITHM = ["bc"]
RESET_TYPE = ["expert_reset"]

ckpts='/home/ne229/garage/garage/'

CMD = "python -u main.py algorithm={algorithm} overrides=model_free_{env} "
CMD += "overrides.subopt_tremble={subopt_tremble} "
CMD += "seed={seed} "
CMD += "overrides.expert_dataset_size={expert_size} "
CMD += "overrides.subopt_dataset_size={subopt_size} " 
CMD += "overrides.wandb_project=garage-{env}-fix-final-reruns "
CMD += "overrides.p_tremble=0.0 "
CMD += "overrides.reset_prob={reset_prob} "
CMD += "overrides.reset_type={reset_type} "
CMD += "overrides.bc_init_steps={bc_init_steps} "

now = datetime.now()
datetimestr = now.strftime("%m%d_%H%M:%S")
slurm_output_dir = f"slurm_output/{datetimestr}"
parser = argparse.ArgumentParser()
parser.add_argument('--base_save_dir', default=f'{os.path.abspath(os.path.join(os.getcwd()))}')
parser.add_argument('--output-dirname', default=slurm_output_dir)
args = parser.parse_args()

jobs = []
for env in ENV:
    for algorithm in ALGORITHM:
        for expert_size in EXPERT_SIZE:
            for bc_init_steps in BC_INIT_STEPS:
                for reset_type in RESET_TYPE:
                    for seed in SEED:
                        for reset_prob in RESET_PROB:
                            # if algorithm in ["filter", "mm", "hybrid_filter"]:
                            #     name = f"{env}|{algorithm}|reset-prob-{reset_prob}|expert-size-{expert_size}|bc-init-steps-{bc_init_steps}|reset-type-{reset_type}|seed-{seed}"
                            #     jobs.append((CMD.format(**locals()), name, None))
                            # else:
                            for subopt_tremble in SUBOPT_TREMBLE:
                                for subopt_size in SUBOPT_SIZE:
                                    name = f"{env}|{algorithm}|subopt-tremble-{subopt_tremble}|reset-prob-{reset_prob}|expert-size-{expert_size}|subopt-size-{subopt_size}|bc-init-steps-{bc_init_steps}|reset-type-{reset_type}|seed-{seed}"
                                    jobs.append((CMD.format(**locals()), name, None))

output_dir = os.path.join(args.base_save_dir, args.output_dirname)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Output Directory: %s" % output_dir)

id_name = uuid.uuid4()
now_name = f'{output_dir}/now_{id_name}.txt'
was_name = f'{output_dir}/was_{id_name}.txt'
log_name = f'{output_dir}/log_{id_name}.txt'
err_name = f'{output_dir}/err_{id_name}.txt'
num_commands = 0
jobs = iter(jobs)
done = False
threshold = 999

while not done:
    for (cmd, name, params) in jobs:

        if os.path.exists(now_name):
            file_logic = 'a'  # append if already exists
        else:
            file_logic = 'w'  # make a new file if not
            print(f'creating new file: {now_name}')

        with open(now_name, 'a') as nowfile,\
             open(was_name, 'a') as wasfile,\
             open(log_name, 'a') as output_namefile,\
             open(err_name, 'a') as error_namefile:

            if nowfile.tell() == 0:
                print(f'a new file or the file was empty: {now_name}')

            now = datetime.now()
            datetimestr = now.strftime("%m%d_%H%M:%S.%f")

            num_commands += 1
            nowfile.write(f'{cmd}\n')
            wasfile.write(f'{cmd}\n')

            output_dir = os.path.join(args.base_save_dir, args.output_dirname)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
            error_namefile.write(f'{(os.path.join(output_dir, name))}.error\n')
            if num_commands == threshold:
                break
    if num_commands != threshold:
        done = True


    # Make a {name}.slurm file in the {output_dir} which defines this job.
    #slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
    start=1
    slurm_script_path = os.path.join(output_dir, f'submit_{start}_{num_commands}.slurm')
    slurm_command = "sbatch %s" % slurm_script_path

    # Make the .slurm file
    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write(f"#SBATCH --array=1-{num_commands}\n")
        slurmfile.write("#SBATCH --output=job_array.log\n")
        slurmfile.write("#SBATCH --error=job_array.error\n")
        slurmfile.write("#SBATCH --requeue\n")
        slurmfile.write("#SBATCH --time=48:00:00\n")

        #### G2
        slurmfile.write("#SBATCH --partition gpu\n")
        slurmfile.write(f"#SBATCH --gres=gpu:1\n")
        slurmfile.write("#SBATCH --exclude=g2-compute-[96-97],ellis-compute-02,tripods-compute-01,coecis-compute-01\n")
        slurmfile.write(f"#SBATCH --mem=20GB\n")
        slurmfile.write("#SBATCH --cpus-per-task=1\n")
        slurmfile.write("#SBATCH --nodes=1\n")

        slurmfile.write("\n")
        slurmfile.write("cd " + args.base_save_dir + '\n')
        slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {log_name} | tail -n 1) --error=$(head -n    $SLURM_ARRAY_TASK_ID {err_name} | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {now_name} | tail -n 1)\n" )
        slurmfile.write("\n")

    os.system("%s &" % slurm_command)

    num_commands = 0
    id_name = uuid.uuid4()
    now_name = f'{output_dir}/now_{id_name}.txt'
    was_name = f'{output_dir}/was_{id_name}.txt'
    log_name = f'{output_dir}/log_{id_name}.txt'
    err_name = f'{output_dir}/err_{id_name}.txt'