import yaml
import subprocess

def run_script(script_name, **kwargs):
    # Build the command with script name and parameters
    command = ['python3.9', script_name] + [item for pair in [[f'--{key}', str(value)] for key, value in kwargs.items()] for item in pair]
    # Run the command
    subprocess.run(command)

def get_script_name(cl_framework, experiment_setup):
    # Mapping of configuration to script names
    script_map = {
        ('icarl', 'adapt_one_new_generator'): 'rotating-icarl.py',
        ('icarl', 'adapt_multiple_new_generators'): 'main-icarl.py',
        ('mtsc', 'adapt_one_new_generator'): 'rotating-mtsc.py',
        ('mtsc', 'adapt_multiple_new_generators'): 'main-mtsc.py',
        ('mtmc', 'adapt_one_new_generator'): 'rotating-mtmc.py',
        ('mtmc', 'adapt_multiple_new_generators'): 'main-mtmc.py',
    }
    # Get the script name from the mapping based on the given configuration
    return script_map.get((cl_framework, experiment_setup))

def main(config_path='master_config.yaml'):
    # Load the configuration from the YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    cl_setup = config.get('cl_setup', {})
    parameters = config.get('parameters', {})
    
    # Determine the script to run based on the configuration
    script_to_run = get_script_name(cl_setup.get('cl_framework'), cl_setup.get('experiment_setup'))
    
    if script_to_run:
        # Run the script with the extracted parameters
        print(script_to_run, parameters)
        run_script(script_to_run, **parameters)

    else:
        raise NotImplementedError("The provided configuration does not fall under permissible values.")

if __name__ == '__main__':
    main()
