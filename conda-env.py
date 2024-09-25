import subprocess
import yaml

def export_conda_env():
    # Get conda packages
    conda_result = subprocess.run(['conda', 'list'], capture_output=True, text=True)
    conda_packages = [line.split() for line in conda_result.stdout.split('\n') if not line.startswith('#')]
    
    # Get pip packages
    pip_result = subprocess.run(['pip', 'list', '--format=freeze'], capture_output=True, text=True)
    pip_packages = [line.split('==') for line in pip_result.stdout.split('\n') if line]
    
    env_dict = {
        'name': '.conda',
        'channels': ['conda-forge', 'defaults'],
        'dependencies': []
    }
    
    pip_deps = []
    
    # Process conda packages
    for package in conda_packages:
        if len(package) >= 3:
            name, version, build = package[:3]
            if name == 'python':
                env_dict['dependencies'].insert(0, f'python={version}')
            elif name in ['ipykernel', 'jupyter', 'pip']:
                env_dict['dependencies'].append(name)
    
    # Process pip packages
    for package in pip_packages:
        if len(package) == 2:
            name, version = package
            if name not in ['pip', 'setuptools', 'wheel']:  # Exclude these as they're usually managed by conda
                pip_deps.append(f'{name}=={version}')
    
    if pip_deps:
        env_dict['dependencies'].append({'pip': pip_deps})
    
    # Write to file
    with open('environment.yml', 'w') as f:
        yaml.dump(env_dict, f, default_flow_style=False, sort_keys=False)

export_conda_env()