# get-started/control-script.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
import urllib.request
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException

def main():
    # versionar el experimento
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='day1-experiment-train')
    config = ScriptRunConfig(source_directory='./src',
                             script='train.py',
                             compute_target='myMLcluster')

    # Crear un entorno de python propio
    env = Environment(name="myMLenv")
    conda_dep = CondaDependencies()
    conda_dep.add_conda_package("tensorflow")
    conda_dep.add_conda_package("numpy")
    conda_dep.add_conda_package("pandas")
    conda_dep.add_conda_package("scikit-learn")
    conda_dep.add_conda_package("Flask")
    env.python.conda_dependencies=conda_dep
    config.run_config.environment = env

    # Cargar el experimento
    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)
    
    # Registrar el modelo
    model = Model.register(ws, model_name="iris_classification", model_path="./src/model.py")
    
    cluster_name = "myMLcluster"
    
    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    except ComputeTargetException:
        config = AmlCompute.provisioning_configuration(vm_size="STANDARD_DS11_V2",
                                                    vm_priority="lowpriority", 
                                                    min_nodes=1, 
                                                    max_nodes=3)
        compute_target = ComputeTarget.create(workspace=ws, name=cluster_name, provisioning_configuration=config)
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

if __name__ == "__main__":
    main()