from azureml.core import Workspace
from azureml.exceptions import ComputeTargetException
from azureml.core.compute import ComputeTarget, DatabricksCompute, AmlCompute


def get_compute_aml(
    workspace: Workspace,
    compute_name: str,
    vm_size: str
):
    try:
        if compute_name in workspace.compute_targets:
            compute_target = workspace.compute_targets[compute_name]
            if compute_target and type(compute_target) is AmlCompute:
                print('Found existing compute target ' + compute_name
                      + ' so using it.')
        else:
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size,
                vm_priority="lowpriority",
                min_nodes=int(0),
                max_nodes=int(4),
                idle_seconds_before_scaledown="300"
            )
            compute_target = ComputeTarget.create(workspace, compute_name,
                                                  compute_config)
            compute_target.wait_for_completion(
                show_output=True,
                min_node_count=None,
                timeout_in_minutes=10)
        return compute_target
    except ComputeTargetException as e:
        print(e)
        print('An error occurred trying to provision compute.')
        exit()