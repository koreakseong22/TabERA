# check_tabzilla_datasets.py
import openml

suite = openml.study.get_suite(379)  # TabZilla Hard Datasets suite ID
print(f"총 task 수: {len(suite.tasks)}")

tasks = openml.tasks.list_tasks(output_format="dataframe")
tabzilla_tasks = tasks[tasks["tid"].isin(suite.tasks)]

for _, row in tabzilla_tasks[["tid", "did", "name"]].iterrows():
    print(f"  task_id={row['tid']}  dataset_id={row['did']}  name={row['name']}")

# import openml
# task = openml.tasks.get_task(3896)
# print(f"task_id={task.task_id}  dataset_id={task.dataset_id}")
# dataset = openml.datasets.get_dataset(task.dataset_id)
# print(f"name={dataset.name}  N={dataset.qualities['NumberOfInstances']}  F={dataset.qualities['NumberOfFeatures']}")