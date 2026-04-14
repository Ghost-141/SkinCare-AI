import boto3
import json

# --- Configuration ---
REGION = "us-east-1"
CLUSTER_NAME = "skincare-cluster"
SERVICE_NAME = "skincare-service"
TASK_DEF_FILE = "task-definition.json"

ecs = boto3.client("ecs", region_name=REGION)


def setup_ecs():
    # 1. Load Task Definition from file
    print(f"Reading Task Definition from {TASK_DEF_FILE}...")
    try:
        with open(TASK_DEF_FILE, "r") as f:
            task_def = json.load(f)
    except Exception as e:
        print(f"Error reading {TASK_DEF_FILE}: {e}")
        return

    # 2. Register Task Definition with AWS
    print(f"Registering Task Definition '{task_def['family']}' with AWS...")
    try:
        # We extract only the fields required by register_task_definition
        response = ecs.register_task_definition(
            family=task_def["family"],
            networkMode=task_def.get("networkMode", "bridge"),
            containerDefinitions=task_def["containerDefinitions"],
            requiresCompatibilities=task_def.get("requiresCompatibilities", ["EC2"]),
            cpu=task_def.get("cpu", "256"),
            memory=task_def.get("memory", "512"),
            executionRoleArn=task_def["executionRoleArn"],
            taskRoleArn=task_def["taskRoleArn"],
        )
        task_def_arn = response["taskDefinition"]["taskDefinitionArn"]
        print(f"SUCCESS: Task Definition registered: {task_def_arn}")
    except Exception as e:
        print(f"Error registering task definition: {e}")
        return

    # 3. Create or Update ECS Service
    print(f"Configuring ECS Service: {SERVICE_NAME}...")
    try:
        # Check if service already exists
        services = ecs.describe_services(cluster=CLUSTER_NAME, services=[SERVICE_NAME])

        # If service exists and is not 'INACTIVE'
        if services["services"] and services["services"][0]["status"] != "INACTIVE":
            print(
                f"Service '{SERVICE_NAME}' already exists. Updating to latest task version..."
            )
            ecs.update_service(
                cluster=CLUSTER_NAME,
                service=SERVICE_NAME,
                taskDefinition=task_def["family"],
            )
            print("SUCCESS: ECS Service updated.")
        else:
            print(f"Creating new ECS Service: {SERVICE_NAME}...")
            ecs.create_service(
                cluster=CLUSTER_NAME,
                serviceName=SERVICE_NAME,
                taskDefinition=task_def["family"],
                desiredCount=1,
                launchType="EC2",
                deploymentConfiguration={
                    "maximumPercent": 200,
                    "minimumHealthyPercent": 0,
                },
            )
            print("-" * 30)
            print(f"SUCCESS: ECS Service '{SERVICE_NAME}' created!")
            print(f"AWS is now attempting to run your container on your EC2 instance.")
            print("-" * 30)

        print("\nNEXT STEPS:")
        print(f"1. Go to AWS Console -> ECS -> Cluster: '{CLUSTER_NAME}'")
        print(f"2. Click on Service: '{SERVICE_NAME}' -> Tasks tab.")
        print("3. Wait for the Task status to change to 'RUNNING'.")
        print("4. Push your code to the 'deployment' branch to trigger CI/CD updates.")

    except Exception as e:
        print(f"Error configuring service: {e}")


if __name__ == "__main__":
    setup_ecs()
