"""Start a SageMaker managed-spot training job for cloud/sagemaker/train_sft.py.

Usage: uv run --with boto3 python cloud/sagemaker/launch.py <job-name> <data_dir> [key=value ...]
  key=value pairs become hyperparameters (see train_sft.py), plus:
  region=eu-north-1  instance=ml.g6e.xlarge  spot=1  max_hours=8  volume_gb=200
  adapter_s3=s3://.../output/  evaluate a finished job's adapter instead of training
Uploads the data and this folder's code to the account's SageMaker bucket, then creates the job.
Nothing is charged until the job starts; it stops by itself when the script ends.
"""
from __future__ import annotations

import sys
from pathlib import Path

import boto3

TAG = "2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker"
HERE = Path(__file__).parent


def main() -> None:
    job, data_dir = sys.argv[1], Path(sys.argv[2])
    kv = dict(a.split("=", 1) for a in sys.argv[3:])
    region = kv.pop("region", "eu-north-1")
    image = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:{TAG}"
    instance = kv.pop("instance", "ml.g6e.xlarge")
    spot = kv.pop("spot", "1") == "1"
    max_s = int(float(kv.pop("max_hours", "8")) * 3600)
    volume = int(kv.pop("volume_gb", "200"))
    account = boto3.client("sts").get_caller_identity()["Account"]
    bucket = f"sagemaker-{region}-{account}"
    role = f"arn:aws:iam::{account}:role/sagemaker_execution_role"
    s3 = boto3.client("s3", region_name=region)
    if bucket not in [b["Name"] for b in s3.list_buckets()["Buckets"]]:
        cfg = {} if region == "us-east-1" else {
            "CreateBucketConfiguration": {"LocationConstraint": region}}
        s3.create_bucket(Bucket=bucket, **cfg)  # private by default
    base = f"arabic-poet/{job}"
    for f in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        s3.upload_file(str(data_dir / f), bucket, f"{base}/data/{f}")
    for f in ("train_sft.py", "requirements.txt"):
        s3.upload_file(str(HERE / f), bucket, f"{base}/code/{f}")
    for f in sorted((HERE / "_bundle").glob("*")):  # from build_eval_bundle.py, if built
        s3.upload_file(str(f), bucket, f"{base}/code/{f.name}")
    adapter = kv.pop("adapter_s3", None)  # s3://.../output/<job>/output/ holding model.tar.gz

    def channel(name: str) -> dict:
        return {"ChannelName": name, "InputMode": "File", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{bucket}/{base}/{name}/",
            "S3DataDistributionType": "FullyReplicated"}}}

    code = "/opt/ml/input/data/code"
    spec = dict(
        TrainingJobName=job,
        AlgorithmSpecification={
            "TrainingImage": image, "TrainingInputMode": "File",
            "ContainerEntrypoint": ["bash", "-c"],
            "ContainerArguments": [
                f"pip install -q -r {code}/requirements.txt && python -u {code}/train_sft.py"]},
        RoleArn=role,
        InputDataConfig=[channel("data") | {"ChannelName": "train"}, channel("code")] + (
            [channel("adapter") | {"DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": adapter,
                "S3DataDistributionType": "FullyReplicated"}}}] if adapter else []),
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/{base}/output"},
        ResourceConfig={"InstanceType": instance, "InstanceCount": 1, "VolumeSizeInGB": volume},
        HyperParameters=kv,
        CheckpointConfig={"S3Uri": f"s3://{bucket}/{base}/checkpoints",
                          "LocalPath": "/opt/ml/checkpoints"},
        EnableManagedSpotTraining=spot,
        StoppingCondition={"MaxRuntimeInSeconds": max_s} | (
            {"MaxWaitTimeInSeconds": max_s + 6 * 3600} if spot else {}),
    )
    boto3.client("sagemaker", region_name=region).create_training_job(**spec)
    print(f"started {job} on {instance} in {region} (spot={spot}); output s3://{bucket}/{base}/output")


if __name__ == "__main__":
    main()
